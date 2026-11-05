#!/usr/bin/env python3
"""OIBC post-hoc correction experiments; no LightGBM refitting.

A: site-specific normalized residual mean+variance (empirical-Bayes shrinkage).
B: observation-station grouped Ridge regression; independent WA/WB station models, averaged.
C: site-specific regularized linear-regression residual correction.

Use the out-of-sample GROUP_OPT 20% of the saved 60/20/20 baseline run to
FIT the correction and to optimize groups. The HOLDOUT 20% is NEVER used to
fit the correction or optimize groups. Predictions from the model-train 60%
are NOT used in posthoc fit, avoiding in-sample baseline residuals.

Default source: output/validation_602020/split602020_baseline/
Other possible source: --source-run split602020_l1_loss

Place all three posthoc_*.py scripts in data/codes/, alongside
baseline_optimizer.py. A is the shared helper imported by B and C.

No test data are read; all competition data processing remains local.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent if HERE.name == "codes" else HERE
METHODS = ("stats", "observation_points", "lr_model")
REG_FEATURES = (
    "pred_cf", "pred_cf_squared", "solar_elev_scaled", "pred_x_elev",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
)
MIN_REGRESSION_ROWS = 80
RIDGE_ALPHA = 100.0


def log(s: str) -> None:
    print(f"[posthoc] {s}", flush=True)


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required code: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_kst(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series)
    return s.dt.tz_localize("Asia/Seoul") if s.dt.tz is None else s.dt.tz_convert("Asia/Seoul")


def delivery_day(series: pd.Series) -> pd.Series:
    return (series - pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%d")


def read_predictions(path: Path, sites: pd.DataFrame) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved 60/20/20 baseline predictions are missing: {path}\n"
            "First run your existing codes/baseline_validation.py."
        )
    pred = pd.read_parquet(path, columns=["site_id", "ts", "pred_kwh"])
    pred["ts"] = to_kst(pred.ts)
    if pred.duplicated(["site_id", "ts"]).any():
        raise ValueError(f"Duplicate predictions in {path}")
    if not pred.site_id.isin(sites.site_id).all():
        raise ValueError(f"Unknown site_id in {path}")
    if not np.isfinite(pred.pred_kwh).all() or (pred.pred_kwh < 0).any():
        raise ValueError(f"Invalid predictions in {path}")
    return pred


def prepare_data(root: Path, source_run: str, base) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, pd.DataFrame]:
    source = root / "output" / "validation_602020" / source_run
    split_path = source / "split_days.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing 60/20/20 day split: {split_path}")
    split = pd.read_csv(split_path, dtype={"day_kst": str, "split": str})
    if split.day_kst.duplicated().any() or not {"model_train", "group_opt", "holdout"}.issubset(set(split.split)):
        raise ValueError("Source split_days.csv is invalid")
    opt_days = set(split.loc[split.split == "group_opt", "day_kst"])
    hold_days = set(split.loc[split.split == "holdout", "day_kst"])
    fit_days = set(split.loc[split.split == "model_train", "day_kst"])
    if (opt_days & hold_days) or (opt_days & fit_days) or (fit_days & hold_days):
        raise ValueError("Source split contains overlapping dates")

    sites = pd.read_csv(root / "sites.csv")[
        ["site_id", "current_group", "capacity_kw", "lat", "lon", "wa_station", "wb_station"]
    ]
    if len(sites) != 253 or sites.site_id.duplicated().any():
        raise ValueError("Expected 253 unique sites")
    if sites[["lat", "lon", "capacity_kw", "wa_station", "wb_station"]].isna().any().any():
        raise ValueError("Missing station/capacity/coordinate values")
    if not np.isfinite(sites.capacity_kw).all() or (sites.capacity_kw <= 0).any():
        raise ValueError("Invalid site capacities")

    gp = read_predictions(source / "group_opt_site_predictions.parquet", sites)
    hp = read_predictions(source / "holdout_site_predictions.parquet", sites)
    if not set(delivery_day(gp.ts)).issubset(opt_days):
        raise ValueError("Group-opt predictions include dates outside source GROUP_OPT split")
    if not set(delivery_day(hp.ts)).issubset(hold_days):
        raise ValueError("Holdout predictions include dates outside source HOLDOUT split")
    if gp.empty or hp.empty:
        raise ValueError("Empty group-opt or holdout predictions")

    generation = pd.read_parquet(root / "train" / "generation.parquet",
                                 columns=["site_id", "ts", "energy"])
    generation["ts"] = to_kst(generation.ts)
    if generation.duplicated(["site_id", "ts"]).any():
        raise ValueError("Duplicate generation labels")
    selected_times = pd.concat([gp.ts, hp.ts], ignore_index=True).unique()
    generation = generation.loc[generation.ts.isin(selected_times)]

    def combine(pred: pd.DataFrame, label: str) -> pd.DataFrame:
        df = pred.merge(sites, on="site_id", how="left", validate="many_to_one")
        df = df.merge(generation, on=["site_id", "ts"], how="left", validate="one_to_one")
        if df.energy.isna().any() or not np.isfinite(df.energy).all():
            raise ValueError(f"{label}: predictions lack actuals, or actuals are non-finite")
        # Base model uses the original NOAA solar calculation, with interval midpoint.
        elev, _, _ = base.solar(pd.DatetimeIndex(df.ts), df.lat.to_numpy(np.float64),
                                df.lon.to_numpy(np.float64))
        df["elev"] = elev
        df["daylight"] = df.elev > 0
        df["actual_norm_error"] = ((df.energy.to_numpy(np.float64)
                                    - df.pred_kwh.to_numpy(np.float64)) /
                                   df.capacity_kw.to_numpy(np.float64))
        if not np.isfinite(df.actual_norm_error).all():
            raise ValueError(f"{label}: nonfinite normalized residual")
        return df

    group = combine(gp, "GROUP_OPT")
    hold = combine(hp, "HOLDOUT")
    del generation, gp, hp
    log(f"Using source={source_run}; fit corrections on {len(group):,} GROUP_OPT site-hours "
        f"({int(group.daylight.sum()):,} daytime), validate on {len(hold):,} HOLDOUT site-hours")
    log("Corrector only fits on GROUP_OPT actuals; HOLDOUT actuals used for scoring only")
    return sites, group, hold, source, split


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Simple 8-feature normalized-error model, only baseline predictions + clock/solar.

    Does NOT use actual generation, any future observed weather, or site ID.
    Group identity is supplied by the separate station/site fitted models.
    """
    q = np.clip(frame.pred_kwh.to_numpy(np.float64) /
                frame.capacity_kw.to_numpy(np.float64), 0., 1.5)
    e = np.maximum(frame.elev.to_numpy(np.float64), 0.) / 90.
    ts = pd.DatetimeIndex(frame.ts)
    h = ts.hour.to_numpy(dtype=np.float64) - 0.5  # right-closed interval midpoint
    doy = ts.dayofyear.to_numpy(dtype=np.float64)
    X = np.column_stack((q, q*q, e, q*e,
                         np.sin(2*np.pi*h/24.), np.cos(2*np.pi*h/24.),
                         np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)))
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite correction features")
    return X


def fit_stats(group: pd.DataFrame, hold: pd.DataFrame):
    """A: per-site normalized residual mean, variance and EB shrinkage.

    Weight = tau^2 / (tau^2 + variance_i / n_i) where tau^2 is the
    estimated between-site variance of mean normalized residuals.
    Lower weight for noisy or poorly sampled site means.
    """
    daytime = group.loc[group.daylight]
    agg = daytime.groupby("site_id", sort=False).actual_norm_error.agg(
        ["mean", "var", "count"])
    if agg.empty:
        raise ValueError("No daytime group-opt records to calibrate")
    agg.columns = ["mean_norm_residual", "var_norm_residual", "n_daytime"]
    agg["var_norm_residual"] = agg.var_norm_residual.fillna(0.).clip(lower=0.)
    noise_var = agg.var_norm_residual / agg.n_daytime
    tau2 = max(float(agg.mean_norm_residual.var(ddof=1)) - float(noise_var.mean()), 0.)
    # Zero tau2 -> shrink all sites to the daytime pooled/global residual mean.
    global_bias = float(daytime.actual_norm_error.mean())
    den = tau2 + noise_var.to_numpy(np.float64)
    shrink = np.divide(tau2, den, out=np.zeros(len(agg), dtype=np.float64), where=den > 0)
    agg["shrink_weight"] = shrink
    agg["final_bias"] = global_bias + shrink * (agg.mean_norm_residual - global_bias)
    agg = agg.reset_index()
    bias = agg.set_index("site_id").final_bias

    def predict(df: pd.DataFrame) -> np.ndarray:
        z = df.site_id.map(bias).fillna(global_bias).to_numpy(np.float64)
        return np.where(df.daylight.to_numpy(), z, 0.)

    metadata = {
        "method": "A: empirical-Bayes normalized residual mean with per-site variance",
        "error_definition": "(actual_kwh - baseline_pred_kwh) / capacity_kw",
        "variance_role": "site-specific variance / daytime sample count controls bias shrinkage",
        "between_site_bias_variance": tau2,
        "global_daytime_bias": global_bias,
        "n_sites_calibrated": len(agg),
    }
    return predict(group), predict(hold), metadata, agg


def fit_group_models(group: pd.DataFrame, hold: pd.DataFrame, method: str):
    """B: separate regularized linear models per WA and WB station; blend 50:50.

    C: separate regularized linear models per site_id, with the same predictors.
    Both fit normalized *out-of-sample* baseline residuals using group_opt only.
    """
    assert method in ("observation_points", "lr_model")
    daytime = group.loc[group.daylight].copy()
    if len(daytime) < MIN_REGRESSION_ROWS:
        raise ValueError("Not enough daytime group-opt rows for regression")
    fallback_bias = float(daytime.actual_norm_error.mean())
    group_keys = ("wa_station", "wb_station") if method == "observation_points" else ("site_id",)
    fit_records: list[dict] = []
    models: dict[str, dict[object, object]] = {}
    for key in group_keys:
        fitted: dict[object, object] = {}
        for value, idx in daytime.groupby(key, sort=False).indices.items():
            chunk = daytime.iloc[idx]
            y = chunk.actual_norm_error.to_numpy(np.float64)
            # Sparse groups use their observed mean as intercept-only correction.
            if len(chunk) < MIN_REGRESSION_ROWS:
                model = float(y.mean())
                fit_type = "group_mean_fallback"
            else:
                model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
                model.fit(design_matrix(chunk), y)
                fit_type = "Ridge"
            fitted[value] = model
            fit_records.append({"group_type": key, "group_id": value,
                                "n_daytime": len(chunk), "model_type": fit_type})
        models[key] = fitted

    def predict(df: pd.DataFrame) -> np.ndarray:
        if not len(df):
            return np.empty(0, dtype=np.float64)
        daytime_mask = df.daylight.to_numpy(dtype=bool)
        all_preds = []
        for key in group_keys:
            residual_prediction = np.zeros(len(df), dtype=np.float64)
            for value, idx in df.groupby(key, sort=False).indices.items():
                idx = np.asarray(idx, dtype=np.int64)
                active = idx[daytime_mask[idx]]
                if not len(active):
                    continue
                model = models[key].get(value, fallback_bias)
                if hasattr(model, "predict"):
                    residual_prediction[active] = model.predict(design_matrix(df.iloc[active]))
                else:
                    residual_prediction[active] = float(model)
            all_preds.append(residual_prediction)
        return np.mean(all_preds, axis=0)

    metadata = {
        "method": "B: station-group Ridge (WA and WB independent, mean ensemble)"
                  if method == "observation_points" else "C: independent site-specific Ridge",
        "error_definition": "(actual_kwh - baseline_pred_kwh) / capacity_kw",
        "regression_features": list(REG_FEATURES),
        "min_rows_per_group_for_ridge": MIN_REGRESSION_ROWS,
        "ridge_alpha": RIDGE_ALPHA,
        "fallback": "group mean for sparse known groups, global daytime bias for unseen groups",
        "n_fitted_models": {key: len(models[key]) for key in group_keys},
    }
    diagnostics = pd.DataFrame(fit_records)
    return predict(group), predict(hold), metadata, diagnostics


def correct_predictions(df: pd.DataFrame, norm_correction: np.ndarray) -> pd.DataFrame:
    if norm_correction.shape != (len(df),):
        raise ValueError("Correction shape mismatch")
    if not np.isfinite(norm_correction).all():
        raise ValueError("Nonfinite model corrections")
    pred = df.pred_kwh.to_numpy(np.float64)
    cap = df.capacity_kw.to_numpy(np.float64)
    daytime = df.daylight.to_numpy(dtype=bool)
    # Baseline: nighttime forced to 0; otherwise nonnegative, no upper cap.
    corrected = np.where(daytime, np.maximum(pred + cap * norm_correction, 0.), 0.)
    result = df[["site_id", "ts"]].copy()
    result["pred_kwh"] = corrected
    if not np.isfinite(result.pred_kwh).all():
        raise ValueError("Nonfinite corrected forecasts")
    return result


def model_metrics(opt, root: Path, labeled: pd.DataFrame, corrected: pd.DataFrame) -> dict:
    df = labeled[["site_id", "ts", "energy"]].merge(corrected, on=["site_id", "ts"],
                                                    how="left", validate="one_to_one")
    if df.pred_kwh.isna().any():
        raise ValueError("Missing corrected predictions for model evaluation")
    error = df.pred_kwh.to_numpy(np.float64) - df.energy.to_numpy(np.float64)
    smp = pd.read_csv(root / "train" / "smp.csv", usecols=["ts", "smp"])
    smp["ts"] = to_kst(smp.ts)
    df = df.merge(smp, on="ts", how="left", validate="many_to_one")
    if df.smp.isna().any():
        raise ValueError("Missing SMP for scored site-hours")
    w = df.smp.to_numpy(np.float64) * np.where(df.ts.dt.hour.between(11,15), 3., 1.)
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Invalid model metric weights")
    return {"MAE_kwh": float(np.abs(error).mean()),
            "RMSE_kwh": float(np.sqrt(np.mean(error*error))),
            "weighted_MAE_kwh": float(np.dot(w, np.abs(error)) / w.sum()),
            "observations": len(df)}


def residual_matrix(df: pd.DataFrame, corrected: pd.DataFrame, site_ids, label: str):
    joined = df[["site_id", "ts", "energy"]].merge(corrected, on=["site_id", "ts"],
                                                    how="left", validate="one_to_one")
    if joined.pred_kwh.isna().any():
        raise ValueError(f"Missing predictions for {label}")
    joined["e"] = joined.pred_kwh - joined.energy
    piv = joined.pivot(index="ts", columns="site_id", values="e")
    piv = piv.reindex(columns=list(site_ids)).sort_index()
    n_missing = int(piv.isna().sum().sum())
    log(f"{label}: {len(piv):,} observed hours, {n_missing:,} missing site-hours "
        "filled with 0 (same as existing 60/20/20 Baseline)")
    return piv.fillna(0.).to_numpy(np.float64), pd.DatetimeIndex(piv.index), n_missing


def group_metrics(opt, root: Path, matrix, times, initial, final) -> dict:
    weights = opt.time_weights(root, times)
    initial_w = float(opt.penalty(matrix, weights, initial))
    final_w = float(opt.penalty(matrix, weights, final))
    lb = float(opt.PENALTY_RATE * np.dot(weights, np.abs(matrix.sum(axis=1))))
    denom = initial_w - lb
    return {"W_initial_krw": initial_w,
            "W_optimized_krw": final_w,
            "W_lower_bound_krw": lb,
            "improvable_cost_krw": final_w - lb,
            "improved_percentage": float(100 * (initial_w - final_w) / denom) if denom > 0 else None,
            "W_reduction_pct": float(100 * (initial_w - final_w) / initial_w) if initial_w > 0 else 0.,
            "hours": len(times)}


def run(method: str = "stats", argv=None):
    if method not in METHODS:
        raise ValueError(f"Unknown posthoc method: {method}")
    parser = argparse.ArgumentParser(description=f"OIBC posthoc experiment: {method}")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-run", default="split602020_baseline",
                        help="Saved 60/20/20 source folder, e.g. split602020_baseline or split602020_l1_loss")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-iterations", type=int, default=0,
                        help="0 means full Greedy until local optimum")
    args = parser.parse_args(argv)
    if (args.source_run in ("", ".", "..") or "/" in args.source_run or
            "\\" in args.source_run):
        parser.error("--source-run must be a single output directory name")
    if args.batch_size < 1 or args.max_iterations < 0:
        parser.error("Invalid Greedy settings")
    start = time.monotonic()
    root = args.data_dir.expanduser().resolve()
    baseline = load_module(root / "baseline.py", "oibc_posthoc_baseline")
    optimizer = load_module(HERE / "baseline_optimizer.py", "oibc_posthoc_optimizer")
    sites, group_df, hold_df, source, split = prepare_data(root, args.source_run, baseline)

    if method == "stats":
        g_corr, h_corr, fit_meta, diagnostic = fit_stats(group_df, hold_df)
    else:
        g_corr, h_corr, fit_meta, diagnostic = fit_group_models(group_df, hold_df, method)
    g_pred = correct_predictions(group_df, g_corr)
    h_pred = correct_predictions(hold_df, h_corr)
    del g_corr, h_corr

    out = root / "output" / "validation_602020" / f"{args.source_run}_posthoc_{method}"
    out.mkdir(parents=True, exist_ok=True)
    g_pred.to_parquet(out / "group_opt_site_predictions.parquet", index=False)
    h_pred.to_parquet(out / "holdout_site_predictions.parquet", index=False)
    diagnostic.to_csv(out / ("site_residual_stats.csv" if method == "stats" else "fitted_group_models.csv"), index=False)

    hold_model = model_metrics(optimizer, root, hold_df, h_pred)
    group_mat, group_times, ng = residual_matrix(group_df, g_pred, sites.site_id, "GROUP_OPT")
    hold_mat, hold_times, nh = residual_matrix(hold_df, h_pred, sites.site_id, "HOLDOUT")
    initial = optimizer.initial_mask(sites)
    optimizer.check_capacity(initial, sites.capacity_kw.to_numpy(np.float64))
    group_weights = optimizer.time_weights(root, group_times)
    optimized, greedy_meta, greedy_trace = optimizer.greedy_best_improvement(
        group_mat, group_weights, sites, initial,
        batch_size=args.batch_size, max_iterations=args.max_iterations)
    train_score = group_metrics(optimizer, root, group_mat, group_times, initial, optimized)
    hold_score = group_metrics(optimizer, root, hold_mat, hold_times, initial, optimized)
    greedy_trace.to_csv(out / "greedy_trace.csv", index=False)
    pd.DataFrame({"site_id": sites.site_id,
                  "group": np.where(optimized, "G1", "G2")}).to_csv(out / "composition.csv", index=False)

    source_summary_file = source / "summary.json"
    source_holdout = None
    if source_summary_file.is_file():
        with source_summary_file.open(encoding="utf-8") as f:
            source_summary = json.load(f)
        source_holdout = source_summary.get("group_evaluation_independent_holdout")
    summary = {
        "method": method,
        "source_run": args.source_run,
        "validation_type": "same_saved_random_delivery_day_60_20_20_split_as_source",
        "model_train_days": int((split.split == "model_train").sum()),
        "group_opt_days": int((split.split == "group_opt").sum()),
        "holdout_days": int((split.split == "holdout").sum()),
        "baseline_models_refit": False,
        "posthoc_fit_data": "GROUP_OPT OOS baseline predictions and actuals, daylight only",
        "holdout_used_to_fit": False,
        "fit": fit_meta,
        "model_metrics_holdout": hold_model,
        "group_optimization_in_sample": train_score,
        "group_evaluation_independent_holdout": hold_score,
        "source_holdout_group_metrics": source_holdout,
        "group_opt_missing_site_hours_filled_zero": ng,
        "holdout_missing_site_hours_filled_zero": nh,
        "greedy_iterations": greedy_meta["iterations"],
        "greedy_local_minimum": bool(greedy_meta["local_minimum_reached"]),
        "elapsed_seconds": float(time.monotonic() - start),
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False, default=str)

    log("=" * 72)
    log(f"EXPERIMENT = posthoc_{method} | SOURCE={args.source_run}")
    log(f"HOLDOUT MODEL weighted MAE     = {hold_model['weighted_MAE_kwh']:.5f} kWh")
    log(f"HOLDOUT GROUP initial W       = {hold_score['W_initial_krw']:,.2f} KRW")
    log(f"HOLDOUT GROUP optimized W     = {hold_score['W_optimized_krw']:,.2f} KRW")
    log(f"HOLDOUT GROUP lower bound     = {hold_score['W_lower_bound_krw']:,.2f} KRW")
    log(f"HOLDOUT GROUP improvable cost = {hold_score['improvable_cost_krw']:,.2f} KRW")
    log(f"HOLDOUT GROUP improved percent= {hold_score['improved_percentage']:.6f}%")
    if source_holdout:
        log(f"UNCHANGED SOURCE optimized W  = {source_holdout['W_optimized_krw']:,.2f} KRW")
        log(f"POSTHOC DELTA vs SOURCE       = "
            f"{hold_score['W_optimized_krw']-source_holdout['W_optimized_krw']:+,.2f} KRW")
    log(f"Output: {out}")
    return summary


if __name__ == "__main__":
    run("stats")
