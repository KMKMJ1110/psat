#!/usr/bin/env python3
"""OIBC: single-fit, randomized delivery-DAY 60/20/20 experiment.

Place this file and the three small variant launchers in data/codes/.
Keep original data/baseline.py and data/codes/baseline_optimizer.py unchanged.
Run: python3 codes/validation_602020_baseline.py

Same seed -> same dates for every variant. Splits are disjoint:
  60% of baseline-period days -> fit LightGBM
  20% of baseline-period days -> predict OOS; fit greedy GROUP assignment
  20% of baseline-period days -> independent prediction + GROUP evaluation

Full-history variant adds earlier (< baseline.TRAIN_FROM) generation to FIT ONLY.
Only preprocessed WA/WB forecasts are read. This is local validation;
no contest data are uploaded. No hidden-test forecasts/submission are generated.

IMPORTANT: To match the user's existing optimizer runs, missing site-hour
residuals are filled with 0 for GROUP scoring. This approximates missing
residuals; it does NOT imply that the true prediction error was zero.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "codes" else SCRIPT_DIR
SEED = 42
VARIANTS = ("baseline", "full_history", "round17_only", "l1_loss")
RUN_NAMES = {
    "baseline": "split602020_baseline",
    "full_history": "split602020_full_history",
    "round17_only": "split602020_round17_only",
    "l1_loss": "split602020_l1_loss",
}
BASELINE_PARAMS = dict(
    n_estimators=600, learning_rate=0.06, num_leaves=127,
    min_child_samples=200, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, verbose=-1, n_jobs=-1,
)


def log(message: str) -> None:
    print(f"[60/20/20] {message}", flush=True)


def to_kst(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series)
    return s.dt.tz_localize("Asia/Seoul") if s.dt.tz is None else s.dt.tz_convert("Asia/Seoul")


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Required Python file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def days_of(series: pd.Series) -> pd.Series:
    # Right-closed timestamps: 01:00 denotes 00:00-01:00.
    return (series - pd.Timedelta(hours=1)).dt.normalize()


def split_602020(base_gen: pd.DataFrame, seed: int):
    days = pd.DatetimeIndex(days_of(base_gen.ts).drop_duplicates().sort_values())
    if len(days) < 5:
        raise ValueError("Need at least five delivery days")
   
    # [수정] 월 단위 징검다리 분할 (Train: 홀수달 / Opt: 2,6,10월 / Holdout: 4,8,12월)
    months = days.month
    fit_days = days[months.isin([1, 3, 5, 7, 9, 11])]
    opt_days = days[months.isin([2, 6, 10])]
    holdout_days = days[months.isin([4, 8, 12])]
    n_fit, n_opt = len(fit_days), len(opt_days)

    if min(len(fit_days), len(opt_days), len(holdout_days)) < 1:
        raise ValueError("Empty data split")
    names = np.select(
        [days.isin(fit_days), days.isin(opt_days)],
        ["model_train", "group_opt"], default="holdout",
    )
    split = pd.DataFrame({"day_kst": days.strftime("%Y-%m-%d"), "split": names})
    row_days = days_of(base_gen.ts)
    fit = base_gen.loc[row_days.isin(fit_days)].copy()
    opt = base_gen.loc[row_days.isin(opt_days)].copy()
    hold = base_gen.loc[row_days.isin(holdout_days)].copy()
    assert (len(fit) + len(opt) + len(hold)) == len(base_gen)
    assert not (set(fit_days) & set(opt_days) or set(fit_days) & set(holdout_days) or set(opt_days) & set(holdout_days))
    log(f"Random delivery-day split seed={seed}: model_train={n_fit}, "
        f"group_opt={n_opt}, independent_holdout={len(holdout_days)} days")
    log(f"Generation rows: model_train={len(fit):,}, group_opt={len(opt):,}, "
        f"holdout={len(hold):,}")
    return fit, opt, hold, split


def load_weather(root: Path, base, minimum: pd.Timestamp):
    weather_root = root / "output" / "preprocessed" / "train"
    if not weather_root.is_dir():
        raise FileNotFoundError(f"Preprocessed weather directory missing: {weather_root}")
    a = pd.read_parquet(weather_root / "forecast_wa.parquet",
                        columns=["wa_station", "round", "ts"] + list(base.WA_COLS))
    b = pd.read_parquet(weather_root / "forecast_wb.parquet",
                        columns=["wb_station", "round", "ts"] + list(base.WB_COLS))
    for d in (a, b):
        d["ts"] = to_kst(d.ts)
    a = a.loc[a.ts >= minimum].copy()
    b = b.loc[b.ts >= minimum].copy()
    log(f"Using PREPROCESSED forecasts from {weather_root} (WA={len(a):,}, WB={len(b):,})")
    return a, b

# [추가] 기상청 A(wa)와 B(wb)의 예보 불일치(Disagreement) 피처 생성 함수 운량이랑 습도 두 개만!!!!!!!!!!
DISAGREE_COLS = [
    "abs_diff_cloud", "diff_cloud",
    "abs_diff_hum", "diff_hum",

]

def add_ab_disagreement(df: pd.DataFrame) -> pd.DataFrame:
    df["diff_cloud"] = (df["wa_cloud"] - df["wb_cloud"]).astype("float32")
    df["abs_diff_cloud"] = df["diff_cloud"].abs()
    df["diff_hum"] = (df["wa_humidity"] - df["wb_rel_hum"]).astype("float32")
    df["abs_diff_hum"] = df["diff_hum"].abs()
    return df

def fit_models(base, sites: pd.DataFrame, fit: pd.DataFrame,
               fa: pd.DataFrame, fb: pd.DataFrame, variant: str):
    rounds = (17,) if variant == "round17_only" else (10, 17)
    params = dict(BASELINE_PARAMS)
    if variant == "l1_loss":
        params["objective"] = "regression_l1"
    models = {}
    for r in rounds:
        df = add_ab_disagreement(base.features(fit, sites, fa, fb, None, r))  # <-- 수정 1: 함수 감싸기
        if len(df) != len(fit):
            raise ValueError(f"round {r}: feature join changed row count")
        df = df.loc[df.elev > 0]
        X = df[base.FEATS + DISAGREE_COLS].astype("float32")                  # <-- 수정 2: 피처 리스트 합치기

        y = (df.energy / df.cap).clip(0, 1.2).astype("float32")
        log(f"Fitting round={r}, X={X.shape}, objective={params.get('objective', 'regression (L2)')}")
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
        log(f"round={r} train normalized MAE={np.abs(model.predict(X)-y).mean():.6f}")
        models[r] = model
        del df, X, y
    return models, rounds, params


def predict_eval(base, models: dict, rounds: tuple, sites: pd.DataFrame,
                 actuals: pd.DataFrame, fa: pd.DataFrame, fb: pd.DataFrame):
    """Generate forecasts only for unseen group-opt + holdout records."""
    combined = None
    for r in rounds:
        df = add_ab_disagreement(base.features(actuals, sites, fa, fb, None, r))  # <-- 수정 1: 함수 감싸기
        if len(df) != len(actuals):
            raise ValueError(f"round {r}: feature join changed evaluation row count")
        # Same baseline behavior: night=0, clip negative capacity-factor to zero.
        pred = models[r].predict(df[base.FEATS + DISAGREE_COLS].astype("float32")) # <-- 수정 2: 피처 리스트 합치기
        
        kwh = np.maximum(np.where(df.elev.to_numpy() > 0, pred, 0.0), 0.0) * df.cap.to_numpy()
        one = df[["site_id", "ts"]].copy()
        one[f"round_{r}"] = kwh
        if one.duplicated(["site_id", "ts"]).any():
            raise ValueError(f"Duplicate site-hour predictions for round {r}")
        combined = one if combined is None else combined.merge(
            one, on=["site_id", "ts"], how="inner", validate="one_to_one")
        log(f"round={r} predicted {len(one):,} unseen site-hours")
        del df, pred, one
    if combined is None or len(combined) != len(actuals):
        raise ValueError("Incomplete evaluation predictions")
    cols = [f"round_{r}" for r in rounds]
    if combined[cols].isna().any().any():
        raise ValueError("Nonfinite round predictions")
    combined["pred_kwh"] = combined[cols].mean(axis=1)  # 50:50 or round17-only
    return combined[["site_id", "ts", "pred_kwh"]]


def model_metrics(root: Path, actuals: pd.DataFrame, predictions: pd.DataFrame):
    scored = actuals.merge(predictions, on=["site_id", "ts"], how="left", validate="one_to_one")
    if len(scored) != len(actuals) or scored.pred_kwh.isna().any():
        raise ValueError("Missing model predictions for observed labels")
    error = scored.pred_kwh.to_numpy(np.float64) - scored.energy.to_numpy(np.float64)
    smp = pd.read_csv(root / "train" / "smp.csv", usecols=["ts", "smp"])
    smp["ts"] = to_kst(smp.ts)
    scored = scored.merge(smp, on="ts", how="left", validate="many_to_one")
    if scored.smp.isna().any() or (scored.smp < 0).any():
        raise ValueError("Missing or negative SMP")
    weight = scored.smp.to_numpy(np.float64) * np.where(
        scored.ts.dt.hour.between(11, 15).to_numpy(), 3., 1.)
    return {
        "n_observed_site_hours": len(scored),
        "MAE_kwh": float(np.mean(np.abs(error))),
        "RMSE_kwh": float(np.sqrt(np.mean(error ** 2))),
        "weighted_MAE_kwh": float(np.dot(weight, np.abs(error)) / weight.sum()),
    }


def residual_matrix(actuals: pd.DataFrame, predictions: pd.DataFrame, site_ids,
                    label: str):
    df = actuals[["site_id", "ts", "energy"]].merge(
        predictions, on=["site_id", "ts"], how="left", validate="one_to_one")
    if len(df) != len(actuals) or df.pred_kwh.isna().any():
        raise ValueError(f"{label}: missing predictions for observed generation")
    df["error"] = df.pred_kwh.to_numpy() - df.energy.to_numpy()
    grid = df.pivot(index="ts", columns="site_id", values="error")
    grid = grid.reindex(columns=list(site_ids)).sort_index()
    missing = int(grid.isna().sum().sum())
    log(f"{label}: {len(grid):,} observed hours; {missing:,} missing site-hour "
        "residuals filled with 0 (matches existing local Greedy scoring)")
    matrix = grid.fillna(0.0).to_numpy(np.float64)
    times = pd.DatetimeIndex(grid.index)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label}: nonfinite residuals")
    return matrix, times, missing


def score_group(opt, root: Path, matrix: np.ndarray,
                times: pd.DatetimeIndex, initial_mask: np.ndarray, optimized_mask: np.ndarray):
    weights = opt.time_weights(root, times)
    initial_w = float(opt.penalty(matrix, weights, initial_mask))
    optimized_w = float(opt.penalty(matrix, weights, optimized_mask))
    lower_bound = float(opt.PENALTY_RATE * np.dot(weights, np.abs(matrix.sum(axis=1))))
    return {
        "W_initial_krw": initial_w,
        "W_optimized_krw": optimized_w,
        "W_lower_bound_krw": lower_bound,
        "W_reduction_krw": initial_w - optimized_w,
        "W_reduction_pct": 100 * (initial_w - optimized_w) / initial_w if initial_w else 0.0,
        "hours": len(times),
    }


def run(variant: str, argv=None):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant}")
    ap = argparse.ArgumentParser(description=f"OIBC 60/20/20: {variant}")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-iterations", type=int, default=0,
                    help="0 = Greedy to local optimum")
    args = ap.parse_args(argv)
    if args.batch_size < 1 or args.max_iterations < 0:
        ap.error("Invalid Greedy settings")
    started = time.monotonic()
    root = args.data_dir.expanduser().resolve()
    base = load_module(root / "baseline.py", "oibc_baseline_602020")
    opt = load_module(SCRIPT_DIR / "baseline_optimizer.py", "oibc_optimizer_602020")
    if tuple(base.ROUNDS) != (10, 17):
        raise ValueError("Unexpected baseline forecast rounds; expected 10 and 17")

    run_name = RUN_NAMES[variant]
    out = root / "output" / "validation_602020" / run_name
    out.mkdir(parents=True, exist_ok=True)
    log(f"EXPERIMENT = {variant}; output = {out}")
    log(f"Baseline period starts at {base.TRAIN_FROM}; Holdout independent of model AND grouping")

    sites = pd.read_csv(root / "sites.csv")[[
        "site_id", "current_group", "capacity_kw", "lat", "lon", "wa_station", "wb_station"]]
    if len(sites) != 253 or sites.site_id.duplicated().any():
        raise ValueError("Invalid sites.csv: expected 253 unique sites")

    gen = pd.read_parquet(root / "train" / "generation.parquet", columns=["site_id", "ts", "energy"])
    gen["ts"] = to_kst(gen.ts)
    if gen.duplicated(["site_id", "ts"]).any():
        raise ValueError("Duplicate generation labels")
    if not np.isfinite(gen.energy).all():
        raise ValueError("Nonfinite generation labels")
    baseline_gen = gen.loc[gen.ts >= base.TRAIN_FROM].copy()
    model_fit, group_actual, holdout_actual, day_split = split_602020(baseline_gen, args.seed)
    day_split.to_csv(out / "split_days.csv", index=False)

    if variant == "full_history":
        historical = gen.loc[gen.ts < base.TRAIN_FROM].copy()
        model_fit = pd.concat([historical, model_fit], ignore_index=True)
        log(f"Full history: added {len(historical):,} earlier site-hours to model FIT ONLY")
    del gen, baseline_gen
    if len(model_fit) == 0 or len(group_actual) == 0 or len(holdout_actual) == 0:
        raise ValueError("Empty split")

    weather_start = model_fit.ts.min() if variant == "full_history" else base.TRAIN_FROM
    fa, fb = load_weather(root, base, weather_start)
    models, rounds, params = fit_models(base, sites, model_fit, fa, fb, variant)
    del model_fit

    eval_actual = pd.concat([group_actual, holdout_actual], ignore_index=True)
    eval_pred = predict_eval(base, models, rounds, sites, eval_actual, fa, fb)
    del models, fa, fb, eval_actual
    # Match OOS predictions to the FIXED day_split, never use 20% holdout to fit groups.
    pred_day = days_of(eval_pred.ts)
    opt_days = pd.to_datetime(day_split.loc[day_split.split == "group_opt", "day_kst"]).dt.tz_localize("Asia/Seoul")
    holdout_days = pd.to_datetime(day_split.loc[day_split.split == "holdout", "day_kst"]).dt.tz_localize("Asia/Seoul")
    group_pred = eval_pred.loc[pred_day.isin(opt_days)].copy()
    holdout_pred = eval_pred.loc[pred_day.isin(holdout_days)].copy()
    assert len(group_pred) == len(group_actual) and len(holdout_pred) == len(holdout_actual)
    del eval_pred
    group_pred.to_parquet(out / "group_opt_site_predictions.parquet", index=False)
    holdout_pred.to_parquet(out / "holdout_site_predictions.parquet", index=False)

    group_model = model_metrics(root, group_actual, group_pred)
    holdout_model = model_metrics(root, holdout_actual, holdout_pred)
    log(f"HOLDOUT MODEL MAE={holdout_model['MAE_kwh']:.5f}, "
        f"RMSE={holdout_model['RMSE_kwh']:.5f}, "
        f"Weighted MAE={holdout_model['weighted_MAE_kwh']:.5f}")

    group_matrix, group_times, missing_group = residual_matrix(
        group_actual, group_pred, sites.site_id, "GROUP_OPT")
    hold_matrix, hold_times, missing_hold = residual_matrix(
        holdout_actual, holdout_pred, sites.site_id, "HOLDOUT")
    del group_actual, holdout_actual, group_pred, holdout_pred

    initial = opt.initial_mask(sites)
    opt.check_capacity(initial, sites.capacity_kw.to_numpy(np.float64))
    group_weights = opt.time_weights(root, group_times)
    log(f"GROUP_OPT initial W={opt.penalty(group_matrix, group_weights, initial):,.2f} KRW")
    optimized, greedy_meta, trace = opt.greedy_best_improvement(
        group_matrix, group_weights, sites, initial,
        batch_size=args.batch_size, max_iterations=args.max_iterations,
    )
    group_score = score_group(opt, root, group_matrix, group_times, initial, optimized)
    hold_score = score_group(opt, root, hold_matrix, hold_times, initial, optimized)

    assignment = pd.DataFrame({"site_id": sites.site_id,
                               "group": np.where(optimized, "G1", "G2")})
    assignment.to_csv(out / "composition.csv", index=False)
    trace.to_csv(out / "greedy_trace.csv", index=False)
    summary = {
        "experiment": variant,
        "validation_type": "random_delivery_day_single_split_60_20_20",
        "seed": args.seed,
        "model_train_days": int((day_split.split == "model_train").sum()),
        "group_optimization_days": int((day_split.split == "group_opt").sum()),
        "independent_holdout_days": int((day_split.split == "holdout").sum()),
        "baseline_train_from": str(base.TRAIN_FROM),
        "historical_training_added": variant == "full_history",
        "model_params": params,
        "rounds": list(rounds),
        "round_aggregation": "100% round17" if variant == "round17_only" else "50:50 round10/17",
        "weather_source": "output/preprocessed/train/{forecast_wa,forecast_wb}.parquet",
        "missing_residual_policy": "missing site-hour residuals filled as 0; not observed zero errors",
        "group_opt_missing_site_hours": missing_group,
        "holdout_missing_site_hours": missing_hold,
        "model_metrics_group_opt": group_model,
        "model_metrics_holdout": holdout_model,
        "group_optimization_in_sample": group_score,
        "group_evaluation_independent_holdout": hold_score,
        "greedy_iterations": greedy_meta["iterations"],
        "greedy_local_minimum": bool(greedy_meta["local_minimum_reached"]),
        "elapsed_seconds": time.monotonic() - started,
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)

    log("=" * 72)
    log(f"EXPERIMENT {variant} | RANDOM DAY 60:20:20 (seed={args.seed})")
    log(f"GROUP OPT     initial W       = {group_score['W_initial_krw']:,.2f} KRW")
    log(f"GROUP OPT     optimized W     = {group_score['W_optimized_krw']:,.2f} KRW")
    log(f"GROUP OPT     lower bound     = {group_score['W_lower_bound_krw']:,.2f} KRW")
    log(f"HOLDOUT MODEL weighted MAE     = {holdout_model['weighted_MAE_kwh']:.5f} kWh")
    log(f"HOLDOUT GROUP initial W       = {hold_score['W_initial_krw']:,.2f} KRW")
    log(f"HOLDOUT GROUP optimized W     = {hold_score['W_optimized_krw']:,.2f} KRW")
    log(f"HOLDOUT GROUP lower bound     = {hold_score['W_lower_bound_krw']:,.2f} KRW")
    log(f"HOLDOUT GROUP W reduction     = {hold_score['W_reduction_pct']:.4f}%")
    log(f"Results saved to {out}")
    return summary


if __name__ == "__main__":
    run("baseline")
