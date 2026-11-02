#!/usr/bin/env python3
"""OIBC 60/20/20 experiment: Rotation Spatial Features + SMP/Time/Capacity-Weighted L1.

Place this file in data/codes/ alongside baseline_optimizer.py.
Keep data/baseline.py and data/codes/baseline_optimizer.py unchanged.
Run: python3 codes/baseline_validation_rotation_weighted_l1.py
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

VARIANTS = (
    "baseline",
    "full_history",
    "round17_only",
    "l1_loss",
    "rotation_features",
    "rotation_weighted_l1",
)
RUN_NAMES = {
    "baseline": "split602020_baseline",
    "full_history": "split602020_full_history",
    "round17_only": "split602020_round17_only",
    "l1_loss": "split602020_l1_loss",
    "rotation_features": "split602020_rotation_features",
    "rotation_weighted_l1": "split602020_rotation_weighted_l1",
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
    return (series - pd.Timedelta(hours=1)).dt.normalize()


def add_rotation_features(df: pd.DataFrame, base_feats: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """발전소 위도(lat)와 경도(lon) 좌표 회전 기반 대각선 축 피처 생성"""
    df = df.copy()
    df["lat_lon_sum"] = (df["lat"] + df["lon"]).astype("float32")
    df["lat_lon_diff"] = (df["lat"] - df["lon"]).astype("float32")
    
    new_feats = list(base_feats) + ["lat_lon_sum", "lat_lon_diff"]
    return df, new_feats



def load_smp_table(root: Path) -> pd.DataFrame:
    """학습 가중치 계산용 SMP를 로드하고 시간 키의 유일성을 확인한다."""
    path = root / "train" / "smp.csv"
    if not path.is_file():
        raise FileNotFoundError(f"SMP file not found: {path}")

    smp = pd.read_csv(path, usecols=["ts", "smp"])
    smp["ts"] = to_kst(smp["ts"])
    smp["smp"] = pd.to_numeric(smp["smp"], errors="coerce")

    if smp["ts"].isna().any() or smp["ts"].duplicated().any():
        raise ValueError("SMP table has missing or duplicate timestamps")
    smp_values = smp["smp"].to_numpy(dtype=np.float64)
    if not np.isfinite(smp_values).all() or (smp_values < 0).any():
        raise ValueError("SMP table contains nonfinite or negative values")
    return smp


def make_weighted_l1_sample_weight(df: pd.DataFrame, smp: pd.DataFrame) -> np.ndarray:
    """weighted_MAE_kwh에 대응하는 L1 학습 sample_weight를 계산한다.

    학습 타깃은 y = energy / cap 이므로,
    kWh 절대오차 = cap * |predicted_normalized - y| 이다.
    따라서 sample_weight = cap * SMP * 시간대 가중치로 설정한다.
    전체 가중치의 평균을 1로 정규화해 가중치의 상대비율은 유지하고
    불필요하게 큰 절대 스케일이 학습 정규화에 미치는 영향을 줄인다.
    """
    required = {"ts", "cap"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns for weighted L1: {sorted(missing)}")

    smp_by_ts = smp.set_index("ts")["smp"]
    smp_values = df["ts"].map(smp_by_ts)
    if smp_values.isna().any():
        missing_count = int(smp_values.isna().sum())
        raise ValueError(f"SMP is missing for {missing_count:,} training rows")

    cap = pd.to_numeric(df["cap"], errors="coerce").to_numpy(dtype=np.float64)
    price = smp_values.to_numpy(dtype=np.float64)
    hour_multiplier = np.where(df["ts"].dt.hour.between(11, 15).to_numpy(), 3.0, 1.0)

    if not np.isfinite(cap).all() or (cap <= 0).any():
        raise ValueError("Training capacity contains nonfinite or nonpositive values")

    raw_weight = cap * price * hour_multiplier
    if not np.isfinite(raw_weight).all() or (raw_weight < 0).any():
        raise ValueError("Computed sample weights contain nonfinite or negative values")

    mean_weight = float(raw_weight.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("Computed sample weights have a nonpositive mean")

    normalized_weight = raw_weight / mean_weight
    log(
        "Weighted L1 sample_weight = cap × SMP × hour_multiplier; "
        f"raw_mean={mean_weight:.6g}, normalized_range="
        f"[{normalized_weight.min():.4f}, {normalized_weight.max():.4f}]"
    )
    return normalized_weight.astype(np.float32)


def split_602020(base_gen: pd.DataFrame, seed: int):
    days = pd.DatetimeIndex(days_of(base_gen.ts).drop_duplicates().sort_values())
    if len(days) < 5:
        raise ValueError("Need at least five delivery days")
    perm = np.random.default_rng(seed).permutation(len(days))
    n_fit = int(np.floor(0.60 * len(days)))
    n_opt = int(np.floor(0.20 * len(days)))
    
    fit_days = days[perm[:n_fit]]
    opt_days = days[perm[n_fit:n_fit + n_opt]]
    holdout_days = days[perm[n_fit + n_opt:]]
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


def fit_models(base, sites: pd.DataFrame, fit: pd.DataFrame,
               fa: pd.DataFrame, fb: pd.DataFrame, variant: str,
               smp: pd.DataFrame | None = None):
    rounds = (17,) if variant == "round17_only" else (10, 17)
    params = dict(BASELINE_PARAMS)

    weighted_l1 = variant == "rotation_weighted_l1"
    if variant == "l1_loss" or weighted_l1:
        params["objective"] = "regression_l1"
    if weighted_l1 and smp is None:
        raise ValueError("SMP table is required for rotation_weighted_l1")

    models = {}
    for r in rounds:
        df = base.features(fit, sites, fa, fb, None, r)
        if len(df) != len(fit):
            raise ValueError(f"round {r}: feature join changed row count")
        df = df.loc[df.elev > 0].copy()

        if variant in ("rotation_features", "rotation_weighted_l1"):
            df, feats_used = add_rotation_features(df, base.FEATS)
        else:
            feats_used = base.FEATS

        X = df[feats_used].astype("float32")
        y = (df.energy / df.cap).clip(0, 1.2).astype("float32")

        fit_kwargs = {}
        if weighted_l1:
            sample_weight = make_weighted_l1_sample_weight(df, smp)
            fit_kwargs["sample_weight"] = sample_weight
        else:
            sample_weight = None

        objective = params.get("objective", "regression (L2)")
        weight_desc = "; weighted sample_weight enabled" if weighted_l1 else ""
        log(f"Fitting round={r}, X={X.shape}, objective={objective}{weight_desc}")
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y, **fit_kwargs)
        log(f"round={r} train normalized MAE={np.abs(model.predict(X)-y).mean():.6f}")
        models[r] = (model, feats_used)
        del df, X, y, sample_weight
    return models, rounds, params


def predict_eval(base, models: dict, rounds: tuple, sites: pd.DataFrame,
                 actuals: pd.DataFrame, fa: pd.DataFrame, fb: pd.DataFrame, variant: str):
    """Generate forecasts only for unseen group-opt + holdout records."""
    combined = None
    for r in rounds:
        model, feats_used = models[r]
        df = base.features(actuals, sites, fa, fb, None, r)
        if len(df) != len(actuals):
            raise ValueError(f"round {r}: feature join changed evaluation row count")

        if variant in ("rotation_features", "rotation_weighted_l1"):
            df, _ = add_rotation_features(df, base.FEATS)

        pred = model.predict(df[feats_used].astype("float32"))
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
    combined["pred_kwh"] = combined[cols].mean(axis=1)
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


def residual_matrix(actuals: pd.DataFrame, predictions: pd.DataFrame, site_ids, label: str):
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


def print_formatted_table_row(variant: str, holdout_model: dict, hold_score: dict) -> None:
    """18개 표준 표 컬럼 출력 함수"""
    w_mae = holdout_model['weighted_MAE_kwh']
    w_init = round(hold_score['W_initial_krw'])
    w_opt = round(hold_score['W_optimized_krw'])
    w_lb = round(hold_score['W_lower_bound_krw'])
    
    improvable_cost = w_opt - w_lb
    potential_total = w_init - w_lb
    improved_pct = ((w_init - w_opt) / potential_total * 100) if potential_total > 0 else 0.0

    val_desc = "학습 기간 전체" if variant == "full_history" else "Baseline"
    has_rotation = variant in ("rotation_features", "rotation_weighted_l1")
    feat_desc = "Feature : Rotation (lat±lon)" if has_rotation else "Baseline"
    if variant == "rotation_weighted_l1":
        learn_desc = "Loss : Weighted L1 (SMP×hour×cap)"
        name_display = "Rotation + Weighted L1"
        notes = "베이스라인 조건 동일 + 회전 피처 2종 + cap×SMP×시간대 가중 L1"
    elif variant == "l1_loss":
        learn_desc = "Loss : MAE"
        name_display = "L1 Loss"
        notes = "베이스라인 조건 동일 + L1 손실함수 적용"
    else:
        learn_desc = "Baseline"
        name_display = "Rotation Features" if variant == "rotation_features" else variant
        notes = "베이스라인 조건 동일 + 위도/경도 대각선 회전 축 피처 2종 추가" if variant == "rotation_features" else variant
    post_desc = "Round 17 Only" if variant == "round17_only" else "None"

    cols = [
        "5",
        name_display,
        val_desc,
        "Baseline",
        feat_desc,
        learn_desc,
        "Baseline",
        post_desc,
        "None",
        "Greedy",
        f"{w_mae:.5f}",
        f"{w_init:,}",
        f"{w_opt:,}",
        f"{w_lb:,}",
        f"{improvable_cost:,}",
        f"{improved_pct:.6f}",
        name_display,
        notes
    ]

    print("\n" + "="*80, flush=True)
    print("📊 [실험 결과 표 규격 출력 - 엑셀/마크다운 복사용]", flush=True)
    print("="*80, flush=True)
    
    # 1. 마크다운 행
    md_row = "| " + " | ".join(cols) + " |"
    print("\n[ 마크다운(Markdown) 표 행 ]:", flush=True)
    print(md_row, flush=True)
    
    # 2. 엑셀 전용 탭구분 행
    tsv_row = "\t".join(cols)
    print("\n[ 엑셀(Excel) 복사 전용 탭구분(TSV) 행 ]:", flush=True)
    print(tsv_row, flush=True)
    print("="*80 + "\n", flush=True)


def run(variant: str, argv=None):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant}")
    ap = argparse.ArgumentParser(description=f"OIBC 60/20/20: {variant}")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-iterations", type=int, default=0, help="0 = Greedy to local optimum")
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

    weather_start = model_fit.ts.min() if variant == "full_history" else base.TRAIN_FROM
    fa, fb = load_weather(root, base, weather_start)
    smp_for_fit = load_smp_table(root) if variant == "rotation_weighted_l1" else None
    models, rounds, params = fit_models(
        base, sites, model_fit, fa, fb, variant, smp=smp_for_fit
    )
    del smp_for_fit
    del model_fit

    eval_actual = pd.concat([group_actual, holdout_actual], ignore_index=True)
    eval_pred = predict_eval(base, models, rounds, sites, eval_actual, fa, fb, variant)
    del models, fa, fb, eval_actual
    
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

    group_matrix, group_times, missing_group = residual_matrix(group_actual, group_pred, sites.site_id, "GROUP_OPT")
    hold_matrix, hold_times, missing_hold = residual_matrix(holdout_actual, holdout_pred, sites.site_id, "HOLDOUT")
    del group_actual, holdout_actual, group_pred, holdout_pred

    initial = opt.initial_mask(sites)
    opt.check_capacity(initial, sites.capacity_kw.to_numpy(np.float64))
    group_weights = opt.time_weights(root, group_times)
    optimized, greedy_meta, trace = opt.greedy_best_improvement(
        group_matrix, group_weights, sites, initial,
        batch_size=args.batch_size, max_iterations=args.max_iterations,
    )
    group_score = score_group(opt, root, group_matrix, group_times, initial, optimized)
    hold_score = score_group(opt, root, hold_matrix, hold_times, initial, optimized)

    assignment = pd.DataFrame({"site_id": sites.site_id, "group": np.where(optimized, "G1", "G2")})
    assignment.to_csv(out / "composition.csv", index=False)
    trace.to_csv(out / "greedy_trace.csv", index=False)
    
    summary = {
        "experiment": variant,
        "training_objective": params.get("objective", "regression (L2)"),
        "training_sample_weight": (
            "capacity_kw × SMP × 3 for hours 11-15 (otherwise ×1); mean-normalized"
            if variant == "rotation_weighted_l1" else None
        ),
        "features": (
            "baseline + lat_lon_sum + lat_lon_diff"
            if variant in ("rotation_features", "rotation_weighted_l1") else "baseline"
        ),
        "model_metrics_holdout": holdout_model,
        "group_evaluation_independent_holdout": hold_score,
        "elapsed_seconds": time.monotonic() - started,
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)

    # 18개 컬럼 표 규격 콘솔 출력
    print_formatted_table_row(variant, holdout_model, hold_score)
    return summary


if __name__ == "__main__":
    run("rotation_weighted_l1")
