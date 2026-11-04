#!/usr/bin/env python3
"""OIBC: Penalty Minimization Baseline (Clean Rotation Feats + Single LGBM + Spatial Hedging)
Working Directory: /Users/ichang-yun/Library/Mobile Documents/com~apple~CloudDocs/대학/P-sat/주제분석/아카이브/
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DEFAULT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "codes" else SCRIPT_DIR
SEED = 42

PARAMS_LGBM = dict(
    n_estimators=600,
    learning_rate=0.06,
    num_leaves=31,
    min_child_samples=300,
    reg_alpha=0.5,
    reg_lambda=1.0,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    objective="regression_l1",
    verbose=-1,
    n_jobs=-1,
)

def log(msg: str) -> None:
    print(f"[PENALTY-OPT] {msg}", flush=True)

def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def to_kst(s):
    s = pd.to_datetime(s)
    return s.dt.tz_localize("Asia/Seoul") if s.dt.tz is None else s.dt.tz_convert("Asia/Seoul")

# =====================================================================
# [1] 경량화 피처 엔지니어링 (군더더기 제거, 회전 피처 중심)
# =====================================================================
def engineer_clean_features(df: pd.DataFrame, base):
    out = df.copy()
    
    # 회전 피처 (Rotation)
    out["lat_lon_sum"] = (out["lat"] + out["lon"]).astype("float32")
    out["lat_lon_diff"] = (out["lat"] - out["lon"]).astype("float32")
    
    # 기상 불일치 (Cloud Disagreement)
    out["cloud_diff"] = out["wa_cloud"] - out["wb_cloud"]
    out["cloud_disagreement"] = out["cloud_diff"].abs().astype("float32")
    
    clean_user_feats = ["lat_lon_sum", "lat_lon_diff", "cloud_diff", "cloud_disagreement"]
    final_feats = list(base.FEATS) + clean_user_feats
    
    return out, final_feats


def residual_matrix(actuals, predictions, site_ids, label):
    q = actuals[["site_id", "ts", "energy"]].merge(
        predictions, on=["site_id", "ts"], how="left", validate="one_to_one"
    )
    if len(q) != len(actuals) or q.pred_kwh.isna().any():
        raise ValueError(f"{label}: missing predictions")
    q["error"] = q.pred_kwh.to_numpy(float) - q.energy.to_numpy(float)
    grid = q.pivot(index="ts", columns="site_id", values="error").reindex(columns=list(site_ids)).sort_index()
    missing = int(grid.isna().sum().sum())
    log(f"{label}: {len(grid):,} timestamps, {missing:,} missing site-hours filled with zero for scoring")
    return grid.fillna(0.0).to_numpy(float), pd.DatetimeIndex(grid.index), missing


# =====================================================================
# [2] 공간 교대 배정 (Spatial Hedging) 초기화 함수
# =====================================================================
def create_spatial_hedging_mask(sites: pd.DataFrame) -> np.ndarray:
    coords = sites[["lat", "lon"]].to_numpy(dtype=float)
    capacities = sites["capacity_kw"].to_numpy(dtype=float)
    coords_centered = coords - coords.mean(axis=0)
    cov = np.cov(coords_centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    main_axis = evecs[:, np.argmax(evals)] 
    spatial_scores = coords_centered @ main_axis
    sorted_indices = np.argsort(spatial_scores)
    
    mask = np.zeros(len(sites), dtype=bool)
    g1_cap = 0.0
    g2_cap = 0.0
    for idx in sorted_indices:
        cap = capacities[idx]
        if g1_cap <= g2_cap:
            mask[idx] = True  
            g1_cap += cap
        else:
            mask[idx] = False 
            g2_cap += cap
            
    log(f"Spatial Hedging Init => G1: {g1_cap:,.1f} kW / G2: {g2_cap:,.1f} kW")
    return mask


def print_summary(metrics: dict, hold_days: int, variant_name: str = "Clean Rotation + Single LGBM + Spatial Hedging") -> None:
    print("\n" + "=" * 80, flush=True)
    print(f"📊 [실험 결과 요약 - 순수 Holdout 원본 패널티 ({hold_days}일 기준)]", flush=True)
    print("=" * 80, flush=True)
    print(f"Name                   : {variant_name}", flush=True)
    print(f"Validation             : Random Split (Train 60% / OPT 20% / Holdout 20%)", flush=True)
    print(f"Model                  : LightGBM Single Model", flush=True)
    print(f"Feature                : Baseline Feats + Rotation + Cloud Disagreement", flush=True)
    print(f"Learning               : Weighted L1 (cap × SMP)", flush=True)
    print(f"Hyperparameter         : Leaves=31, LR=0.06", flush=True)
    print(f"Prediction Postprocess : None (Raw Model Output)", flush=True)
    print(f"Ensemble               : None", flush=True)
    print(f"Group Algorithm        : Spatial Hedging Initialization + Greedy", flush=True)
    print(f"Weighted MAE           : {metrics['weighted_MAE_kwh']:.5f} kWh", flush=True)
    print(f"Intial Penalty         : {round(metrics['target_initial_penalty']):,} 원", flush=True)
    print(f"Optimized Penalty      : {round(metrics['target_optimized_penalty']):,} 원", flush=True)
    print(f"Theoretical Lower Bound: {round(metrics['target_lower_bound']):,} 원", flush=True)
    print("=" * 80 + "\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-iterations", type=int, default=0)
    args = ap.parse_args()
    started = time.monotonic()

    root = args.data_dir.expanduser().resolve()
    base = load_module(root / "baseline.py", "oibc_baseline_rs")
    opt = load_module(SCRIPT_DIR / "baseline_optimizer.py", "oibc_optimizer_rs")

    sites = pd.read_csv(root / "sites.csv")[
        ["site_id", "current_group", "capacity_kw", "lat", "lon", "wa_station", "wb_station"]
    ]

    gen = pd.read_parquet(root / "train" / "generation.parquet", columns=["site_id", "ts", "energy"])
    gen["ts"] = to_kst(gen.ts)
    gen = gen.dropna(subset=["energy"]).copy() 

    wr = root / "output" / "preprocessed" / "train"
    fa = pd.read_parquet(wr / "forecast_wa.parquet", columns=["wa_station", "round", "ts"] + list(base.WA_COLS))
    fb = pd.read_parquet(wr / "forecast_wb.parquet", columns=["wb_station", "round", "ts"] + list(base.WB_COLS))
    fa["ts"] = to_kst(fa.ts)
    fb["ts"] = to_kst(fb.ts)

    smp = pd.read_csv(root / "train" / "smp.csv", usecols=["ts", "smp"])
    smp["ts"] = to_kst(smp.ts)
    smp["smp"] = pd.to_numeric(smp.smp, errors="coerce").ffill()
    smp_map = smp.set_index("ts").smp

    timestamps = gen["ts"].drop_duplicates().sort_values().reset_index(drop=True)
    n_ts = len(timestamps)
    train_end_idx = int(n_ts * 0.60)
    opt_end_idx = int(n_ts * 0.80)

    train_ts = set(timestamps[:train_end_idx])
    opt_ts = set(timestamps[train_end_idx:opt_end_idx])
    hold_ts = set(timestamps[opt_end_idx:])

    train = gen.loc[gen["ts"].isin(train_ts)].copy()
    opt_actual = gen.loc[gen["ts"].isin(opt_ts)].copy()
    hold_actual = gen.loc[gen["ts"].isin(hold_ts)].copy()

    log(f"Random Split: Train={len(train_ts):,} ts, OPT={len(opt_ts):,} ts, Holdout={len(hold_ts):,} ts")

    models_lgb = {}
    
    for r in base.ROUNDS:
        d = base.features(train, sites, fa, fb, None, r)
        d, feats = engineer_clean_features(d, base)
        d = d.loc[d.elev > 0].copy()
        
        X = d[feats].astype("float32")
        y = (d.energy / d.cap).clip(0, 1.2).astype("float32")

        price = d.ts.map(smp_map)
        raw = d.cap.to_numpy(float) * price.to_numpy(float)
        sw = (raw / raw.mean()).astype("float32")

        log(f"Round {r} 슬림 모델 학습 시작 (Feats: {len(feats)})...")
        lgb_model = lgb.LGBMRegressor(**PARAMS_LGBM).fit(X, y, sample_weight=sw)
        models_lgb[r] = (lgb_model, feats)

    def predict(actual):
        preds = []
        for r in base.ROUNDS:
            lgb_model, feats = models_lgb[r]
            d = base.features(actual, sites, fa, fb, None, r)
            d, _ = engineer_clean_features(d, base)
            X = d[feats].astype("float32")
            
            p_lgb = lgb_model.predict(X)
            kwh = np.maximum(np.where(d.elev.to_numpy() > 0, p_lgb, 0.0), 0.0) * d.cap.to_numpy()
            
            one = d[["site_id", "ts"]].copy()
            one[f"p{r}"] = kwh
            preds.append(one)
            
        z = preds[0]
        for p in preds[1:]:
            z = z.merge(p, on=["site_id", "ts"], how="inner", validate="one_to_one")
        z["pred_kwh"] = z[[c for c in z.columns if c.startswith("p")]].mean(axis=1)
        return z[["site_id", "ts", "pred_kwh"]]

    opt_pred = predict(opt_actual)
    hold_pred = predict(hold_actual)

    initial = create_spatial_hedging_mask(sites)
    
    opt_matrix, opt_times, _ = residual_matrix(opt_actual, opt_pred, sites.site_id, "RS_OPT")
    hold_matrix, hold_times, _ = residual_matrix(hold_actual, hold_pred, sites.site_id, "RS_HOLD")

    weights = opt.time_weights(root, opt_times)
    selected, meta, trace = opt.greedy_best_improvement(
        opt_matrix, weights, sites, initial, batch_size=args.batch_size, max_iterations=args.max_iterations
    )

    hold_weights = opt.time_weights(root, hold_times)
    hold_initial = opt.penalty(hold_matrix, hold_weights, initial)
    hold_optimized = opt.penalty(hold_matrix, hold_weights, selected)
    hold_lb = opt.PENALTY_RATE * np.dot(hold_weights, np.abs(hold_matrix.sum(axis=1)))

    hold_days_count = len(pd.DatetimeIndex(hold_actual.ts).normalize().unique())

    q = hold_actual[["site_id", "ts", "energy"]].merge(hold_pred, on=["site_id", "ts"], validate="one_to_one")
    e = q.pred_kwh.to_numpy() - q.energy.to_numpy()
    sm = q.ts.map(smp_map).to_numpy(float)
    w_mae = float(np.dot(sm, np.abs(e)) / sm.sum())

    metrics = {
        "weighted_MAE_kwh": w_mae,
        "target_initial_penalty": float(hold_initial),
        "target_optimized_penalty": float(hold_optimized),
        "target_lower_bound": float(hold_lb),
    }

    print_summary(metrics, hold_days=hold_days_count)
    log(f"전체 실행 시간: {(time.monotonic() - started) / 60:.1f}분")

if __name__ == "__main__":
    main()
