#!/usr/bin/env python3
"""OIBC: Submit_29 + Categorical Site_ID + FullHistory + Ensemble(3:1)
Validation: Full History Train + Delivery-Day Split
Model: LightGBM (75%) + CatBoost (25%)
Features: Submit_29 + Categorical site_id + Solar Elev + Rotation
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import lightgbm as lgb
import catboost as cb
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DEFAULT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "codes" else SCRIPT_DIR
SEED = 42

PARAMS_LGBM = dict(
    n_estimators=600,
    learning_rate=0.06,
    num_leaves=63,             # site_id 범주 구분을 위한 분기 노드 확장
    min_child_samples=200,
    reg_alpha=0.5,
    reg_lambda=1.0,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    objective="regression_l1",  # 표준 L1 손실함수 (MAE)
    verbose=-1,
    n_jobs=-1,
)

PARAMS_CAT = dict(
    iterations=600,
    learning_rate=0.06,
    depth=6,
    loss_function='MAE',
    verbose=0,
    thread_count=-1
)

def log(msg: str) -> None:
    print(f"[BEST-3to1] {msg}", flush=True)

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

def delivery_day(ts: pd.Series) -> pd.Series:
    return (ts - pd.Timedelta(hours=1)).dt.normalize()

# =====================================================================
# [1] Submit_29 피처 + 정석 범주형 site_id + 태양고도 + Nat-Cloud
# =====================================================================
def engineer_submit29_roadmap_features(df: pd.DataFrame, base):
    out = df.copy()
    out = out.sort_values(by=["site_id", "ts"])
    
    wa_cols = list(base.WA_COLS)
    wb_cols = list(base.WB_COLS)
    
    # 1. 보간 및 시차/이동평균 (Submit_29 확장 피처)
    for col in wa_cols + wb_cols:
        if col in out.columns:
            out[col] = out.groupby("site_id")[col].transform(
                lambda x: pd.to_numeric(x, errors="coerce").interpolate(method="linear").ffill().bfill()
            )
            
    out["date"] = out["ts"].dt.date
    for c in wa_cols + wb_cols:
        out[f"{c}_prev"] = out.groupby("site_id")[c].shift(1)
        out[f"{c}_next"] = out.groupby("site_id")[c].shift(-1)
        out[f"{c}_daymean"] = out.groupby(["site_id", "date"])[c].transform("mean")
    out.drop(columns=["date"], inplace=True)
    
    midnight = out["ts"].dt.hour == 0
    for c in wa_cols + wb_cols:
        out.loc[midnight, f"{c}_next"] = np.nan
        
    # 2. 전국 평균 기상 피처
    nat_wa = out.groupby("ts")[wa_cols].mean().add_prefix("nat_")
    nat_wb = out.groupby("ts")[wb_cols].mean().add_prefix("nat_")
    out = out.merge(nat_wa, on="ts", how="left").merge(nat_wb, on="ts", how="left")
    
    # 3. 회전 및 기상 차이
    out["cloud_diff"] = out["wa_cloud"] - out["wb_cloud"]
    out["cloud_disagreement"] = out["cloud_diff"].abs().astype("float32")
    out["lat_lon_sum"] = (out["lat"] + out["lon"]).astype("float32")
    out["lat_lon_diff"] = (out["lat"] - out["lon"]).astype("float32")
    
    # 4. site_id를 정석 category 타입으로 지정
    out["site_id_cat"] = out["site_id"].astype("category")
    
    added_feats = []
    for c in wa_cols + wb_cols:
        added_feats.extend([f"{c}_prev", f"{c}_next", f"{c}_daymean", f"nat_{c}"])
        
    user_feats = [
        "cloud_diff", "cloud_disagreement", "lat_lon_sum", "lat_lon_diff", "site_id_cat"
    ]
    if "elev" in out.columns and "elev" not in base.FEATS:
        user_feats.append("elev")
        
    final_feats = list(base.FEATS) + added_feats + user_feats
    return out.sort_index(), final_feats


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


def print_summary(metrics: dict, hold_days: int, variant_name: str = "Submit29 + Categorical Site_ID + FullHistory + Ensemble(3:1)") -> None:
    print("\n" + "=" * 80, flush=True)
    print(f"📊 [실험 결과 요약 - 순수 Holdout 원본 패널티 ({hold_days}일 기준)]", flush=True)
    print("=" * 80, flush=True)
    print(f"Name                   : {variant_name}", flush=True)
    print(f"Validation             : Full History Train + Delivery-Day Split (Holdout 20%)", flush=True)
    print(f"Model                  : LightGBM (75%) + CatBoost (25%)", flush=True)
    print(f"Feature                : Submit_29 Feats + Categorical site_id + Solar Elev + Rotation", flush=True)
    print(f"Learning               : Weighted L1 (cap × SMP)", flush=True)
    print(f"Prediction Postprocess : None (Raw Model Output)", flush=True)
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
    
    if hasattr(base, "TRAIN_FROM"):
        historical_gen = gen.loc[gen.ts < base.TRAIN_FROM].copy()
        base_gen = gen.loc[gen.ts >= base.TRAIN_FROM].copy()
    else:
        historical_gen = pd.DataFrame()
        base_gen = gen.copy()

    base_gen["day_kst"] = delivery_day(base_gen.ts)
    days = pd.DatetimeIndex(base_gen["day_kst"].drop_duplicates().sort_values())
    
    perm = np.random.default_rng(SEED).permutation(len(days))
    n_fit = int(np.floor(0.60 * len(days)))
    n_opt = int(np.floor(0.20 * len(days)))
    
    fit_days = days[perm[:n_fit]]
    opt_days = days[perm[n_fit:n_fit + n_opt]]
    holdout_days = days[perm[n_fit + n_opt:]]

    train_60 = base_gen.loc[base_gen.day_kst.isin(fit_days)].drop(columns=["day_kst"]).copy()
    train = pd.concat([historical_gen, train_60], ignore_index=True)
    
    opt_actual = base_gen.loc[base_gen.day_kst.isin(opt_days)].drop(columns=["day_kst"]).copy()
    hold_actual = base_gen.loc[base_gen.day_kst.isin(holdout_days)].drop(columns=["day_kst"]).copy()

    log(f"Data Split: Full History Train = {len(train):,} rows / OPT = {len(opt_days)} days / Holdout = {len(holdout_days)} days")

    wr = root / "output" / "preprocessed" / "train"
    fa = pd.read_parquet(wr / "forecast_wa.parquet", columns=["wa_station", "round", "ts"] + list(base.WA_COLS))
    fb = pd.read_parquet(wr / "forecast_wb.parquet", columns=["wb_station", "round", "ts"] + list(base.WB_COLS))
    fa["ts"] = to_kst(fa.ts)
    fb["ts"] = to_kst(fb.ts)

    smp = pd.read_csv(root / "train" / "smp.csv", usecols=["ts", "smp"])
    smp["ts"] = to_kst(smp.ts)
    smp["smp"] = pd.to_numeric(smp.smp, errors="coerce").ffill()
    smp_map = smp.set_index("ts").smp

    models_lgb = {}
    models_cat = {}
    
    cat_features = ["site_id_cat"]
    
    for r in base.ROUNDS:
        d = base.features(train, sites, fa, fb, None, r)
        d, feats = engineer_submit29_roadmap_features(d, base)
        d = d.loc[d.elev > 0].copy()
        
        X = d[feats]
        y = (d.energy / d.cap).clip(0, 1.2).astype("float32")

        price = d.ts.map(smp_map)
        raw = d.cap.to_numpy(float) * price.to_numpy(float)
        sw = (raw / raw.mean()).astype("float32")

        log(f"Round {r} 3:1 앙상블 학습 시작 (Feats: {len(feats)})...")
        
        # LightGBM 범주형 명시
        lgb_model = lgb.LGBMRegressor(**PARAMS_LGBM).fit(
            X, y, sample_weight=sw, categorical_feature=cat_features
        )
        # CatBoost 범주형 명시
        X_cat = X.copy()
        X_cat["site_id_cat"] = X_cat["site_id_cat"].astype(str)
        cat_model = cb.CatBoostRegressor(**PARAMS_CAT).fit(
            X_cat, y, sample_weight=sw, cat_features=cat_features
        )
        
        models_lgb[r] = (lgb_model, feats)
        models_cat[r] = (cat_model, feats)

    def predict(actual):
        preds = []
        for r in base.ROUNDS:
            lgb_model, feats = models_lgb[r]
            cat_model, _ = models_cat[r]
            
            d = base.features(actual, sites, fa, fb, None, r)
            d, _ = engineer_submit29_roadmap_features(d, base)
            X = d[feats]
            
            p_lgb = lgb_model.predict(X)
            
            X_cat = X.copy()
            X_cat["site_id_cat"] = X_cat["site_id_cat"].astype(str)
            p_cat = cat_model.predict(X_cat)
            
            # [3:1 앙상블 비율]
            p_ens = (0.75 * p_lgb) + (0.25 * p_cat)
            
            kwh = np.maximum(np.where(d.elev.to_numpy() > 0, p_ens, 0.0), 0.0) * d.cap.to_numpy()
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

    hold_days_count = len(holdout_days)

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
