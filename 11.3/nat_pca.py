#!/usr/bin/env python3
"""OIBC No.28 + one independent change: station-preserving national cloud PCA.

Put this file in data/codes/ alongside the EXISTING No.28 dependencies:
    baseline_validation.py, baseline_optimizer.py, validation_timesplit.py
and keep data/baseline.py unchanged.

Random validation (same split / pre-baseline history / LGBM+CatBoost / 50:50 / Greedy as No.28):
    python3 codes/nat_pca.py --split random

Temporal validation (same scheme as nat_feat.py):
    python3 codes/nat_pca.py --split time

Preflight test without any competition data:
    python3 codes/nat_pca.py --self-test

ONLY change relative to No.28: 4 WA cloud spatial PCs and 4 WB cloud spatial PCs
(8 additional weather-forecast-only features). PCA is fitted on forecast rows whose
TARGET timestamps occur in model_fit; GROUP_OPT/HOLDOUT timestamps never fit PCA.
No external data or services are accessed. This script does NOT generate submissions.

Do not compare a 'time' split score numerically to a 'random' split score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

A_COLS = ["wa_cloud", "wa_rain", "wa_humidity"]
B_COLS = ["wb_cloud", "wb_uv_idx", "wb_total_liq"]
NAT_FEATS = [
    "a_nat_cloud_mean", "a_nat_cloud_std", "a_nat_rain_mean", "a_cloud_dev",
    "b_nat_cloud_mean", "b_nat_cloud_std", "b_nat_uv_mean", "b_cloud_dev",
]
MODEL_A = "l1_cap_smpk"
MODEL_B = "cat_l1_cap_smpk"
BLEND_A = 0.5  # identical to No.28's recorded 50:50 LGBM:CatBoost
NO28_RANDOM_HOLDOUT_W_REFERENCE = 63_595_148  # comparison reference only


def add_time_feats(f: pd.DataFrame, station: str, cols: list[str]):
    """Unchanged from nat_feat.py: 1h prev/next and delivery-day mean per station/round."""
    f = f.sort_values([station, "round", "ts"]).copy()
    grp = f.groupby([station, "round"])
    added = []
    for c in cols:
        f[c + "_prev"] = grp[c].shift(1)
        f[c + "_next"] = grp[c].shift(-1)
        added += [c + "_prev", c + "_next"]
    f["_day"] = (f["ts"] - pd.Timedelta(hours=1)).dt.normalize()
    grpday = f.groupby([station, "round", "_day"])
    for c in cols:
        f[c + "_daymean"] = grpday[c].transform("mean")
        added.append(c + "_daymean")
    return f.drop(columns="_day"), added


def add_national(fa: pd.DataFrame, fb: pd.DataFrame):
    """Unchanged from nat_feat.py: all 8 national-summary features."""
    na = fa.groupby(["round", "ts"]).agg(
        a_nat_cloud_mean=("wa_cloud", "mean"),
        a_nat_cloud_std=("wa_cloud", "std"),
        a_nat_rain_mean=("wa_rain", "mean"),
    ).reset_index()
    nb = fb.groupby(["round", "ts"]).agg(
        b_nat_cloud_mean=("wb_cloud", "mean"),
        b_nat_cloud_std=("wb_cloud", "std"),
        b_nat_uv_mean=("wb_uv_idx", "mean"),
    ).reset_index()
    fa = fa.merge(na, on=["round", "ts"], how="left", validate="many_to_one", sort=False)
    fb = fb.merge(nb, on=["round", "ts"], how="left", validate="many_to_one", sort=False)
    fa["a_cloud_dev"] = fa["wa_cloud"] - fa["a_nat_cloud_mean"]
    fb["b_cloud_dev"] = fb["wb_cloud"] - fb["b_nat_cloud_mean"]
    return fa, fb


def _spatial_cloud_pca(
    f: pd.DataFrame,
    station_col: str,
    cloud_col: str,
    prefix: str,
    fit_timestamps: pd.DatetimeIndex,
    n_components: int,
):
    """Fit spatial PCA on TRAIN timestamps only and join same-round PC scores.

    Every row of the station-by-(round, target timestamp) matrix represents a
    geographically ordered *map* of forecast clouds. Station column identities
    and ordering are frozen at fit; all preprocessing statistics come from fit.
    """
    necessary = {station_col, cloud_col, "round", "ts"}
    if not necessary.issubset(f.columns):
        raise ValueError(f"Missing {sorted(necessary - set(f.columns))} in {prefix} forecast")
    if f[station_col].isna().any() or f[["round", "ts"]].isna().any().any():
        raise ValueError(f"{prefix}: missing station/round/ts key")
    if f.duplicated([station_col, "round", "ts"]).any():
        raise ValueError(f"{prefix}: duplicate station-round-ts keys: cannot form unambiguous map")
    if n_components < 1:
        raise ValueError("n_components must be >=1")

    series = f[["round", "ts", station_col, cloud_col]].copy()
    series[cloud_col] = pd.to_numeric(series[cloud_col], errors="coerce")
    wide = series.pivot(index=["round", "ts"], columns=station_col, values=cloud_col)
    wide = wide.sort_index()
    if len(wide) == 0:
        raise ValueError(f"{prefix}: empty forecast weather panel")

    fit_timestamps = pd.DatetimeIndex(fit_timestamps).unique()
    fit_mask = wide.index.get_level_values("ts").isin(fit_timestamps)
    n_fit = int(fit_mask.sum())
    if n_fit <= n_components:
        raise ValueError(f"{prefix}: only {n_fit} training forecast time/round rows")
    # No group_opt/holdout target ts is allowed in the PCA fitting subset.
    train_wide = wide.iloc[np.flatnonzero(fit_mask)]
    # Station inclusion also depends ONLY on train; don't learn even availability
    # of stations that appeared exclusively in evaluation forecasts.
    station_cols = train_wide.columns[train_wide.notna().any(axis=0)].tolist()
    if len(station_cols) < 2:
        raise ValueError(f"{prefix}: fewer than two stations have training data")
    wide = wide.reindex(columns=station_cols)
    train_wide = train_wide.reindex(columns=station_cols)
    x_train = train_wide.to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(x_train, axis=0)
    if not np.isfinite(med).all():
        raise ValueError(f"{prefix}: missing train-only station medians")

    # Median imputation + standardization: both fitted *only* from model_fit.
    x_train = np.where(np.isfinite(x_train), x_train, med)
    mean = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    x_train = (x_train - mean) / sd
    # n_components <= rank bounds; if fewer stations, pad unused outputs by 0.
    k_fit = min(n_components, len(station_cols), n_fit)
    pca = PCA(n_components=k_fit, svd_solver="auto", random_state=42)
    pca.fit(x_train)

    x_all = wide.to_numpy(dtype=np.float64)
    x_all = np.where(np.isfinite(x_all), x_all, med)
    x_all = (x_all - mean) / sd
    z = pca.transform(x_all)
    if not np.isfinite(z).all():
        raise ValueError(f"{prefix}: nonfinite PCA scores")

    featcols = [f"{prefix}_cloud_spc_{i+1}" for i in range(n_components)]
    data = np.zeros((len(wide), n_components), dtype=np.float32)
    data[:, :k_fit] = z.astype(np.float32)
    scores = pd.DataFrame(data, index=wide.index, columns=featcols).reset_index()

    old_n = len(f)
    # Broadcast each forecast spatial map score to each station at the same
    # (round, ts); this does not access any realized weather or generation.
    f = f.merge(scores, how="left", on=["round", "ts"], validate="many_to_one", sort=False)
    if len(f) != old_n or f[featcols].isna().any().any():
        raise ValueError(f"{prefix}: PCA feature join lost, duplicated, or failed on rows")

    model_info = {
        "station_col": station_col,
        "cloud_col": cloud_col,
        "station_columns": station_cols,
        "impute_median": med,
        "standardize_mean": mean,
        "standardize_std": sd,
        "pca": pca,
        "feature_names": featcols,
    }
    diag = {
        "provider": prefix,
        "n_train_forecast_round_hours": n_fit,
        "n_all_forecast_round_hours": len(wide),
        "n_train_stations": len(station_cols),
        "components_requested": n_components,
        "components_fitted": k_fit,
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
    }
    return f, featcols, model_info, diag


def add_spatial_pca(fa, fb, model_fit_timestamps, n_components):
    """Separate provider PCA: WA 4 + WB 4 (default), total 8 features."""
    train_ts = pd.DatetimeIndex(model_fit_timestamps).unique()
    fa, ca, ma, da = _spatial_cloud_pca(fa, "wa_station", "wa_cloud", "a", train_ts, n_components)
    fb, cb, mb, db = _spatial_cloud_pca(fb, "wb_station", "wb_cloud", "b", train_ts, n_components)
    return fa, fb, ca + cb, {"wa": ma, "wb": mb}, {"wa": da, "wb": db}


def to_mats(actual, pred, site_ids):
    """No.28 compatible: only observed labels; its established 0-fill convention."""
    df = actual[["site_id", "ts", "energy"]].merge(
        pred, on=["site_id", "ts"], how="left", validate="one_to_one"
    )
    if len(df) != len(actual) or df["pred_kwh"].isna().any():
        raise ValueError("Missing forecast on observed evaluation rows")
    P = df.pivot(index="ts", columns="site_id", values="pred_kwh").reindex(columns=list(site_ids)).sort_index()
    Y = df.pivot(index="ts", columns="site_id", values="energy").reindex(columns=list(site_ids)).sort_index()
    return P.fillna(0).to_numpy(), Y.fillna(0).to_numpy(), pd.DatetimeIndex(P.index)


def _self_test():
    # 6 stations per provider, two rounds, 40 train and 12 held-out hours.
    rng = np.random.default_rng(123)
    ts = pd.date_range("2024-01-01 01:00", periods=52, freq="h", tz="Asia/Seoul")
    fit_ts = pd.DatetimeIndex(ts[:40])
    def make(provider, station_col, cloud_col):
        rows = []
        for rnd in (10, 17):
            for j in range(6):
                cloud = (15 + j*8 + np.sin(np.arange(len(ts))/5 + j)*10
                         + rng.normal(0, 2, len(ts)))
                for idx, t in enumerate(ts):
                    rows.append({station_col: f"{provider}{j}", "round": rnd, "ts": t,
                                 cloud_col: float(cloud[idx])})
        out = pd.DataFrame(rows)
        out.loc[0, cloud_col] = np.nan
        return out
    a = make("wa", "wa_station", "wa_cloud")
    b = make("wb", "wb_station", "wb_cloud")
    a2, b2, cols, models, diag = add_spatial_pca(a, b, fit_ts, 4)
    assert len(cols) == 8 and len(a) == len(a2) and len(b) == len(b2)
    assert not a2[cols[:4]].isna().any().any() and not b2[cols[4:]].isna().any().any()
    assert set(a2.columns) == set(a.columns) | set(cols[:4])
    assert set(b2.columns) == set(b.columns) | set(cols[4:])
    # Ensure HOLDOUT cloud values cannot alter PCA training mean, medians, or components.
    a_shift = a.copy()
    a_shift.loc[~a_shift.ts.isin(fit_ts), "wa_cloud"] += 3000.0
    a3, _, _, models_shift, _ = add_spatial_pca(a_shift, b, fit_ts, 4)
    for key in ("impute_median", "standardize_mean", "standardize_std"):
        np.testing.assert_allclose(models["wa"][key], models_shift["wa"][key], atol=1e-10)
    np.testing.assert_allclose(models["wa"]["pca"].components_,
                               models_shift["wa"]["pca"].components_, atol=1e-10)
    np.testing.assert_allclose(
        a2.loc[a2.ts.isin(fit_ts), cols[:4]].to_numpy(),
        a3.loc[a3.ts.isin(fit_ts), cols[:4]].to_numpy(), atol=1e-6,
    )
    # Duplicate station-round-time must fail rather than silently averaging.
    try:
        _spatial_cloud_pca(pd.concat([a, a.iloc[[2]]]), "wa_station", "wa_cloud", "a", fit_ts, 4)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate forecast key not detected")
    print("[SELF-TEST PASS] train-only PCA, 8 features, broadcast, holdout independence, duplicate guard")
    print(json.dumps(diag, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("random", "time"), default="random")
    ap.add_argument("--components-per-provider", type=int, default=4,
                    help="WA and WB each receive this many spatial cloud PCA components (default: 4)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if not (1 <= args.components_per_provider <= 16):
        ap.error("--components-per-provider must be between 1 and 16")

    # These imports are local scripts from the exact No.28 project folder.
    import baseline_validation as val
    import baseline_optimizer as opt
    import validation_timesplit as vt
    opt.log = lambda message: None  # same as No.28
    root = val.DEFAULT_ROOT

    base = val.load_module(root / "baseline.py", "oibc_baseline_nat_pca")
    new_feats = [f"{c}_{suffix}" for c in (A_COLS + B_COLS)
                 for suffix in ("prev", "next", "daymean")]
    new_feats.append("cloud_diff")
    new_feats += NAT_FEATS
    pca_features = ([f"a_cloud_spc_{i+1}" for i in range(args.components_per_provider)] +
                    [f"b_cloud_spc_{i+1}" for i in range(args.components_per_provider)])
    new_feats += pca_features
    if len(set(new_feats)) != len(new_feats):
        raise AssertionError("Duplicate engineered feature names")
    if any(x in base.FEATS for x in new_feats):
        raise ValueError("Feature already in baseline: possible repeated mutation")
    base.FEATS = list(base.FEATS) + new_feats

    original_features = base.features
    def features_plus(*a, **k):
        d = original_features(*a, **k)
        d["cloud_diff"] = d["wa_cloud"] - d["wb_cloud"]
        return d
    base.features = features_plus

    folder = (root / "output" / "nat_pca" /
              f"{args.split}_wa{args.components_per_provider}_wb{args.components_per_provider}")
    folder.mkdir(parents=True, exist_ok=True)
    print(f"[NAT PCA] Output: {folder}")
    print(f"[NAT PCA] Feature count={len(base.FEATS)}; only addition vs No.28: {pca_features}")
    print("[NAT PCA] Ensemble: LGBM 50% + CatBoost 50%; no postprocessing; No.28 Greedy")

    sites = pd.read_csv(root / "sites.csv")[[
        "site_id", "current_group", "capacity_kw", "lat", "lon", "wa_station", "wb_station"
    ]]
    gen = pd.read_parquet(root / "train" / "generation.parquet", columns=["site_id", "ts", "energy"])
    gen["ts"] = val.to_kst(gen["ts"])
    if gen.duplicated(["site_id", "ts"]).any():
        raise ValueError("Duplicate site-hour labels in generation")
    if args.split == "random":
        baseline_gen = gen.loc[gen["ts"] >= base.TRAIN_FROM].copy()
        model_fit, group_actual, hold_actual, _ = val.split_602020(baseline_gen, val.SEED)
        old = gen.loc[gen["ts"] < base.TRAIN_FROM].copy()
        model_fit = pd.concat([old, model_fit], ignore_index=True)
        print(f"[NAT PCA] No.28 full-history fit: +{len(old):,} early site-hours (fit only)")
    else:
        model_fit, _valid, group_actual, hold_actual = vt.split_year(gen)
    del gen
    if min(len(model_fit), len(group_actual), len(hold_actual)) <= 0:
        raise ValueError("Empty train / group / hold split")
    train_ts = pd.DatetimeIndex(model_fit["ts"].drop_duplicates())
    group_ts = pd.DatetimeIndex(group_actual["ts"].drop_duplicates())
    hold_ts = pd.DatetimeIndex(hold_actual["ts"].drop_duplicates())
    if len(train_ts.intersection(group_ts)) or len(train_ts.intersection(hold_ts)) or len(group_ts.intersection(hold_ts)):
        raise ValueError("Train / group / hold target timestamps overlap: abort to prevent leakage")
    print(f"[NAT PCA] rows: train={len(model_fit):,} group={len(group_actual):,} hold={len(hold_actual):,}")

    # Load and enrich weather ONCE, then use the identical feature table for BOTH models.
    fa, fb = val.load_weather(root, base, model_fit["ts"].min())
    fa, _ = add_time_feats(fa, "wa_station", A_COLS)
    fb, _ = add_time_feats(fb, "wb_station", B_COLS)
    fa, fb = add_national(fa, fb)
    fa, fb, new_pca_cols, pca_models, diag = add_spatial_pca(
        fa, fb, train_ts, args.components_per_provider
    )
    if new_pca_cols != pca_features:
        raise AssertionError("PCA feature order not identical to model feature list")
    import joblib
    joblib.dump(pca_models, folder / "train_only_spatial_pca.joblib")
    with open(folder / "pca_diagnostics.json", "w", encoding="utf-8") as h:
        json.dump(diag, h, indent=2, ensure_ascii=False)
    print("[NAT PCA] PCA train-only diagnostics:", json.dumps(diag, ensure_ascii=False))

    eval_actual = pd.concat([group_actual, hold_actual], ignore_index=True)
    preds = []
    for variant in (MODEL_A, MODEL_B):
        print(f"[NAT PCA] Fitting {variant} (No.28 params unchanged)...", flush=True)
        models, rounds, _params = vt.fit_models_weighted(base, sites, model_fit, fa, fb, variant)
        pred = val.predict_eval(base, models, rounds, sites, eval_actual, fa, fb)
        if pred.duplicated(["site_id", "ts"]).any() or len(pred) != len(eval_actual):
            raise ValueError(f"{variant}: unexpected evaluation prediction keys/row count")
        if not np.isfinite(pred["pred_kwh"].to_numpy()).all():
            raise ValueError(f"{variant}: non-finite predictions")
        pred.to_parquet(folder / (variant + "_eval_pred.parquet"), index=False)
        preds.append(pred)
        del models
    del model_fit, fa, fb

    both = preds[0].merge(preds[1], on=["site_id", "ts"], suffixes=("_a", "_b"), validate="one_to_one")
    if len(both) != len(eval_actual):
        raise ValueError("LGBM / CatBoost prediction key mismatch")
    pred = both[["site_id", "ts"]].copy()
    pred["pred_kwh"] = BLEND_A * both["pred_kwh_a"] + (1.0 - BLEND_A) * both["pred_kwh_b"]
    pred.to_parquet(folder / "ensemble_50_50_eval_pred.parquet", index=False)
    pred_h = pred.merge(hold_actual[["site_id", "ts"]], on=["site_id", "ts"], validate="one_to_one")
    m_h = val.model_metrics(root, hold_actual, pred_h)
    Pg, Yg, tg = to_mats(group_actual, pred, sites.site_id)
    Ph, Yh, th = to_mats(hold_actual, pred, sites.site_id)
    wg = opt.time_weights(root, tg)
    init_mask = opt.initial_mask(sites)
    print("[NAT PCA] Greedy best improvement on GROUP_OPT (No.28 method)...", flush=True)
    optimized, _meta, _trace = opt.greedy_best_improvement(Pg - Yg, wg, sites, init_mask)
    score_h = val.score_group(opt, root, Ph - Yh, th, init_mask, optimized)
    summary = {
        "experiment": "No.28 + national station-preserving cloud PCA only",
        "split": args.split,
        "seed": int(val.SEED) if args.split == "random" else None,
        "pca_components_per_provider": args.components_per_provider,
        "pca_feature_count": len(pca_features),
        "ensemble": "50% LGBM + 50% CatBoost",
        "postprocess": "none",
        "weighted_MAE": float(m_h["weighted_MAE_kwh"]),
        "initial_W": float(score_h["W_initial_krw"]),
        "optimized_W": float(score_h["W_optimized_krw"]),
        "lower_bound_W": float(score_h["W_lower_bound_krw"]),
        "hours": int(score_h["hours"]),
        "penalty_per_hour": float(score_h["W_optimized_krw"] / score_h["hours"]),
        "pca_diagnostics": diag,
    }
    with open(folder / "summary.json", "w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2, ensure_ascii=False)
    pd.DataFrame([{
        "model": "No.28 + Cloud PCA (4 WA + 4 WB by default)",
        "weighted_MAE": round(summary["weighted_MAE"], 5),
        "initial": round(summary["initial_W"]),
        "optimized": round(summary["optimized_W"]),
        "lower_bound": round(summary["lower_bound_W"]),
        "W_per_hour": round(summary["penalty_per_hour"], 2),
    }]).to_csv(folder / "result.csv", index=False)
    print("=" * 74)
    print(f"[NAT PCA] Holdout MAE={summary['weighted_MAE']:.5f}")
    print(f"[NAT PCA] Holdout initial={summary['initial_W']:,.2f} W; optimized={summary['optimized_W']:,.2f} W")
    print(f"[NAT PCA] Holdout lower bound={summary['lower_bound_W']:,.2f} W; hours={summary['hours']}")
    if args.split == "random":
        delta = summary["optimized_W"] - NO28_RANDOM_HOLDOUT_W_REFERENCE
        print(f"[NAT PCA] Difference from recorded No.28 optimized W (positive is worse): {delta:+,.2f} KRW")
    print(f"[NAT PCA] saved: {folder / 'result.csv'}")


if __name__ == "__main__":
    main()
