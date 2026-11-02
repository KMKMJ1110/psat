#!/usr/bin/env python3
"""OIBC: chronological historical-CF features x four Boolean configurations.

Uses the original baseline.py's 24 features, two forecast rounds (10/17),
LightGBM L2 hyperparameters, and baseline_optimizer.py Greedy as supplied.
Does NOT edit the original files.

Chronology (right-closed delivery-hour timestamps, KST):
  - All available generation before 2024-06-01 01:00 is development data.
    The provided generation begins 2021-01-19 (NOT baseline.py TRAIN_FROM 2023-06).
  - OOF Fold 1: fit all < 2022-01-01 01:00, predict 2022 calendar year.
  - OOF Fold 2: fit all < 2023-01-01 01:00, predict 2023 calendar year.
  - OOF Fold 3: fit all < 2024-01-01 01:00, predict 2024 Jan-May.
  - Train final forecaster on ALL development data, predict 2024 Jun-2025 May.
  - Greedy group composition is learned ONLY on concatenated temporal OOF errors,
    then frozen and independently scored on the last one-year holdout.

Past-generation feature variants:
  (False,False): original L2 baseline (24 features)
  (True, False): past mean CF by site/month/right-closed hour
  (False,True): past p90 CF by site/5-degree solar-elevation bin
  (True, True): both features

Historical target encoding is causal even *inside* each model's training set.
Each training row's statistics use ONLY generation timestamps <= 17:00 KST
on the previous calendar day (contest forecast issue cutoff). For OOF/holdout
predictions, statistical tables are frozen at each forecast model's training
boundary: validation-period actuals are not used for feature lookup.

Outputs are per-variant under output/temporal_cf_features_4way/.
Execution: python -u codes/temporal_cf_features_4way.py
           python -u codes/temporal_cf_features_4way.py --only 00,10
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# The teammate's validation_timesplit.py uses the 60/20/20 helper API.
# Some copies have that API in baseline_validation.py; older copies retain
# the 80/20 helper API there and name the newer file differently.
import importlib
val = importlib.import_module("baseline_validation")
REQUIRED_VAL = ("days_of", "to_kst", "load_module", "load_weather",
                "model_metrics", "residual_matrix", "score_group", "BASELINE_PARAMS")
if not all(hasattr(val, name) for name in REQUIRED_VAL):
    try:
        val = importlib.import_module("validation_602020_baseline")
    except ImportError as exc:
        raise ImportError(
            "Need the teammate's 60/20/20 baseline_validation.py (with "
            "days_of, load_weather, model_metrics, residual_matrix, score_group), "
            "or validation_602020_baseline.py in codes/."
        ) from exc
    if not all(hasattr(val, name) for name in REQUIRED_VAL):
        raise ImportError("60/20/20 validation helper API is incomplete")

import baseline_optimizer as opt

ROOT = val.DEFAULT_ROOT
HOLDOUT_FROM = pd.Timestamp("2024-06-01 01:00", tz="Asia/Seoul")
OOF_EDGES = [
    pd.Timestamp("2022-01-01 01:00", tz="Asia/Seoul"),
    pd.Timestamp("2023-01-01 01:00", tz="Asia/Seoul"),
    pd.Timestamp("2024-01-01 01:00", tz="Asia/Seoul"),
    HOLDOUT_FROM,
]
ELEV_BIN_DEG = 5.0
FEATURES = {
    "00": (False, False),
    "10": (True, False),
    "01": (False, True),
    "11": (True, True),
}
FEAT_MH = "site_month_hour_mean_cf"
FEAT_P90 = "site_elev5_p90_cf"


def info(msg: str) -> None:
    print(f"[Temporal CF] {msg}", flush=True)


def issue_cutoff(ts: pd.Series) -> pd.Series:
    """Delivery day D's forecast issued no later than (D-1) 17:00 KST."""
    day = val.days_of(ts)
    return day - pd.Timedelta(days=1) + pd.Timedelta(hours=17)


def check_data(gen: pd.DataFrame, sites: pd.DataFrame) -> None:
    if len(sites) != 253 or sites.site_id.isna().any() or sites.site_id.duplicated().any():
        raise ValueError("Expected 253 unique sites in sites.csv")
    if gen.duplicated(["site_id", "ts"]).any():
        raise ValueError("Duplicated (site_id, ts) in generation data")
    if gen.energy.isna().any() or not np.isfinite(gen.energy.to_numpy()).all():
        raise ValueError("NaN/Inf in generation labels")
    if (~gen.site_id.isin(sites.site_id)).any():
        raise ValueError("Generation includes unknown site IDs")
    if gen.ts.min() >= OOF_EDGES[0]:
        raise ValueError("No early data for first temporal OOF training fold")
    if gen.ts.max() < HOLDOUT_FROM:
        raise ValueError("No independent last-year holdout")


def make_source_stats(base, history: pd.DataFrame, sites: pd.DataFrame,
                      need_mh: bool, need_p90: bool) -> dict[str, pd.DataFrame]:
    """Build time-indexed, cumulative historical statistics *from pre-holdout only*.

    Sources retain each past observation's timestamp. merge_asof applies the
    issue deadline and individual model's train_end cutoff; constructing these
    histories never grants model inputs access to later ground-truth values.
    """
    if not (need_mh or need_p90):
        return {}
    d = history[["site_id", "ts", "energy"]].copy()
    site_info = sites.set_index("site_id")[["capacity_kw", "lat", "lon"]]
    d["cap"] = d.site_id.map(site_info.capacity_kw)
    if d.cap.isna().any() or (d.cap <= 0).any():
        raise ValueError("Missing/nonpositive capacity in generation")
    d["cf"] = np.clip((d.energy / d.cap).to_numpy(np.float64), 0., 1.2).astype("float32")
    d["month"] = d.ts.dt.month.astype("int8")
    d["hour"] = d.ts.dt.hour.astype("int8")
    if need_p90:
        lat = d.site_id.map(site_info.lat).to_numpy(np.float64)
        lon = d.site_id.map(site_info.lon).to_numpy(np.float64)
        elev, _, _ = base.solar(pd.DatetimeIndex(d.ts), lat, lon)
        d["elev_bin"] = np.floor(elev / ELEV_BIN_DEG).astype("int16")
        del lat, lon, elev
    out = {}
    if need_mh:
        key = ["site_id", "month", "hour"]
        a = d[[*key, "ts", "cf"]].sort_values(key + ["ts"]).reset_index(drop=True)
        g = a.groupby(key, sort=False, observed=True)
        a[FEAT_MH] = (g.cf.cumsum() / (g.cumcount() + 1)).astype("float32")
        out[FEAT_MH] = a[[*key, "ts", FEAT_MH]].sort_values("ts", kind="stable").reset_index(drop=True)
        del a, g
        info("Historical site/month/hour cumulative mean created")
    if need_p90:
        key = ["site_id", "elev_bin"]
        a = d[[*key, "ts", "cf"]].sort_values(key + ["ts"]).reset_index(drop=True)
        q = a.groupby(key, sort=False, observed=True).cf.expanding().quantile(.9)
        a[FEAT_P90] = q.droplevel(list(range(len(key)))).to_numpy(np.float32)
        out[FEAT_P90] = a[[*key, "ts", FEAT_P90]].sort_values("ts", kind="stable").reset_index(drop=True)
        del a, q
        info("Historical site/elevation-bin cumulative P90 created")
    del d
    gc.collect()
    return out


def add_historical_features(df: pd.DataFrame, history_tables: dict,
                            model_train_end: pd.Timestamp, flags: tuple[bool, bool]) -> pd.DataFrame:
    """As-of historical features at min(issue deadline, training-period last ts)."""
    want_mh, want_p90 = flags
    if not (want_mh or want_p90):
        return df
    df = df.copy()
    df["_original_pos"] = np.arange(len(df), dtype=np.int64)
    df["_cf_cutoff"] = issue_cutoff(df.ts).clip(upper=model_train_end)
    if want_mh:
        df["month"] = df.ts.dt.month.astype("int8")
        df["hour"] = df.ts.dt.hour.astype("int8")
        df = _asof_feature(df, history_tables[FEAT_MH], ["site_id", "month", "hour"], FEAT_MH)
        df.drop(columns=["month"], inplace=True)
        # hour already exists in baseline FEATS: keep original baseline hour!
    if want_p90:
        df["elev_bin"] = np.floor(df.elev.to_numpy() / ELEV_BIN_DEG).astype("int16")
        df = _asof_feature(df, history_tables[FEAT_P90], ["site_id", "elev_bin"], FEAT_P90)
        df.drop(columns=["elev_bin"], inplace=True)
    df = df.sort_values("_original_pos", kind="stable")
    df.drop(columns=["_original_pos", "_cf_cutoff"], inplace=True)
    return df



def _asof_feature(df: pd.DataFrame, src: pd.DataFrame, keys: list[str], feature: str) -> pd.DataFrame:
    q = df.sort_values("_cf_cutoff", kind="stable")
    src = src.rename(columns={"ts": "_source_ts"})

    # pandas merge_asof: 양쪽 datetime 정밀도를 ns로 통일
    q["_cf_cutoff"] = q["_cf_cutoff"].dt.as_unit("ns")
    src["_source_ts"] = src["_source_ts"].dt.as_unit("ns")

    q = pd.merge_asof(
        q,
        src,
        left_on="_cf_cutoff",
        right_on="_source_ts",
        by=keys,
        direction="backward",
        allow_exact_matches=True,
    )

    q.drop(columns=["_source_ts"], inplace=True)
    return q



def build_matrix(base, gen: pd.DataFrame, sites: pd.DataFrame,
                 fa: pd.DataFrame, fb: pd.DataFrame, round_: int,
                 history_tables: dict, train_end: pd.Timestamp,
                 flags: tuple[bool, bool]) -> pd.DataFrame:
    df = base.features(gen, sites, fa, fb, None, round_)
    if len(df) != len(gen):
        raise ValueError(f"round={round_}: baseline feature merge changed row count")
    return add_historical_features(df, history_tables, train_end, flags)


def fit_model_pair(base, train: pd.DataFrame, sites: pd.DataFrame,
                   fa: pd.DataFrame, fb: pd.DataFrame, tables: dict,
                   flags: tuple[bool, bool], params: dict) -> dict:
    train_end = train.ts.max()
    use = list(base.FEATS)
    if flags[0]: use.append(FEAT_MH)
    if flags[1]: use.append(FEAT_P90)
    models = {}
    for rnd in (10, 17):
        d = build_matrix(base, train, sites, fa, fb, rnd, tables, train_end, flags)
        d = d.loc[d.elev > 0]
        X = d[use].astype("float32")
        y = (d.energy / d.cap).clip(0, 1.2).astype("float32")
        info(f"Fitting round={rnd}, features={len(use)}, rows={len(X):,}, train_end={train_end}")
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
        models[rnd] = model
        del d, X, y, model
        gc.collect()
    return models


def predict_model_pair(base, models: dict, actuals: pd.DataFrame,
                       sites: pd.DataFrame, fa: pd.DataFrame, fb: pd.DataFrame,
                       tables: dict, train_end: pd.Timestamp,
                       flags: tuple[bool, bool]) -> pd.DataFrame:
    use = list(base.FEATS)
    if flags[0]: use.append(FEAT_MH)
    if flags[1]: use.append(FEAT_P90)
    result = None
    for rnd in (10, 17):
        d = build_matrix(base, actuals, sites, fa, fb, rnd, tables, train_end, flags)
        X = d[use].astype("float32")
        cf_hat = models[rnd].predict(X)
        pred = np.maximum(np.where(d.elev.to_numpy() > 0., cf_hat, 0.), 0.) * d.cap.to_numpy()
        one = d[["site_id", "ts"]].copy()
        one[f"pred_{rnd}"] = pred
        result = one if result is None else result.merge(
            one, on=["site_id", "ts"], how="inner", validate="one_to_one")
        del d, X, cf_hat, pred, one
        gc.collect()
    if result is None or len(result) != len(actuals):
        raise ValueError("Incomplete predictions")
    result["pred_kwh"] = (result.pred_10 + result.pred_17) / 2.0
    if not np.isfinite(result.pred_kwh).all():
        raise ValueError("Invalid predictions")
    return result[["site_id", "ts", "pred_kwh"]]


def fold_iterator(development: pd.DataFrame):
    for start, stop in zip(OOF_EDGES[:-1], OOF_EDGES[1:]):
        train = development.loc[development.ts < start]
        valid = development.loc[(development.ts >= start) & (development.ts < stop)]
        if train.empty or valid.empty:
            raise ValueError(f"Empty OOF train/validation period: {start} - {stop}")
        yield start, stop, train, valid


def run_variant(variant: str, base, sites: pd.DataFrame,
                development: pd.DataFrame, holdout: pd.DataFrame,
                fa: pd.DataFrame, fb: pd.DataFrame, tables: dict,
                root: Path, out_parent: Path, params: dict):
    start_time = time.monotonic()
    flags = FEATURES[variant]
    out = out_parent / f"cf_{variant}"
    out.mkdir(parents=True, exist_ok=True)
    info(f"========== FEATURE CONFIG {variant}: month_hour={flags[0]}, elev_p90={flags[1]} ==========")
    oof_parts = []
    fold_summaries = []
    for n, (start, stop, train, valid) in enumerate(fold_iterator(development), start=1):
        info(f"OOF {n}: fit {train.ts.min()}..{train.ts.max()} ({len(train):,} rows); "
             f"predict {valid.ts.min()}..{valid.ts.max()} ({len(valid):,} rows)")
        models = fit_model_pair(base, train, sites, fa, fb, tables, flags, params)
        p = predict_model_pair(base, models, valid, sites, fa, fb, tables, train.ts.max(), flags)
        m = val.model_metrics(root, valid, p)
        fold_summaries.append({"fold": n, "start": str(start), "end": str(stop),
                               "train_rows": int(len(train)), "oof_rows": int(len(valid)),
                               "model_metrics": m})
        oof_parts.append(p)
        info(f"OOF fold {n} weighted MAE: {m['weighted_MAE_kwh']:.5f}")
        del models, p, train, valid
        gc.collect()
    oof_predictions = pd.concat(oof_parts, ignore_index=True)
    oof_actuals = development.loc[development.ts >= OOF_EDGES[0]]
    if len(oof_predictions) != len(oof_actuals):
        raise AssertionError("OOF predictions do not cover the expected source hours")
    oof_predictions.to_parquet(out / "oof_site_predictions.parquet", index=False)
    oof_metrics = val.model_metrics(root, oof_actuals, oof_predictions)
    mat_oof, ts_oof, missing_oof = val.residual_matrix(
        oof_actuals, oof_predictions, sites.site_id, f"OOF_{variant}")
    initial = opt.initial_mask(sites)
    weights_oof = opt.time_weights(root, ts_oof)
    optimal_mask, optimizer_meta, trace = opt.greedy_best_improvement(
        mat_oof, weights_oof, sites, initial)
    score_oof = val.score_group(opt, root, mat_oof, ts_oof, initial, optimal_mask)
    trace.to_csv(out / "greedy_trace.csv", index=False)
    pd.DataFrame({"site_id": sites.site_id,
                  "group": np.where(optimal_mask, "G1", "G2")}).to_csv(
        out / "composition.csv", index=False)
    del mat_oof, oof_predictions, oof_parts
    gc.collect()

    # Full development interval from 2021, with no forced baseline.TRAIN_FROM.
    info(f"FINAL fit: {development.ts.min()}..{development.ts.max()} ({len(development):,} rows)")
    final_models = fit_model_pair(base, development, sites, fa, fb, tables, flags, params)
    pred_holdout = predict_model_pair(base, final_models, holdout, sites, fa, fb,
                                      tables, development.ts.max(), flags)
    pred_holdout.to_parquet(out / "holdout_site_predictions.parquet", index=False)
    m_hold = val.model_metrics(root, holdout, pred_holdout)
    mat_h, times_h, missing_h = val.residual_matrix(
        holdout, pred_holdout, sites.site_id, f"HOLDOUT_{variant}")
    score_h = val.score_group(opt, root, mat_h, times_h, initial, optimal_mask)
    denom = score_h["W_initial_krw"] - score_h["W_lower_bound_krw"]
    score_h["improvable_cost_krw"] = score_h["W_optimized_krw"] - score_h["W_lower_bound_krw"]
    score_h["improved_percentage"] = (
        100 * (score_h["W_initial_krw"] - score_h["W_optimized_krw"]) / denom
        if abs(denom) > 1e-12 else None)
    result = {
        "config": variant,
        "feature_flags": {FEAT_MH: flags[0], FEAT_P90: flags[1]},
        "model_type": "LightGBM L2 baseline",
        "feature_count": len(base.FEATS) + sum(flags),
        "model_params": params,
        "development_min_ts": str(development.ts.min()),
        "development_max_ts": str(development.ts.max()),
        "holdout_min_ts": str(holdout.ts.min()),
        "holdout_max_ts": str(holdout.ts.max()),
        "oof_folds": fold_summaries,
        "oof_model_metrics": oof_metrics,
        "oof_group_opt": score_oof,
        "oof_missing_site_hours": missing_oof,
        "greedy_meta": optimizer_meta,
        "holdout_model_metrics": m_hold,
        "holdout_group_eval": score_h,
        "holdout_missing_site_hours": missing_h,
        "elapsed_seconds": time.monotonic() - start_time,
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
    info(f"Result {variant}: Holdout Weighted MAE={m_hold['weighted_MAE_kwh']:.5f}; "
         f"W init={score_h['W_initial_krw']:,.0f}, "
         f"W optimized={score_h['W_optimized_krw']:,.0f}, "
         f"W LB={score_h['W_lower_bound_krw']:,.0f}")
    del final_models, pred_holdout, mat_h
    gc.collect()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="00,10,01,11",
                    help="comma-separated configurations among 00,10,01,11")
    ap.add_argument("--data-dir", type=Path, default=ROOT)
    args = ap.parse_args()
    chosen = [x.strip() for x in args.only.split(",") if x.strip()]
    if not chosen or len(chosen) != len(set(chosen)) or any(x not in FEATURES for x in chosen):
        ap.error("--only must contain unique names from 00,10,01,11")
    root = args.data_dir.expanduser().resolve()
    start_time = time.monotonic()
    out_parent = root / "output" / "temporal_cf_features_4way"
    out_parent.mkdir(parents=True, exist_ok=True)
    base = val.load_module(root / "baseline.py", "oibc_base_temporal_cf")
    if tuple(base.ROUNDS) != (10, 17):
        raise ValueError("Only baseline rounds 10 and 17 supported")
    sites = pd.read_csv(root / "sites.csv")[[
        "site_id", "current_group", "capacity_kw", "lat", "lon", "wa_station", "wb_station"]]
    gen = pd.read_parquet(root / "train" / "generation.parquet", columns=["site_id", "ts", "energy"])
    gen["ts"] = val.to_kst(gen.ts)
    check_data(gen, sites)
    # Note: baseline.py TRAIN_FROM intentionally NOT applied.
    info(f"Raw data range: {gen.ts.min()} -> {gen.ts.max()}, rows={len(gen):,}")
    development = gen.loc[gen.ts < HOLDOUT_FROM].copy()
    holdout = gen.loc[gen.ts >= HOLDOUT_FROM].copy()
    del gen
    info(f"Development 2021+: rows={len(development):,}; holdout last year rows={len(holdout):,}")
    fa, fb = val.load_weather(root, base, development.ts.min())
    need_mh = any(FEATURES[x][0] for x in chosen)
    need_p90 = any(FEATURES[x][1] for x in chosen)
    tables = make_source_stats(base, development, sites, need_mh, need_p90)
    params = dict(val.BASELINE_PARAMS)
    results = []
    for x in chosen:
        result = run_variant(x, base, sites, development, holdout,
                             fa, fb, tables, root, out_parent, params)
        h = result["holdout_group_eval"]
        results.append({
            "variant": x,
            "month_hour_cf": int(FEATURES[x][0]),
            "elev_p90_cf": int(FEATURES[x][1]),
            "holdout_weighted_mae": result["holdout_model_metrics"]["weighted_MAE_kwh"],
            "holdout_W_initial": h["W_initial_krw"],
            "holdout_W_optimized": h["W_optimized_krw"],
            "holdout_W_lower_bound": h["W_lower_bound_krw"],
            "improvable_cost": h["improvable_cost_krw"],
            "improved_percentage": h["improved_percentage"],
            "oof_weighted_mae": result["oof_model_metrics"]["weighted_MAE_kwh"],
            "oof_W_optimized": result["oof_group_opt"]["W_optimized_krw"],
        })
    comp = pd.DataFrame(results)
    comp.to_csv(out_parent / "comparison.csv", index=False)
    info("========== HOLDOUT COMPARISON (not OOF/in-sample optimization) ==========")
    print(comp.to_string(index=False, float_format=lambda x: f"{x:,.5f}"), flush=True)
    info(f"Elapsed total: {time.monotonic()-start_time:,.0f}s")
    info(f"Saved: {out_parent / 'comparison.csv'}")


if __name__ == "__main__":
    main()
