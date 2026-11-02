"""Station-error graph -> median weather correction -> unchanged generation baseline.

Run from Spyder: %run codes/spatial_weather_validation.py
Requires the ORIGINAL baseline_validation.py and baseline_optimizer.py in codes/.
No held-out observed weather or generation is used for fitting or graph selection.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "codes" else HERE
WEATHER_PARAMS = dict(objective="regression_l1", n_estimators=200,
                      learning_rate=0.05, num_leaves=31,
                      min_child_samples=200, n_jobs=-1, verbosity=-1,
                      random_state=42, deterministic=True, force_col_wise=True)
CASES = {"wa": [("temperature", "wa_temp"), ("cloud", "wa_cloud"),
                ("humidity", "wa_humidity")],
         "wb": [("temperature", "wb_temp"), ("cloud", "wb_cloud"),
                ("humidity", "wb_rel_hum")]}


def log(s):
    print(f"[WEATHER GRAPH] {s}", flush=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def day_of(ts):
    return (ts - pd.Timedelta(hours=1)).dt.normalize()


def valid(x, variable):
    return np.isfinite(x) & ((x >= -50) & (x <= 60) if variable == "temperature"
                             else (x >= 0) & (x <= 100))


def error_graph(frame, fit_mask, ids, min_overlap=1000, threshold=0.2, k=5):
    """W[target, source], positive error correlations; no compulsory neighbors."""
    d = frame.loc[fit_mask & frame.paired, ["station", "ts", "raw", "obs"]].copy()
    d["error"] = d.obs - d.raw
    mid = d.ts - pd.Timedelta(minutes=30)
    # Remove station x month x hour climatology using this fit partition ONLY.
    d["error"] -= d.groupby(["station", mid.dt.month, mid.dt.hour])["error"].transform("mean")
    p = d.pivot(index="ts", columns="station", values="error").reindex(columns=ids)
    corr = p.corr(min_periods=min_overlap).to_numpy()
    count = p.notna().to_numpy(dtype=np.int32)
    overlap = count.T @ count
    allowed = np.isfinite(corr) & (corr >= threshold) & (overlap >= min_overlap)
    np.fill_diagonal(allowed, False)
    directed = np.zeros_like(allowed)
    for i in range(len(ids)):
        candidates = np.flatnonzero(allowed[i])
        order = np.argsort(-corr[i, candidates], kind="stable")
        directed[i, candidates[order[:k]]] = True
    edges = directed | directed.T
    weight = np.where(edges, np.nan_to_num(corr) ** 2, 0.0)
    den = weight.sum(axis=1, keepdims=True)
    weight = np.divide(weight, den, out=np.zeros_like(weight), where=den > 0)
    assert np.all(np.diag(weight) == 0)
    assert np.allclose(weight.sum(axis=1)[den[:, 0] > 0], 1)
    return weight, corr, overlap


def design(frame, ids, weight):
    station_index = pd.Index(ids).get_indexer(frame.station)
    if (station_index < 0).any():
        raise ValueError("Unexpected station")
    mid = frame.ts - pd.Timedelta(minutes=30)
    hour = mid.dt.hour.to_numpy() + 0.5
    season = (mid.dt.dayofyear.to_numpy() - 1 + hour / 24) / 365.25
    own = pd.DataFrame({"raw": frame.raw.to_numpy(dtype=np.float32),
                        "hour_sin": np.sin(2*np.pi*hour/24),
                        "hour_cos": np.cos(2*np.pi*hour/24),
                        "season_sin": np.sin(2*np.pi*season),
                        "season_cos": np.cos(2*np.pi*season),
                        "station_code": station_index}).astype("float32")
    times = pd.Index(frame.ts.unique()).sort_values()
    ti = times.get_indexer(frame.ts)
    grid = np.full((len(times), len(ids)), np.nan)
    grid[ti, station_index] = frame.raw.to_numpy()
    good = np.isfinite(grid)
    den = good.astype(float) @ weight.T
    num = np.nan_to_num(grid) @ weight.T
    neighbor = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    neighbor = neighbor[ti, station_index]
    graph = own.copy()
    graph["neighbor_contrast"] = np.where(np.isfinite(neighbor), neighbor-frame.raw, 0)
    graph["neighbor_available_weight"] = den[ti, station_index]
    graph["neighbor_missing"] = (~np.isfinite(neighbor)).astype("float32")
    return own, graph.astype("float32")


def read_observed(root, prefix, columns, v):
    p = root / "train" / f"observed_{prefix}.parquet"
    if not p.is_file():
        raise FileNotFoundError(f"Required observed weather: {p}")
    station = prefix + "_station"
    obs = pd.read_parquet(p, columns=[station, "ts"] + columns)
    obs["ts"] = v.to_kst(obs.ts)
    # Collapse exact duplicate values; conflicting station-hours become missing.
    grouped = obs.groupby([station, "ts"], sort=False)
    conflict = grouped[columns].nunique(dropna=False).gt(1)
    obs = grouped[columns].first().mask(conflict).reset_index()
    return obs


def prepare(root, stage, args, v, base):
    gen = pd.read_parquet(root / "train/generation.parquet", columns=["ts"])
    gen["ts"] = v.to_kst(gen.ts)
    gen = gen.loc[gen.ts >= base.TRAIN_FROM]
    days = pd.DatetimeIndex(day_of(gen.ts).drop_duplicates().sort_values())
    perm = np.random.default_rng(args.seed).permutation(len(days))
    n_fit, n_opt = int(np.floor(.6*len(days))), int(np.floor(.2*len(days)))
    fit_days = days[perm[:n_fit]]
    split = pd.DataFrame({"day_kst": days.strftime("%Y-%m-%d"), "split": np.select(
        [days.isin(fit_days), days.isin(days[perm[n_fit:n_fit+n_opt]])],
        ["model_train", "group_opt"], default="holdout")})
    existing = root / "output/validation_602020/split602020_baseline/split_days.csv"
    if existing.is_file():
        pd.testing.assert_frame_equal(pd.read_csv(existing), split)
        log("Existing baseline split matches exactly.")
    split.to_csv(stage / "split_days.csv", index=False)
    # Day-level cross-fitting: every forecast on a fit day receives OOF correction.
    fold_map = {day: i % args.folds for i, day in enumerate(fit_days)}
    roles = dict(zip(pd.to_datetime(split.day_kst).dt.tz_localize("Asia/Seoul"), split.split))
    del gen
    gc.collect()
    metrics, audit = [], []
    for prefix, cases in CASES.items():
        cols = list(base.WA_COLS if prefix == "wa" else base.WB_COLS)
        station = prefix + "_station"
        path = root / "output/preprocessed/train" / f"forecast_{prefix}.parquet"
        names = pq.read_schema(path).names
        flags = ["imputed_"+c for _, c in cases if "imputed_"+c in names]
        forecast = pd.read_parquet(path, columns=[station, "round", "ts"]+cols+flags)
        forecast["ts"] = v.to_kst(forecast.ts)
        forecast = forecast.loc[(forecast.ts >= base.TRAIN_FROM) & forecast["round"].isin((10, 17))].reset_index(drop=True)
        if forecast.duplicated([station, "round", "ts"]).any():
            raise ValueError(f"Duplicate {prefix} station/round/hour")
        obs = read_observed(root, prefix, [c for _, c in cases], v)
        ids = sorted(forecast[station].unique())
        corrected = {kind: forecast[[station, "round", "ts"]+cols].copy()
                     for kind in ("own", "graph")}
        for r in (10, 17):
            row_ids = np.flatnonzero(forecast["round"].to_numpy() == r)
            part = forecast.iloc[row_ids].merge(obs, on=[station, "ts"], how="left",
                                                 suffixes=("", "_obs"), validate="many_to_one")
            day = day_of(part.ts)
            fold = day.map(fold_map).fillna(-1).to_numpy(dtype=int)
            fit = fold >= 0
            role = day.map(roles).to_numpy()
            for variable, col in cases:
                label = f"{prefix}_{variable}_round{r}"
                frame = pd.DataFrame({"station": part[station], "ts": part.ts,
                                      "raw": part[col].where(valid(part[col], variable)),
                                      "obs": part[col+"_obs"].where(valid(part[col+"_obs"], variable))})
                frame["paired"] = frame.raw.notna() & frame.obs.notna()
                flag = "imputed_" + col
                if flag in part:
                    frame["paired"] &= ~part[flag].fillna(False).astype(bool)
                predictions = {kind: part[col].to_numpy(dtype=float).copy() for kind in corrected}
                # OOF models for generation fit rows, then full-fit models for both evaluation splits.
                for f in list(range(args.folds)) + [-1]:
                    train = fit & (fold != f) if f >= 0 else fit
                    target = fold == f if f >= 0 else ~fit
                    assert not np.any(train & target)
                    w, corr, overlap = error_graph(frame, train, ids)
                    own_x, graph_x = design(frame, ids, w)
                    candidate = np.flatnonzero(train & frame.paired.to_numpy())
                    if len(candidate) < 1000:
                        raise ValueError(f"Too few paired training observations: {label}, fold={f}")
                    # Identical sample for own/graph, fixed without evaluation labels.
                    rng = np.random.default_rng(args.seed + f + 1)
                    if len(candidate) > args.max_weather_rows:
                        candidate = np.sort(rng.choice(candidate, args.max_weather_rows, replace=False))
                    target_ids = np.flatnonzero(target & frame.raw.notna().to_numpy())
                    y = (frame.obs-frame.raw).to_numpy()[candidate]
                    log(f"{label}, fold={f}: fit {len(candidate):,}, predict {len(target_ids):,}")
                    for kind, x in (("own", own_x), ("graph", graph_x)):
                        model = lgb.LGBMRegressor(**{**WEATHER_PARAMS, "random_state": args.seed})
                        model.fit(x.iloc[candidate], y, categorical_feature=["station_code"])
                        if len(target_ids):
                            pred = frame.raw.to_numpy()[target_ids] + model.predict(x.iloc[target_ids])
                            if variable != "temperature":
                                pred = np.clip(pred, 0, 100)
                            predictions[kind][target_ids] = pred
                        if f == -1:
                            # Python handles Unicode Windows paths; avoid native fopen.
                            (stage / f"{label}_{kind}.txt").write_text(
                                model.booster_.model_to_string(), encoding="utf-8")
                        del model
                    audit.append({"case": label, "fold": f, "n_train_days": int(day[train].nunique()),
                                  "n_target_days": int(day[target].nunique()),
                                  "paired_fit_rows": len(candidate), "target_rows": len(target_ids),
                                  "directed_nonzero_weights": int((w > 0).sum()),
                                  "isolated_stations": int((w.sum(axis=1) == 0).sum())})
                    if f == -1:
                        pd.DataFrame(w, index=ids, columns=ids).to_csv(stage / f"{label}_weights.csv", index_label="target_station")
                        pd.DataFrame(corr, index=ids, columns=ids).to_csv(stage / f"{label}_correlations.csv", index_label="station")
                        pd.DataFrame(overlap, index=ids, columns=ids).to_csv(stage / f"{label}_overlap.csv", index_label="station")
                    del own_x, graph_x
                for kind in corrected:
                    corrected[kind].loc[row_ids, col] = predictions[kind]
                for split_name in ("model_train", "group_opt", "holdout"):
                    ix = (role == split_name) & frame.paired.to_numpy()
                    if not ix.any():
                        continue
                    for kind, pred in [("raw", part[col].to_numpy())] + list(predictions.items()):
                        err = pred[ix] - frame.obs.to_numpy()[ix]
                        if not np.isfinite(err).all():
                            raise ValueError("Nonfinite weather score")
                        metrics.append({"case": label, "split": split_name, "model": kind,
                                        "n": int(ix.sum()), "MAE": np.abs(err).mean(),
                                        "RMSE": np.sqrt(np.mean(err**2)), "bias": err.mean()})
                pd.DataFrame(metrics).to_csv(stage / "weather_metrics.csv", index=False)
                pd.DataFrame(audit).to_csv(stage / "crossfit_audit.csv", index=False)
                gc.collect()
        for kind, data in corrected.items():
            data.to_parquet(stage / f"{kind}_forecast_{prefix}.parquet", index=False)
        del part, forecast, obs, corrected
        gc.collect()
    return split


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--max-weather-rows", type=int, default=50000)
    ap.add_argument("--mode", choices=("both", "own", "graph", "prepare"), default="both")
    args = ap.parse_args(argv)
    if args.folds < 2 or args.max_weather_rows < 1000:
        ap.error("folds >= 2 and max-weather-rows >= 1000 required")
    root = args.data_dir.expanduser().resolve()
    v = load_module(HERE / "baseline_validation.py", "spatial_validation_base")
    base = load_module(root / "baseline.py", "spatial_generation_base")
    original_weather = v.load_weather
    stage = root / "output/weather_error_graph"
    stage.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "folds": args.folds, "max_weather_rows": args.max_weather_rows,
                "weather_params": {**WEATHER_PARAMS, "random_state": args.seed},
                "baseline_sha256": sha(root / "baseline.py"),
                "validation_sha256": sha(HERE / "baseline_validation.py"),
                "optimizer_sha256": sha(HERE / "baseline_optimizer.py"),
                "runner_sha256": sha(Path(__file__)),
                "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in ("lightgbm", "numpy", "pandas", "pyarrow", "scikit-learn")},
                "graph": "per provider/variable/round; fit-only station-month-hour centered errors; rho>=.2, overlap>=1000, top5 union, rho^2 row normalization; isolated rows zero",
                "training": f"{args.folds}-fold delivery-day OOF weather features for generation fit; full-fit correction for group_opt and holdout; no extra historical training",
                "comparison": "Median LightGBM weather correction, not previous GAM; downstream baseline unchanged"}
    # No implicit cache reuse: inputs may have been replaced since a previous run.
    prepare(root, stage, args, v, base)
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.mode == "prepare":
        return
    results = []
    existing = root / "output/validation_602020/split602020_baseline/summary.json"
    if existing.is_file():
        results.append(("baseline", json.loads(existing.read_text(encoding="utf-8"))))
    modes = ("own", "graph") if args.mode == "both" else (args.mode,)
    for kind in modes:
        log(f"Generation experiment: {kind}; generation hyperparameters unchanged.")
        def load_corrected(data_root, baseline, minimum, kind=kind):
            frames = []
            for prefix in ("wa", "wb"):
                d = pd.read_parquet(stage / f"{kind}_forecast_{prefix}.parquet")
                frames.append(d.loc[d.ts >= minimum].copy())
            return tuple(frames)
        v.load_weather = load_corrected
        v.RUN_NAMES["baseline"] = f"split602020_weather_median_{kind}"
        try:
            s = v.run("baseline", ["--data-dir", str(root), "--seed", str(args.seed)])
        finally:
            v.load_weather = original_weather
        out = root / "output/validation_602020" / v.RUN_NAMES["baseline"]
        s["experiment"] = f"weather_median_{kind}"
        s["weather_source"] = str(stage / f"{kind}_forecast_{{wa,wb}}.parquet")
        s["weather_manifest"] = str(stage / "manifest.json")
        (out / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        results.append((s["experiment"], s))
        rows = []
        for name, summary in results:
            m = summary["model_metrics_holdout"]
            g = summary["group_evaluation_independent_holdout"]
            rows.append({"experiment": name, "MAE_kwh": m["MAE_kwh"],
                         "weighted_MAE_kwh": m["weighted_MAE_kwh"],
                         "RMSE_kwh": m["RMSE_kwh"], **g})
        pd.DataFrame(rows).to_csv(stage / "generation_comparison.csv", index=False)
        gc.collect()
    log(f"DONE. Compare results in {stage / 'generation_comparison.csv'}")


if __name__ == "__main__":
    main()
