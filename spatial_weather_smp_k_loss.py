"""Reuse graph-corrected OOF weather; train generation with SMP*k*capacity L1.

Spyder: %run codes/spatial_weather_smp_k_loss.py
Optional unweighted-L1 control: add --loss l1
Requires the ORIGINAL baseline_validation.py and baseline_optimizer.py in codes/.
No weather model or graph is refitted. No original experiment is overwritten.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "codes" else HERE


def load_module(path, name):
    if not path.is_file():
        raise FileNotFoundError(f"Required code missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_weights(df, prices):
    """Relative weights for CF loss proportional to SMP*k*kWh absolute loss."""
    price = df.ts.map(prices).to_numpy(np.float64)
    capacity = df.cap.to_numpy(np.float64)
    k = np.where(df.ts.dt.hour.between(11, 15).to_numpy(), 3.0, 1.0)
    if (not np.isfinite(price).all() or (price < 0).any()
            or not np.isfinite(capacity).all() or (capacity <= 0).any()):
        raise ValueError("Missing/nonfinite/negative SMP or invalid capacity on training rows")
    raw = price * k * capacity
    if not np.isfinite(raw).all() or raw.mean() <= 0:
        raise ValueError("Training weights must be finite with positive mean")
    return raw / raw.mean(), raw


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-iterations", type=int, default=0)
    ap.add_argument("--loss", choices=("smp_k", "l1"), default="smp_k")
    args = ap.parse_args(argv)
    if args.batch_size < 1 or args.max_iterations < 0:
        ap.error("Invalid Greedy settings")
    root = args.data_dir.expanduser().resolve()
    stage = root / "output/weather_error_graph"
    meta_path = stage / "manifest.json"
    if not meta_path.is_file():
        raise FileNotFoundError("Run spatial_weather_validation.py to completion first: missing weather manifest")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("seed") != args.seed:
        raise ValueError("Seed differs from saved OOF weather. Reusing these features would invalidate the split.")
    for field, path in (("baseline_sha256", root / "baseline.py"),
                        ("validation_sha256", HERE / "baseline_validation.py"),
                        ("optimizer_sha256", HERE / "baseline_optimizer.py")):
        if meta.get(field) != sha(path):
            raise ValueError(f"Code changed since graph experiment: {path.name}. Restore the same original code.")
    v = load_module(HERE / "baseline_validation.py", "graph_smp_k_validation")
    expected_split = pd.read_csv(stage / "split_days.csv")
    suffix = "l1_smp_k" if args.loss == "smp_k" else "l1"
    run_name = f"split602020_weather_median_graph_{suffix}"
    out = root / "output/validation_602020" / run_name
    v.RUN_NAMES["baseline"] = run_name
    original_split = v.split_602020
    original_weather = v.load_weather
    original_fit = v.fit_models
    audit, inputs = [], {}

    def checked_split(gen, seed):
        fit, group, hold, split = original_split(gen, seed)
        pd.testing.assert_frame_equal(split, expected_split)
        v.log("Saved graph OOF split matches exactly; no weather refitting")
        return fit, group, hold, split

    def cached_weather(data_root, base, minimum):
        frames = []
        for prefix, cols in (("wa", base.WA_COLS), ("wb", base.WB_COLS)):
            path = stage / f"graph_forecast_{prefix}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Missing saved graph-corrected weather: {path}")
            station = prefix + "_station"
            data = pd.read_parquet(path, columns=[station, "round", "ts"] + list(cols))
            data["ts"] = v.to_kst(data.ts)
            if data.duplicated([station, "round", "ts"]).any():
                raise ValueError(f"Duplicate cached {prefix} forecast keys")
            stat = path.stat()
            inputs[prefix] = {"path": str(path), "size_bytes": stat.st_size,
                              "mtime_ns": stat.st_mtime_ns, "rows": len(data)}
            frames.append(data.loc[data.ts >= minimum].copy())
            v.log(f"REUSE corrected {prefix} forecasts: {len(frames[-1]):,} rows")
        return tuple(frames)

    def fit_models(base, sites, fit, fa, fb, variant):
        rounds = (10, 17)
        params = {**v.BASELINE_PARAMS, "objective": "regression_l1"}
        smp = pd.read_csv(root / "train/smp.csv", usecols=["ts", "smp"])
        smp["ts"] = v.to_kst(smp.ts)
        if smp.ts.duplicated().any():
            raise ValueError("Duplicate SMP timestamps")
        prices = pd.to_numeric(smp.set_index("ts").smp, errors="coerce")
        models = {}
        for r in rounds:
            d = base.features(fit, sites, fa, fb, None, r)
            if len(d) != len(fit):
                raise ValueError(f"Round {r}: feature join changed row count")
            d = d.loc[d.elev > 0]
            x = d[base.FEATS].astype("float32")
            y = (d.energy / d.cap).clip(0, 1.2).astype("float32")
            model = lgb.LGBMRegressor(**params)
            if args.loss == "smp_k":
                weight, raw_weight = training_weights(d, prices)
                v.log(f"Fitting round={r}, X={x.shape}, weighted L1: SMP*k*capacity; mean-normalized")
                model.fit(x, y, sample_weight=weight)
            else:
                weight = np.ones(len(d))
                raw_weight = weight
                v.log(f"Fitting round={r}, X={x.shape}, unweighted L1 control")
                model.fit(x, y)
            error = model.predict(x) - y.to_numpy()
            v.log(f"round={r} train normalized MAE={np.abs(error).mean():.6f}; "
                  f"weighted normalized MAE={np.average(np.abs(error), weights=weight):.6f}")
            audit.append({"round": r, "loss": args.loss, "fit_rows": len(d),
                          "weight_raw_mean": float(raw_weight.mean()),
                          "weight_normalized_mean": float(weight.mean()),
                          "weight_min": float(weight.min()), "weight_max": float(weight.max()),
                          "zero_weight_rows": int((weight == 0).sum()),
                          "fit_min_ts": str(d.ts.min()), "fit_max_ts": str(d.ts.max())})
            # Python Unicode I/O avoids native LightGBM save_model path issues.
            (out / f"generation_round{r}.txt").write_text(model.booster_.model_to_string(), encoding="utf-8")
            models[r] = model
            del d, x, y, weight, raw_weight, error
            gc.collect()
        pd.DataFrame(audit).to_csv(out / "training_weight_audit.csv", index=False)
        return models, rounds, params

    v.split_602020, v.load_weather, v.fit_models = checked_split, cached_weather, fit_models
    try:
        summary = v.run("baseline", ["--data-dir", str(root), "--seed", str(args.seed),
                                      "--batch-size", str(args.batch_size),
                                      "--max-iterations", str(args.max_iterations)])
    finally:
        v.split_602020, v.load_weather, v.fit_models = original_split, original_weather, original_fit
    summary.update({"experiment": f"weather_median_graph_{suffix}",
                    "weather_source": str(stage / "graph_forecast_{wa,wb}.parquet"),
                    "weather_manifest": str(meta_path), "weather_models_refitted": False,
                    "training_loss": "mean-normalized SMP*k*capacity weighted L1 on clipped CF" if args.loss == "smp_k" else "unweighted L1 on clipped CF",
                    "sample_weight_definition": "SMP * k * capacity / mean(SMP * k * capacity)" if args.loss == "smp_k" else "none",
                    "k_definition": "3 for right-closed timestamp hours 11..15 inclusive, otherwise 1",
                    "weight_audit": audit,
                    "loss_note": "Site-hour absolute loss surrogate; does not directly optimize absolute aggregate group errors."})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    manifest = {"source_weather_manifest_sha256": sha(meta_path), "cached_weather": inputs,
                "baseline_sha256": sha(root / "baseline.py"),
                "validation_sha256": sha(HERE / "baseline_validation.py"),
                "optimizer_sha256": sha(HERE / "baseline_optimizer.py"),
                "script_sha256": sha(Path(__file__)), "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "lightgbm", "pyarrow", "scikit-learn")}}
    (out / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    names = ["split602020_baseline", "split602020_weather_median_graph",
             "split602020_weather_median_graph_l1", "split602020_weather_median_graph_l1_smp_k"]
    for name in names:
        folder = root / "output/validation_602020" / name
        path = folder / "summary.json"
        if not path.is_file():
            continue
        other = json.loads(path.read_text(encoding="utf-8"))
        if other.get("seed") != args.seed:
            continue
        saved_split = folder / "split_days.csv"
        if not saved_split.is_file():
            continue
        pd.testing.assert_frame_equal(pd.read_csv(saved_split), expected_split)
        metric = other["model_metrics_holdout"]
        score = other["group_evaluation_independent_holdout"]
        initial, final, lower = (score[k] for k in ("W_initial_krw", "W_optimized_krw", "W_lower_bound_krw"))
        room = initial-lower
        rows.append({"Name": name, "Weighted MAE": metric["weighted_MAE_kwh"],
                     "MAE": metric["MAE_kwh"], "RMSE": metric["RMSE_kwh"],
                     "Initial Penalty": initial, "Optimized Penalty": final,
                     "Theoretical Lower Bound": lower, "Improvable Cost": final-lower,
                     "Improved Percentage": 100*(initial-final)/room if room > 0 else None,
                     "Group Reduction Percentage": score["W_reduction_pct"]})
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False, encoding="utf-8-sig")
    v.log(f"DONE: {out / 'comparison.csv'}")


if __name__ == "__main__":
    main()
