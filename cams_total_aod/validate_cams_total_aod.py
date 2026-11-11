"""Compare the team's GFS baseline with CAMS Total AOD additions."""

import argparse
from contextlib import chdir
from datetime import datetime
import json
from pathlib import Path
import sys
from tempfile import mkdtemp
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from build_cams_total_aod import AOD_COLS, AOD_FINAL, AOD_NEXT, AOD_PREV, EXCLUDED, FINAL_NAME, KST, PARAM_ID, VARIABLE


def load_inputs(team, root):
    external = root / "output" / "external"
    gfs = team.checked_hourly(pd.read_parquet(team.GFS_PATH, columns=["ts", *team.GFS_15]), "GFS")
    path = external / FINAL_NAME
    provenance = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    if provenance["variable"] != VARIABLE or provenance["param_id"] != PARAM_ID:
        raise ValueError("The provenance must describe CAMS Total AOD 550 nm.")
    cams = team.checked_hourly(pd.read_parquet(path, columns=["ts", *AOD_FINAL, EXCLUDED]), "CAMS Total AOD")
    target_day = (cams["ts"] - pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%d")
    omitted = {entry["target_day_kst"] for entry in provenance["excluded_cycles"]}
    excluded = cams[EXCLUDED]
    if not pd.api.types.is_bool_dtype(excluded) or excluded.isna().any() or not excluded.equals(target_day.isin(omitted)):
        raise ValueError("The CAMS mask does not match the excluded dates in its provenance.")
    if excluded.sum() != 24 * len(omitted) or excluded.sum() != provenance["excluded_cams_hours"]:
        raise ValueError("The number of excluded CAMS hours does not match its provenance.")
    if cams.loc[excluded, AOD_FINAL].notna().any().any():
        raise ValueError("Excluded hours must not contain Total AOD values.")
    active = cams.loc[~excluded]
    missing = active[AOD_FINAL].isna()
    missing.loc[active["ts"].dt.hour.eq(1), AOD_PREV] = False
    missing.loc[active["ts"].dt.hour.eq(0), AOD_NEXT] = False
    if missing.any().any():
        raise ValueError("Total AOD is missing outside the excluded days or expected day edges.")
    return gfs, cams


def compare(team, root, gfs, cams, weights):
    val = team.val
    base = val.load_module(root / "baseline.py", "oibc_baseline_total_aod")
    sites = pd.read_csv(root / "sites.csv")[[
        "site_id", "current_group", "capacity_kw", "lat", "lon", "wa_station", "wb_station",
    ]]
    gen = pd.read_parquet(root / "train" / "generation.parquet", columns=["site_id", "ts", "energy"])
    gen["ts"] = val.to_kst(gen["ts"])
    recent = gen[gen["ts"] >= base.TRAIN_FROM].copy()
    model_fit, group_actual, hold_actual, _ = val.split_602020(recent, val.SEED)
    model_fit = pd.concat([gen[gen["ts"] < base.TRAIN_FROM], model_fit], ignore_index=True)
    eval_actual = pd.concat([group_actual, hold_actual], ignore_index=True)
    del gen, recent

    required_ts = pd.Index(model_fit["ts"].unique()).union(pd.Index(eval_actual["ts"].unique()))
    absent = required_ts.difference(pd.Index(cams["ts"]))
    if len(absent):
        raise ValueError(f"Total AOD is missing {len(absent):,} required hours. First missing hour: {absent[0]}")
    hourly = gfs.merge(cams[["ts", *AOD_FINAL]], on="ts", how="outer", validate="one_to_one")
    keys = eval_actual[team.KEYS].copy()
    excluded = keys["ts"].isin(cams.loc[cams[EXCLUDED], "ts"])
    ensemble_inputs = team.aligned_ensemble_inputs(keys, weights)

    output = root / "output" / "feat_gfs_cams_total_aod"
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo(KST)).strftime("%Y%m%d_%H%M%S")
    run = Path(mkdtemp(prefix=f"run_{stamp}_", dir=output))
    print(f"Total AOD run: {run}", flush=True)
    print(f"Tree/MLP/TabPFN weights: {weights}; GFS fallback hours: {keys.loc[excluded, 'ts'].nunique():,}", flush=True)

    # CatBoost can write training logs to the process working directory.
    with chdir(run):
        team.fit_tree(base, sites, model_fit, eval_actual, gfs, run / "gfs")
        reference = team.aligned_tree(run / "gfs", keys)
        fixed = np.zeros(len(keys), dtype=float)
        for weight, values in ensemble_inputs:
            fixed += weight * values.fillna(reference).mask(reference.le(0), 0).to_numpy()

        predictions = {}
        rows = []
        for name, cols in (("gfs", []), ("total_aod2", AOD_COLS), ("total_aod6", AOD_FINAL)):
            tree = reference
            if cols:
                team.fit_tree(base, sites, model_fit, eval_actual, hourly[[*gfs.columns, *cols]], run / name)
                tree = team.aligned_tree(run / name, keys).where(~excluded, reference)
            pred = keys.assign(pred_kwh=weights[0] * tree + fixed)
            predictions[name] = pred
            rows.append({"variant": name, **team.score(pred, sites, group_actual)})

    result = pd.DataFrame(rows)
    winner = result.loc[result["group_opt_optimized"].idxmin(), "variant"]
    holdout = pd.DataFrame([
        {"variant": name, **team.score(predictions[name], sites, group_actual, hold_actual)}
        for name in dict.fromkeys(["gfs", winner])
    ])
    result.to_csv(run / "result.csv", index=False, mode="x")
    holdout.to_csv(run / "holdout.csv", index=False, mode="x")
    print(result.to_string(index=False))
    print(f"Selected on group_opt: {winner}")
    print(holdout.to_string(index=False))
    print(f"Saved {run}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Team data directory")
    parser.add_argument("--w", nargs=3, type=float, metavar=("TREE", "MLP", "TABPFN"), default=[0.5, 0.0, 0.5])
    args = parser.parse_args()
    root = args.root.resolve()
    if not np.isfinite(args.w).all() or min(args.w) < 0 or not np.isclose(sum(args.w), 1.0, rtol=0, atol=1e-9):
        parser.error("--w values must be nonnegative and sum to 1.")
    team_script = root / "codes" / "gfs_tree.py"
    if not team_script.is_file():
        parser.error(f"Team validation script missing: {team_script}")
    sys.path.insert(0, str(team_script.parent))
    import gfs_tree as team

    gfs, cams = load_inputs(team, root)
    compare(team, root, gfs, cams, args.w)


if __name__ == "__main__":
    main()
