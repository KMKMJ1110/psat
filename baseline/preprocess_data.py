#!/usr/bin/env python3
"""OIBC preprocessing: items 3, 4, 5, and 6 only.

Place this script in <data-root>/codes/preprocess_data.py and run:
    python3 codes/preprocess_data.py

Original Parquet files are never changed. Outputs are written to:
    <data-root>/output/preprocessed/{train,test}/...
    <data-root>/output/preprocessed/preprocessing_summary.json

Rules:
3. Train/Test WA forecast humidity above 100 -> 100.
4. Test WA forecast missing entire (station, round, ts) rows -> copy opposite round.
5. Train WB forecast missing precipitation fields -> same station+ts from opposite round.
6. Train observed weather: mark specified outlying *values* as NaN; do not touch wb_vis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRECIP_COLS = [
    "wb_precip_prob",
    "wb_rain_prob",
    "wb_snow_prob",
    "wb_ice_prob",
    "wb_total_liq",
    "wb_rain",
    "wb_snow",
    "wb_ice",
]

# Item 6. Visibility (wb_vis) is deliberately excluded.
OBSERVED_OUTLIER_RULES = {
    "observed_wa": [("wa_wind_speed", ">", 50.0)],
    "observed_wb": [
        ("wb_cloud", ">", 100.0),
        ("wb_uv_idx", ">", 15.0),
        ("wb_wind_speed", ">", 200.0),
        ("wb_pressure", "<", 900.0),
    ],
}


def log(message: str) -> None:
    print(f"[preprocess] {message}", flush=True)


def load(root: Path, relative_path: str) -> pd.DataFrame:
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    return pd.read_parquet(path)


def save(out: Path, relative_path: str, frame: pd.DataFrame) -> None:
    path = out / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    log(f"Saved {relative_path}: {len(frame):,} rows")


def check_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    n = int(frame.duplicated(keys).sum())
    if n:
        raise ValueError(f"{label}: {n:,} duplicate keys in {keys}")


def clip_wa_humidity(frame: pd.DataFrame) -> int:
    n = int((frame["wa_humidity"] > 100).sum())
    frame["wa_humidity"] = frame["wa_humidity"].clip(upper=100)
    return n


def fill_test_wa_missing_rows(frame: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, dict]:
    keys = ["wa_station", "round", "ts"]
    check_unique(frame, keys, "test/forecast_wa (original)")
    rounds = set(frame["round"].dropna().unique())
    if rounds != {10, 17}:
        raise ValueError(f"Unexpected WA forecast rounds: {rounds}")

    # For each existing row, prepare a candidate with the opposite round.
    # Only candidates with an absent original key are copied.
    existing_keys = pd.MultiIndex.from_frame(frame[keys])
    swapped_keys = frame[keys].copy()
    swapped_keys["round"] = swapped_keys["round"].map({10: 17, 17: 10})
    to_insert = ~pd.MultiIndex.from_frame(swapped_keys).isin(existing_keys)

    new_rows = frame.loc[to_insert].copy()
    new_rows["round"] = new_rows["round"].map({10: 17, 17: 10})
    inserted_from_round10 = int((frame.loc[to_insert, "round"] == 10).sum())
    inserted_from_round17 = int((frame.loc[to_insert, "round"] == 17).sum())

    result = pd.concat([frame, new_rows], ignore_index=True)
    check_unique(result, keys, "test/forecast_wa (processed)")

    # Dataset-specific completeness check: 61 WA stations x 2 rounds x 8,760 hours.
    sites = pd.read_csv(root / "sites.csv", usecols=["wa_station"])
    smp = pd.read_csv(root / "smp.csv", usecols=["ts"])
    n_stations = int(sites["wa_station"].nunique())
    n_hours = int(smp["ts"].nunique())
    expected = n_stations * 2 * n_hours
    if len(result) != expected:
        raise ValueError(
            f"WA grid still incomplete: {len(result):,} rows versus "
            f"{expected:,} expected ({n_stations} stations x 2 rounds x {n_hours} hours). "
            "Some station/time combinations may be absent from both rounds."
        )

    result = result.sort_values(keys, kind="stable").reset_index(drop=True)
    info = {
        "missing_rows_recovered": int(len(new_rows)),
        "round10_missing_filled_from_round17": inserted_from_round17,
        "round17_missing_filled_from_round10": inserted_from_round10,
        "expected_complete_rows": expected,
        "remaining_missing_key_combinations": 0,
    }
    return result, info


def fill_train_wb_precip(frame: pd.DataFrame, nearest_hours: float = 3.0) -> dict:
    """Fill WB precipitation NaNs, avoiding failure when a Round-10 ROW is absent.

    Priority:
      1) Round 10 at the same station + target timestamp.
      2) (Optional) nearest valid Round 10 at the same station + target DATE,
         no more than nearest_hours away. This is a documented fallback,
         not an exact opposite-round substitute.
      3) Leave any unresolved values as NaN and report counts (never invent data).
    """
    keys = ["wb_station", "ts"]
    check_unique(frame, ["wb_station", "round", "ts"], "train/forecast_wb")
    target_mask = frame[PRECIP_COLS].isna().any(axis=1)
    n_rows = int(target_mask.sum())
    n_cells = int(frame.loc[target_mask, PRECIP_COLS].isna().sum().sum())
    if not n_rows:
        return {
            "affected_rows": 0,
            "affected_cells": 0,
            "exact_round10_rows_filled": 0,
            "nearest_round10_rows_filled": 0,
            "remaining_precip_nan_cells": 0,
        }

    affected_rounds = set(frame.loc[target_mask, "round"].unique())
    if affected_rounds != {17}:
        raise ValueError(f"Unexpected WB precipitation missing rounds: {affected_rounds}")

    reference = frame.loc[frame["round"] == 10, keys + PRECIP_COLS].set_index(keys)
    if not reference.index.is_unique:
        raise ValueError("Duplicate Round 10 station+ts keys in train/forecast_wb")

    wanted_index = pd.MultiIndex.from_frame(frame.loc[target_mask, keys])
    replacement = reference.reindex(wanted_index)[PRECIP_COLS].to_numpy(dtype=float)
    current = frame.loc[target_mask, PRECIP_COLS].to_numpy(dtype=float)
    exact_filled = np.isnan(current) & ~np.isnan(replacement)
    filled = np.where(np.isnan(current), replacement, current)
    exact_rows = int(exact_filled.any(axis=1).sum())
    missing_pair_rows = int(np.isnan(replacement).all(axis=1).sum())

    # Some Round-17 rows have no matching Round-10 *row*, even though all
    # values in the existing Round-10 rows are nonmissing (92 rows in the
    # reported dataset: 736 missing values across 8 columns).
    near_rows = 0
    near_cells = 0
    if nearest_hours > 0 and np.isnan(filled).any():
        positions = np.flatnonzero(np.isnan(filled).any(axis=1))
        targets = frame.loc[target_mask, keys].iloc[positions].copy()
        targets["local_date"] = targets["ts"].dt.date
        targets["__position"] = positions

        # Limit the candidate search to the affected dates/stations. Every
        # candidate remains a Round-10 forecast available before target day.
        in_target_dates = frame["ts"].dt.date.isin(targets["local_date"].unique())
        in_target_stations = frame["wb_station"].isin(targets["wb_station"].unique())
        src = frame.loc[
            frame["round"].eq(10) & in_target_dates & in_target_stations,
            keys + PRECIP_COLS,
        ].copy()
        src["local_date"] = src["ts"].dt.date
        src["matched_round10_ts"] = src["ts"]

        if not src.empty:
            nearest = pd.merge_asof(
                targets.sort_values("ts"),
                src.sort_values("ts"),
                on="ts",
                by=["wb_station", "local_date"],
                direction="nearest",
                tolerance=pd.Timedelta(hours=nearest_hours),
            ).sort_values("__position")
            near_values = nearest[PRECIP_COLS].to_numpy(dtype=float)
            original_part = filled[positions]
            usable = np.isnan(original_part) & ~np.isnan(near_values)
            near_cells = int(usable.sum())
            near_rows = int(usable.any(axis=1).sum())
                        # 인접 Round 10 예보로 대체한 행들의 실제 시간 차이
            used_rows = usable.any(axis=1)

            time_diff = (
                nearest.loc[used_rows, "matched_round10_ts"]
                - nearest.loc[used_rows, "ts"]
            )

            hour_counts = (
                (time_diff.dt.total_seconds() / 3600)
                .value_counts()
                .sort_index()
            )

            log("WB precipitation: nearby Round 10 time offsets")
            for hours, count in hour_counts.items():
                log(f"  {hours:+g} hours: {count:,} rows")
            filled[positions] = np.where(usable, near_values, original_part)

    remaining = int(np.isnan(filled).sum())
    frame.loc[target_mask, PRECIP_COLS] = filled
    result = {
        "affected_rows": n_rows,
        "affected_cells": n_cells,
        "exact_round10_rows_filled": exact_rows,
        "exact_round10_cells_filled": int(exact_filled.sum()),
        "rows_without_exact_round10_reference": missing_pair_rows,
        "nearest_round10_tolerance_hours": nearest_hours,
        "nearest_round10_rows_filled": near_rows,
        "nearest_round10_cells_filled": near_cells,
        "remaining_precip_nan_cells": remaining,
        "remaining_precip_nan_rows": int(frame[PRECIP_COLS].isna().any(axis=1).sum()),
    }
    log(
        "WB precipitation: "
        f"{exact_rows:,} rows filled by exact Round-10 match; "
        f"{near_rows:,} by nearby Round-10 forecast; "
        f"{remaining:,} unresolved NaN cells"
    )
    if remaining:
        log("WARNING: Some WB precipitation cells remain NaN; no replacement was invented.")
    return result


def mask_observed_outliers(frame: pd.DataFrame, source: str) -> dict:
    counts = {}
    for column, operator, threshold in OBSERVED_OUTLIER_RULES[source]:
        mask = frame[column] > threshold if operator == ">" else frame[column] < threshold
        counts[column] = int(mask.sum())
        frame.loc[mask, column] = np.nan
    return counts


def run(root: Path, out: Path, nearest_hours: float = 3.0) -> dict:
    if not (root / "sites.csv").exists() or not (root / "smp.csv").exists():
        raise FileNotFoundError("Data root must contain sites.csv and smp.csv")
    # Never allow writing processed data inside the raw train/test folders.
    resolved_out = out.resolve()
    for raw in (root / "train", root / "test"):
        if resolved_out == raw.resolve() or raw.resolve() in resolved_out.parents:
            raise ValueError("Output directory cannot be inside raw train/ or test/")

    summary = {"source_data_root": str(root), "output_directory": str(out), "operations": {}}

    # 3: Train and test WA relative humidity clipped to 100.
    train_wa = load(root, "train/forecast_wa.parquet")
    count = clip_wa_humidity(train_wa)
    save(out, "train/forecast_wa.parquet", train_wa)
    summary["operations"]["3_train_wa_humidity_clipped"] = count
    del train_wa

    test_wa = load(root, "test/forecast_wa.parquet")
    count = clip_wa_humidity(test_wa)
    test_wa, repair_info = fill_test_wa_missing_rows(test_wa, root)
    save(out, "test/forecast_wa.parquet", test_wa)
    summary["operations"]["3_test_wa_humidity_clipped"] = count
    summary["operations"]["4_test_wa_missing_rows"] = repair_info
    del test_wa

    # 5: Train WB precipitation columns are NaN only in Round 17.
    train_wb = load(root, "train/forecast_wb.parquet")
    wb_info = fill_train_wb_precip(train_wb, nearest_hours=nearest_hours)
    save(out, "train/forecast_wb.parquet", train_wb)
    summary["operations"]["5_train_wb_precip"] = wb_info
    del train_wb

    # Test WB has no relevant missingness: copy to the processed-data directory.
    test_wb = load(root, "test/forecast_wb.parquet")
    save(out, "test/forecast_wb.parquet", test_wb)
    summary["operations"]["test_wb"] = "unchanged copy"
    del test_wb

    # 6: Replace only targeted OBSERVED variables with NaN (not whole rows).
    for source in ("observed_wa", "observed_wb"):
        obs = load(root, f"train/{source}.parquet")
        counts = mask_observed_outliers(obs, source)
        save(out, f"train/{source}.parquet", obs)
        summary["operations"][f"6_{source}_outliers_to_nan"] = counts
        del obs

    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "preprocessing_summary.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"Saved {report_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="OIBC preprocessing items 3-6")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent,
        help="OIBC data directory containing sites.csv, train/, test/, and smp.csv",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output folder; default: <root>/output/preprocessed",
    )
    parser.add_argument(
        "--wb-nearest-hours", type=float, default=3.0,
        help="Fallback only: nearest same-station, same-day Round-10 forecast within N hours; 0 disables fallback",
    )
    args = parser.parse_args()
    if args.wb_nearest_hours < 0:
        parser.error("--wb-nearest-hours must be nonnegative")
    root = args.root.expanduser().resolve()
    out = args.out.expanduser().resolve() if args.out else root / "output" / "preprocessed"
    summary = run(root, out, nearest_hours=args.wb_nearest_hours)
    log("Completed. Original source files were not modified.")
    for key, value in summary["operations"].items():
        log(f"  {key}: {value}")


if __name__ == "__main__":
    main()
