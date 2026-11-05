#!/usr/bin/env python3
"""OIBC 2026: model-agnostic best-improvement Greedy group optimizer.

Requires numpy, pandas, pyarrow (for parquet). Does not train a forecast model.

Input (OUT-OF-SAMPLE validation predictions): site_id, ts, pred_kwh
Optional test predictions (for submission): site_id, ts, pred_kwh

- Initial assignment is sites.csv/current_group (A->G1, B->G2).
- Minimize 0.10 * sum_t SMP_t*k_t*(|e_G1,t|+|e_G2,t|).
- Exhaustively test all feasible single-site Moves and cross-group Swaps
  each iteration; apply the best strictly improving change.
- Each group must have at least 20,000 kW of installed capacity.
- Only *complete observed hours* (all sites have generation labels AND
  predictions) are optimized/scored; missing actuals are NOT fabricated.
- Optional time-ordered holdout is scored but NEVER used to optimize groups.
- Initializer is isolated for future model-informed initialization.

Place in data/codes/baseline_optimizer.py. The default data root is its parent.
Example:
  python codes/baseline_optimizer.py \
    --validation-preds runs/lgbm_default/validation_site_predictions.parquet \
    --test-preds runs/lgbm_default/test_site_predictions.parquet \
    --run-name lgbm_greedy \
    --opt-start '2025-01-01 01:00' \
    --holdout-start '2025-04-01 01:00'

If --holdout-start is omitted, the reported optimization W is IN-SAMPLE
with respect to group assignment; it is NOT an independent performance estimate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

MIN_GROUP_CAPACITY = 20_000.0
PENALTY_RATE = 0.10
TIMEZONE = "Asia/Seoul"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "codes" else SCRIPT_DIR


def log(message: str) -> None:
    print(message, flush=True)


def path_in(root: Path, name: str | Path) -> Path:
    p = Path(name).expanduser()
    return p if p.is_absolute() else root / p


def to_kst(values) -> pd.Series:
    s = pd.to_datetime(values)
    if s.dt.tz is None:
        return s.dt.tz_localize(TIMEZONE)
    return s.dt.tz_convert(TIMEZONE)


def parse_time(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize(TIMEZONE) if ts.tzinfo is None else ts.tz_convert(TIMEZONE)


def load_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if columns and not set(columns).issubset(df.columns):
            raise ValueError(f"{path}: missing {sorted(set(columns) - set(df.columns))}")
        return df[columns] if columns else df
    raise ValueError(f"Use .csv or .parquet: {path}")


def load_sites(root: Path) -> pd.DataFrame:
    sites = pd.read_csv(root / "sites.csv")[[
        "site_id", "current_group", "capacity_kw"
    ]].copy()
    if sites.site_id.isna().any() or sites.site_id.duplicated().any():
        raise ValueError("sites.csv has missing/duplicate site_id")
    if not sites.current_group.isin(("A", "B")).all():
        raise ValueError("current_group must be A/B")
    if not np.isfinite(sites.capacity_kw).all() or (sites.capacity_kw <= 0).any():
        raise ValueError("Invalid capacity_kw")
    if len(sites) != 253:
        log(f"WARNING: sites.csv has {len(sites)} sites; contest normally expects 253")
    return sites


def load_predictions(path: Path, sites: pd.DataFrame) -> pd.DataFrame:
    cols = ["site_id", "ts", "pred_kwh"]
    p = load_table(path, cols)
    p["ts"] = to_kst(p["ts"])
    p["pred_kwh"] = pd.to_numeric(p["pred_kwh"], errors="coerce")
    if p[["site_id", "ts"]].isna().any().any() or p.duplicated(["site_id", "ts"]).any():
        raise ValueError(f"Missing/duplicate site_id,ts in {path}")
    if (~p.site_id.isin(sites.site_id)).any():
        raise ValueError(f"Unknown site_id in {path}")
    if not np.isfinite(p.pred_kwh).all() or (p.pred_kwh < 0).any():
        raise ValueError(f"pred_kwh must be finite and nonnegative: {path}")
    return p


def load_observed_residuals(root: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    # Load only time bounds covered by provided predictions; no test labels.
    actual = pd.read_parquet(
        root / "train" / "generation.parquet",
        columns=["site_id", "ts", "energy"],
    )
    actual["ts"] = to_kst(actual.ts)
    actual = actual.loc[actual.ts.isin(predictions.ts.unique())]
    if actual.duplicated(["site_id", "ts"]).any():
        raise ValueError("Duplicate actual site_id,ts in generation.parquet")
    actual = actual.merge(
        predictions, on=["site_id", "ts"], how="left", validate="one_to_one"
    )
    if actual.pred_kwh.isna().any():
        raise ValueError(
            f"Missing OOS predictions for {actual.pred_kwh.isna().sum():,} observed labels. "
            "Provide predictions for all available site×time pairs in the validation window."
        )
    if not np.isfinite(actual.energy).all():
        raise ValueError("Actual generation contains nonfinite values")
    actual["residual_kwh"] = actual.pred_kwh - actual.energy
    return actual[["site_id", "ts", "residual_kwh"]]


def complete_matrix(
    residuals: pd.DataFrame, site_ids: list[str],
    start: pd.Timestamp | None, end: pd.Timestamp | None,
    *, label: str,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    part = residuals
    if start is not None:
        part = part.loc[part.ts >= start]
    if end is not None:
        part = part.loc[part.ts < end]
    if part.empty:
        raise ValueError(f"No residual observations in {label} period")
    all_hours = part.ts.nunique()
    piv = part.pivot(index="ts", columns="site_id", values="residual_kwh")
    piv = piv.reindex(columns=site_ids).sort_index()
        # Missing residuals are treated as zero.
    missing_count = int(piv.isna().sum().sum())

    piv = piv.fillna(0.0)

    times = pd.DatetimeIndex(piv.index)

    log(
        f"{label}: {len(times):,}/{all_hours:,} hours used; "
        f"missing residuals filled with 0: {missing_count:,}; "
        f"range {times.min()} ~ {times.max()}"
    )

    return piv.to_numpy(dtype=np.float64), times


def time_weights(root: Path, times: pd.DatetimeIndex) -> np.ndarray:
    smp = pd.read_csv(root / "train" / "smp.csv", usecols=["ts", "smp"])
    smp["ts"] = to_kst(smp.ts)
    if smp.ts.duplicated().any():
        raise ValueError("Duplicate timestamps in train/smp.csv")
    prices = pd.to_numeric(smp.set_index("ts").smp.reindex(times), errors="coerce")
    if not np.isfinite(prices).all() or (prices < 0).any():
        raise ValueError("Missing/negative/nonfinite SMP for validation times")
    k = np.where((times.hour >= 11) & (times.hour <= 15), 3.0, 1.0)
    return prices.to_numpy(np.float64) * k


def initial_mask(sites: pd.DataFrame, *, method: str = "current_group") -> np.ndarray:
    """Future extension point: method='model_informed' from OOS residuals."""
    if method != "current_group":
        raise ValueError(f"Initializer not implemented: {method}")
    return sites.current_group.eq("A").to_numpy(dtype=bool)


def check_capacity(mask: np.ndarray, capacities: np.ndarray) -> tuple[float, float]:
    c1 = float(capacities[mask].sum())
    c2 = float(capacities.sum() - c1)
    if min(c1, c2) + 1e-8 < MIN_GROUP_CAPACITY:
        raise ValueError(f"Capacity violation: G1={c1:.1f}kW, G2={c2:.1f}kW")
    return c1, c2


def penalty(matrix: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    g1 = matrix[:, mask].sum(axis=1)
    total = matrix.sum(axis=1)
    return float(PENALTY_RATE * np.dot(weights, np.abs(g1) + np.abs(total - g1)))


def evaluate_batch(g1: np.ndarray, total: np.ndarray, weights: np.ndarray,
                   changes: np.ndarray) -> np.ndarray:
    """Candidate W for each column of changes (T x candidates).

    Uses |G1|+|G2|=max(|total|, |2*G1-total|) for two groups.
    """
    candidate_g1 = g1[:, None] + changes
    term = np.maximum(np.abs(total[:, None]), np.abs(2*candidate_g1 - total[:, None]))
    return PENALTY_RATE * np.sum(weights[:, None] * term, axis=0)


def greedy_best_improvement(
    matrix: np.ndarray, weights: np.ndarray, sites: pd.DataFrame,
    mask: np.ndarray, *, batch_size: int = 128, max_iterations: int = 0,
) -> tuple[np.ndarray, dict, pd.DataFrame]:
    """All feasible Moves + all cross-group Swaps at every step.

    max_iterations=0 means stop only when there is no strictly improving move.
    """
    if batch_size < 1 or max_iterations < 0:
        raise ValueError("batch_size must be >=1 and max_iterations >=0")
    capacities = sites.capacity_kw.to_numpy(np.float64)
    mask = mask.copy()
    cap1, cap2 = check_capacity(mask, capacities)
    total_cap = cap1 + cap2
    total = matrix.sum(axis=1)
    g1 = matrix[:, mask].sum(axis=1)
    current = float(PENALTY_RATE * np.dot(weights, np.abs(g1)+np.abs(total-g1)))
    initial_w = current
    history = []
    started = time.monotonic()
    iteration = 0
    reached_local_minimum = False

    while True:
        if max_iterations and iteration >= max_iterations:
            break
        best_cost = current
        best_action = None
        # Avoid accepting numerical round-off differences as real improvements.
        tolerance = max(1e-6, abs(current) * 1e-12)

        # 1. Every feasible single-site move, in batches.
        move_ids = np.arange(len(mask))
        new_caps = cap1 + np.where(mask, -capacities, capacities)
        move_ids = move_ids[(new_caps >= MIN_GROUP_CAPACITY - 1e-8) &
                            (new_caps <= total_cap - MIN_GROUP_CAPACITY + 1e-8)]
        for offset in range(0, len(move_ids), batch_size):
            ids = move_ids[offset:offset + batch_size]
            sign = np.where(mask[ids], -1., 1.)
            costs = evaluate_batch(g1, total, weights, matrix[:, ids] * sign)
            j = int(np.argmin(costs))
            if costs[j] < best_cost - tolerance:
                best_cost = float(costs[j]); best_action = ("move", int(ids[j]), None)

        # 2. Every feasible cross-group swap, in batches.
        a, b = np.flatnonzero(mask), np.flatnonzero(~mask)
        ii = np.repeat(a, len(b))
        jj = np.tile(b, len(a))
        new_caps = cap1 - capacities[ii] + capacities[jj]
        ok = (new_caps >= MIN_GROUP_CAPACITY - 1e-8) & (
            new_caps <= total_cap - MIN_GROUP_CAPACITY + 1e-8)
        ii, jj = ii[ok], jj[ok]
        for offset in range(0, len(ii), batch_size):
            ia = ii[offset:offset + batch_size]
            ib = jj[offset:offset + batch_size]
            costs = evaluate_batch(g1, total, weights, matrix[:, ib] - matrix[:, ia])
            j = int(np.argmin(costs))
            if costs[j] < best_cost - tolerance:
                best_cost = float(costs[j]); best_action = ("swap", int(ia[j]), int(ib[j]))

        if best_action is None:
            reached_local_minimum = True
            break

        action, i, j = best_action
        old = current
        if action == "move":
            change = (-matrix[:, i] if mask[i] else matrix[:, i])
            cap1 += (-capacities[i] if mask[i] else capacities[i])
            mask[i] = ~mask[i]
        else:
            change = -matrix[:, i] + matrix[:, j]
            cap1 += -capacities[i] + capacities[j]
            mask[i] = False; mask[j] = True
        g1 = g1 + change
        current = float(PENALTY_RATE * np.dot(weights, np.abs(g1) + np.abs(total-g1)))
        if current >= old - tolerance:
            raise ArithmeticError("Greedy move failed to strictly improve W")
        iteration += 1
        history.append({
            "iteration": iteration, "action": action,
            "site_a": str(sites.site_id.iloc[i]),
            "site_b": (str(sites.site_id.iloc[j]) if j is not None else ""),
            "W_before_krw": old, "W_after_krw": current,
            "improvement_krw": old - current,
            "capacity_G1_kw": cap1, "capacity_G2_kw": total_cap-cap1,
        })
        log(f"Greedy step {iteration:3d}: {action:4s} "
            f"{sites.site_id.iloc[i]}" +
            (f" <-> {sites.site_id.iloc[j]}" if j is not None else "") +
            f", W={current:,.0f} KRW (saved {old-current:,.0f})")

    c1, c2 = check_capacity(mask, capacities)
    seconds = time.monotonic() - started
    meta = {
        "algorithm": "Best-improvement Greedy (all feasible moves and swaps)",
        "initializer": "current_group", "iterations": iteration,
        "local_minimum_reached": reached_local_minimum,
        "max_iterations": max_iterations,
        "W_initial_krw": initial_w, "W_optimized_krw": current,
        "W_reduction_krw": initial_w-current,
        "W_reduction_pct": 100*(initial_w-current)/initial_w if initial_w else 0.,
        "capacity_G1_kw": c1, "capacity_G2_kw": c2,
        "count_G1": int(mask.sum()), "count_G2": int((~mask).sum()),
        "elapsed_seconds": seconds,
    }
    log(f"Greedy finished: {iteration} improvements, local minimum={reached_local_minimum}, "
        f"W={current:,.0f} KRW, elapsed={seconds:.1f}s")
    return mask, meta, pd.DataFrame(history, columns=[
        "iteration", "action", "site_a", "site_b", "W_before_krw", "W_after_krw",
        "improvement_krw", "capacity_G1_kw", "capacity_G2_kw",
    ])


def generate_test_forecast(root: Path, site_pred_path: Path,
                           assignment: pd.DataFrame, output_dir: Path) -> int:
    test_times = pd.DatetimeIndex(to_kst(pd.read_csv(root / "smp.csv", usecols=["ts"]).ts))
    if test_times.duplicated().any():
        raise ValueError("Duplicate test timestamps in smp.csv")
    sites = assignment[["site_id"]]
    predictions = load_predictions(site_pred_path, sites)
    target = predictions.loc[predictions.ts.isin(test_times)].copy()
    if len(target) != len(sites) * len(test_times):
        raise ValueError(
            f"Test predictions incomplete: {len(target):,}/"
            f"{len(sites)*len(test_times):,} site-hours."
        )
    if target.groupby("ts").site_id.nunique().ne(len(sites)).any():
        raise ValueError("Not all sites have a prediction at each test hour")
    output = target.merge(assignment, on="site_id", how="left", validate="many_to_one")
    output = (output.groupby(["group", "ts"], as_index=False).pred_kwh.sum()
              .rename(columns={"pred_kwh": "forecast_kwh"})
              .sort_values(["group", "ts"]))
    if len(output) != 2 * len(test_times):
        raise ValueError("Output must contain exactly G1/G2 × test hours")
    output.to_csv(output_dir / "forecast.csv", index=False)
    return len(output)


def main() -> None:
    ap = argparse.ArgumentParser(description="Model-agnostic OIBC Greedy group optimizer")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT,
                    help="Directory containing sites.csv, train/, smp.csv")
    ap.add_argument("--validation-preds", required=True,
                    help="Out-of-sample site_id,ts,pred_kwh parquet/csv")
    ap.add_argument("--test-preds", default=None,
                    help="Optional test site predictions for forecast.csv")
    ap.add_argument("--run-name", default="greedy_baseline")
    ap.add_argument("--opt-start", default=None, help="Optimization start (inclusive), KST")
    ap.add_argument("--opt-end", default=None, help="Optimization end (exclusive), KST")
    ap.add_argument("--holdout-start", default=None,
                    help="Independent holdout start (inclusive), KST")
    ap.add_argument("--holdout-end", default=None,
                    help="Independent holdout end (exclusive), KST")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-iterations", type=int, default=0,
                    help="0: until local minimum (default); >0: stop at limit")
    args = ap.parse_args()

    if args.run_name in ("", ".", "..") or "/" in args.run_name or "\\" in args.run_name:
        ap.error("--run-name must be a simple directory name")
    if args.holdout_end and not args.holdout_start:
        ap.error("--holdout-end requires --holdout-start")
    root = args.data_dir.expanduser().resolve()
    out = root / "output" / "baseline_optimizer" / args.run_name
    sites = load_sites(root)
    val_path = path_in(root, args.validation_preds)
    pred = load_predictions(val_path, sites)
    if pred.empty:
        raise ValueError("Empty validation predictions")
    test_start = to_kst(pd.read_csv(root / "smp.csv", usecols=["ts"]).ts).min()
    if (pred.ts >= test_start).any():
        raise ValueError("Validation predictions include test timestamps; forbidden")
    # The model producing these predictions MUST have been trained on earlier data.
    res = load_observed_residuals(root, pred)
    start = parse_time(args.opt_start)
    end = parse_time(args.opt_end)
    hold_start = parse_time(args.holdout_start)
    hold_end = parse_time(args.holdout_end)
    if hold_start is not None:
        if end is None:
            end = hold_start
        if end > hold_start:
            raise ValueError("Optimization period overlaps holdout period")
        if start is not None and start >= end:
            raise ValueError("Invalid optimization time bounds")
        if hold_end is not None and hold_start >= hold_end:
            raise ValueError("Invalid holdout time bounds")
    elif end is not None and start is not None and start >= end:
        raise ValueError("Invalid optimization time bounds")

    matrix, times = complete_matrix(res, sites.site_id.to_list(), start, end, label="OPT")
    weights = time_weights(root, times)
    lower_bound = float(PENALTY_RATE * np.dot(weights, np.abs(matrix.sum(axis=1))))
    log(f"Theoretical Lower Bound = {lower_bound:,.2f} KRW")
    initial = initial_mask(sites, method="current_group")
    log("Initial group assignment: current_group (A->G1, B->G2)")
    init_w = penalty(matrix, weights, initial)
    log(f"OPT initial W={init_w:,.0f} KRW; {matrix.shape[0]:,} complete hours")
    mask, metrics, trace = greedy_best_improvement(
        matrix, weights, sites, initial,
        batch_size=args.batch_size, max_iterations=args.max_iterations,
    )
    summary = {
        "model_prediction_file": str(val_path),
        "optimization_window_start": str(start if start is not None else times.min()),
        "optimization_window_end_exclusive": str(end) if end is not None else None,
        "optimization_complete_hours": len(times),
        "objective": "0.10 * sum(SMP_t * k_t * (abs(group1_error_t)+abs(group2_error_t)))",
        "error_definition": "pred_kwh - actual_kwh",
        "missing_labels_policy": "exclude incomplete site-hours (no imputation)",
        "W_lower_bound_krw": lower_bound,
        "optimization_score_is_in_sample_for_assignment": True,
        **metrics,
    }
    if hold_start is not None:
        hold_mat, hold_times = complete_matrix(
            res, sites.site_id.to_list(), hold_start, hold_end, label="HOLDOUT")
        hold_weights = time_weights(root, hold_times)
        w_ref = penalty(hold_mat, hold_weights, initial)
        w_opt = penalty(hold_mat, hold_weights, mask)
        summary["holdout"] = {
            "start": str(hold_start), "end_exclusive": str(hold_end) if hold_end else None,
            "complete_hours": len(hold_times),
            "W_current_group_krw": w_ref,
            "W_optimized_group_krw": w_opt,
            "W_reduction_krw": w_ref-w_opt,
            "W_reduction_pct": 100*(w_ref-w_opt)/w_ref if w_ref else 0.,
        }
        log(f"HOLDOUT: current={w_ref:,.0f}, optimized={w_opt:,.0f} KRW, "
            f"change={w_ref-w_opt:,.0f} KRW")
    else:
        log("WARNING: no holdout; OPT W reduction is an in-sample grouping result only")

    out.mkdir(parents=True, exist_ok=True)
    assignment = pd.DataFrame({
        "site_id": sites.site_id,
        "group": np.where(mask, "G1", "G2"),
    })
    assignment.to_csv(out / "composition.csv", index=False)
    trace.to_csv(out / "greedy_trace.csv", index=False)
    if args.test_preds:
        test_path = path_in(root, args.test_preds)
        rows = generate_test_forecast(root, test_path, assignment, out)
        summary["test_prediction_file"] = str(test_path)
        summary["forecast_rows"] = rows
        log(f"Saved {rows:,} rows to forecast.csv")
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    log(f"Output directory: {out}")


if __name__ == "__main__":
    main()
