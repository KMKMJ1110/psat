#!/usr/bin/env python3
"""OIBC 60/20/20: LightGBM MAE Loss + 5-minute MILP group assignment.

Place next to the user's 60/20/20 ``codes/baseline_validation.py``.
Requires completed 60/20/20 MAE Loss experiment with saved predictions.

    python3 codes/baseline_validation_mae_loss_milp.py
    python3 codes/baseline_validation_mae_loss_milp.py --time-limit 300

Uses scipy.optimize.milp (HiGHS) with a hard solver time limit.  It does NOT
retrain LightGBM.  Uses the SAME 60:20:20 split and the SAME site-hour residual
policy as 60/20/20 baseline.  Never uses HOLDOUT residuals to choose groups.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, eye, hstack, vstack

HERE = Path(__file__).resolve().parent
ROOT_DEFAULT = HERE.parent if HERE.name == "codes" else HERE


def load_core():
    path = HERE / "baseline_validation.py"
    sp = importlib.util.spec_from_file_location("psat_602020_core_milp", path)
    if sp is None or sp.loader is None:
        raise RuntimeError(f"Cannot load core: {path}")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    needed = ("split_602020", "residual_matrix", "score_group", "load_module", "model_metrics")
    if not all(hasattr(mod, n) for n in needed):
        raise RuntimeError(f"{path} must be the 60:20:20 baseline, not old 5-Fold script")
    return mod


def solve_milp(matrix: np.ndarray, weights: np.ndarray, capacities: np.ndarray,
               seconds: float = 300.0, mip_rel_gap: float = 1e-4):
    """Exact MILP formulation, with one auxiliary per nonzero-weight hour.

    Let e_t = residuals, E_t = sum_i e_it, g_t = sum_i e_it*x_i.
    |g_t|+|E_t-g_t| = max(|E_t|, |2g_t-E_t|).

    Min 0.1 sum_t w_t*z_t,
    s.t. z_t >= |E_t|, 2*g_t-z_t <= E_t, -2*g_t-z_t <= -E_t,
    and 20000 <= sum_i cap_i*x_i <= total_cap-20000, x binary.

    z lower bounds contain the |E_t| inequality, so 2 linear rows per hour.
    Scaling the objective by 1e6 leaves the optimizer unchanged.
    """
    if matrix.ndim != 2 or len(capacities) != matrix.shape[1] or len(weights) != matrix.shape[0]:
        raise ValueError("Shape mismatch")
    if not np.isfinite(matrix).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid residuals / weights")
    n_sites = matrix.shape[1]
    total_cap = float(np.sum(capacities))
    min_cap = 20_000.0
    if total_cap < 2 * min_cap:
        raise ValueError("Both capacity constraints cannot be satisfied")

    # Night and zero-weight hours add no information or objective contribution.
    active = (weights > 0) & np.any(matrix != 0.0, axis=1)
    a = np.asarray(matrix[active, :], dtype=np.float64)
    w = np.asarray(weights[active], dtype=np.float64)
    totals = a.sum(axis=1)
    n_hours = a.shape[0]
    if n_hours == 0:
        raise ValueError("No active weighted residual hours")

    # 253 binary variables followed by one continuous nonnegative z per hour.
    c = np.concatenate([np.zeros(n_sites), 0.1 * w / 1_000_000.0])
    integrality = np.concatenate([np.ones(n_sites), np.zeros(n_hours)]).astype(np.uint8)
    lower = np.concatenate([np.zeros(n_sites), np.abs(totals)])
    upper = np.concatenate([np.ones(n_sites), np.full(n_hours, np.inf)])
    bounds = Bounds(lower, upper)

    a_sparse = csr_matrix(2.0 * a)
    minus_identity = -eye(n_hours, dtype=np.float64, format="csr")
    # For each t: 2 A x - z <= total, -2 A x - z <= -total.
    lhs_pos = hstack([a_sparse, minus_identity], format="csr")
    lhs_neg = hstack([-a_sparse, minus_identity], format="csr")
    cap_row = hstack([csr_matrix(capacities.reshape(1, -1)),
                      csr_matrix((1, n_hours))], format="csr")
    constraints = LinearConstraint(
        vstack([lhs_pos, lhs_neg, cap_row], format="csr"),
        lb=np.concatenate([np.full(2*n_hours, -np.inf), [min_cap]]),
        ub=np.concatenate([totals, -totals, [total_cap - min_cap]]),
    )

    print(f"[MILP] sites={n_sites}, active hours={n_hours}/{len(weights)}, "
          f"continuous vars={n_hours}, solver time_limit={seconds:.0f}s, relative gap={mip_rel_gap:g}", flush=True)
    t0 = time.monotonic()
    res = milp(c=c, integrality=integrality, bounds=bounds,
               constraints=constraints,
               options={"time_limit": seconds, "mip_rel_gap": mip_rel_gap,
                        "presolve": True, "disp": False})
    elapsed = time.monotonic() - t0

    selected_mask = None
    if res.x is not None and np.isfinite(res.x[:n_sites]).all():
        x = res.x[:n_sites]
        # Never silently accept a fractional/non-feasible MILP incumbent.
        if (np.max(np.abs(x - np.rint(x))) <= 1e-5):
            possible = np.rint(x).astype(bool)
            cap1 = float(np.sum(capacities[possible]))
            if min_cap-1e-6 <= cap1 <= total_cap-min_cap+1e-6:
                selected_mask = possible

    def safe_num(x):
        if x is None:
            return None
        try:
            fx = float(x)
            return fx if np.isfinite(fx) else None
        except (TypeError, ValueError):
            return None

    info = {
        "solver": "SciPy scipy.optimize.milp (HiGHS)",
        "status": int(res.status),
        "message": str(res.message),
        "success": bool(res.success),
        "feasible_incumbent_found": selected_mask is not None,
        "solver_seconds": elapsed,
        "time_limit_seconds": seconds,
        "relative_gap_tolerance": mip_rel_gap,
        "mip_gap": safe_num(getattr(res, "mip_gap", None)),
        "mip_node_count": safe_num(getattr(res, "mip_node_count", None)),
        "mip_dual_bound_million_krw": safe_num(getattr(res, "mip_dual_bound", None)),
        "mip_primal_bound_million_krw": safe_num(getattr(res, "fun", None)),
        "active_hours": n_hours,
        "total_hours": len(weights),
    }
    print(f"[MILP] status={info['status']} ({info['message']}), "
          f"elapsed={elapsed:.1f}s, feasible={selected_mask is not None}, "
          f"reported gap={info['mip_gap']}", flush=True)
    return selected_mask, info


def main():
    ap = argparse.ArgumentParser(description="MAE Loss + MILP, 60:20:20")
    ap.add_argument("--data-dir", type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--source-dir", type=Path, default=None,
                    help="Existing 60/20/20 MAE Loss output folder")
    ap.add_argument("--time-limit", type=float, default=300.0,
                    help="MILP solver limit in seconds (default 300 = 5 minutes)")
    ap.add_argument("--mip-rel-gap", type=float, default=1e-4)
    args = ap.parse_args()
    if args.time_limit <= 0 or args.mip_rel_gap < 0:
        ap.error("time-limit must be >0 and gap >=0")
    core = load_core()
    root = args.data_dir.expanduser().resolve()
    source = (args.source_dir or (root / "output" / "validation_602020" / "split602020_l1_loss")).expanduser()
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    out = root / "output" / "validation_602020" / "split602020_l1_loss_milp"
    out.mkdir(parents=True, exist_ok=True)

    summary_file = source / "summary.json"
    if not summary_file.is_file():
        raise FileNotFoundError(
            f"Missing completed 60:20:20 MAE Loss run: {summary_file}\n"
            "Run python3 codes/baseline_validation_mae_loss.py first, "
            "or pass --source-dir PATH."
        )
    with summary_file.open(encoding="utf-8") as f:
        old = json.load(f)
    if old.get("experiment") != "l1_loss" or old.get("validation_type") != "random_delivery_day_single_split_60_20_20":
        raise ValueError("Source summary is not the 60:20:20 MAE Loss experiment")
    if old.get("seed") != 42:
        print(f"[MILP] NOTE source seed={old.get('seed')}, preserving its exact split", flush=True)

    opt = core.load_module(HERE / "baseline_optimizer.py", "psat_greedy_for_milp")
    sites = pd.read_csv(root / "sites.csv")[["site_id", "current_group", "capacity_kw"]]
    if len(sites) != 253 or sites.site_id.duplicated().any():
        raise ValueError("Expected 253 unique sites")
    capacities = sites.capacity_kw.to_numpy(np.float64)

    split = pd.read_csv(source / "split_days.csv")
    if split.day_kst.duplicated().any() or set(split.split) != {"model_train", "group_opt", "holdout"}:
        raise ValueError("Invalid source split_days.csv")
    days_opt = set(split.loc[split.split == "group_opt", "day_kst"])
    days_hold = set(split.loc[split.split == "holdout", "day_kst"])
    if days_opt & days_hold:
        raise ValueError("Group-opt and holdout dates overlap")

    gen = pd.read_parquet(root / "train" / "generation.parquet", columns=["site_id", "ts", "energy"])
    gen["ts"] = core.to_kst(gen.ts)
    gen_days = core.days_of(gen.ts).dt.strftime("%Y-%m-%d")
    group_actual = gen.loc[gen_days.isin(days_opt)].copy()
    hold_actual = gen.loc[gen_days.isin(days_hold)].copy()
    del gen, gen_days

    def read_source_preds(name: str, desired_days: set[str]):
        preds = pd.read_parquet(source / name, columns=["site_id", "ts", "pred_kwh"])
        preds["ts"] = core.to_kst(preds.ts)
        pred_days = set(core.days_of(preds.ts).dt.strftime("%Y-%m-%d"))
        if pred_days != desired_days:
            raise ValueError(f"{name} dates do not match source split_days.csv")
        return preds

    group_pred = read_source_preds("group_opt_site_predictions.parquet", days_opt)
    hold_pred = read_source_preds("holdout_site_predictions.parquet", days_hold)
    print(f"[MILP] Reusing MAE Loss model predictions from {source}", flush=True)

    group_matrix, group_times, group_missing = core.residual_matrix(
        group_actual, group_pred, sites.site_id, "GROUP_OPT")
    hold_matrix, hold_times, hold_missing = core.residual_matrix(
        hold_actual, hold_pred, sites.site_id, "HOLDOUT")
    del group_actual, hold_actual, group_pred, hold_pred

    group_weights = opt.time_weights(root, group_times)
    initial = opt.initial_mask(sites)
    opt.check_capacity(initial, capacities)
    greedy_file = source / "composition.csv"
    greedy = None
    if greedy_file.is_file():
        g = pd.read_csv(greedy_file)
        if g.site_id.duplicated().any() or set(g.site_id) != set(sites.site_id):
            raise ValueError("Source MAE loss Greedy assignment does not cover 253 sites")
        greedy = sites[["site_id"]].merge(g, on="site_id", how="left", validate="one_to_one")
        if not greedy.group.isin(["G1", "G2"]).all():
            raise ValueError("Invalid Greedy composition")
        greedy = greedy.group.eq("G1").to_numpy(dtype=bool)
        opt.check_capacity(greedy, capacities)

    milp_mask, solver = solve_milp(
        group_matrix, group_weights, capacities, seconds=args.time_limit, mip_rel_gap=args.mip_rel_gap
    )
    # Always evaluate the raw MILP solution if one exists. Do NOT select based
    # on independent holdout or auto-choose Greedy just because it beats MILP.
    # If the solver finds no feasible solution in 300s, retain Greedy ONLY as
    # a clearly labeled fallback; that run is not a successful MILP experiment.
    if milp_mask is None:
        if greedy is None:
            raise RuntimeError("MILP found no feasible incumbent; no saved MAE-loss Greedy assignment for fallback")
        selected = greedy
        selected_algorithm = "Greedy fallback (MILP found no feasible incumbent; NOT a MILP result)"
    else:
        selected = milp_mask
        selected_algorithm = "MILP incumbent"
    opt.check_capacity(selected, capacities)

    group_score = core.score_group(opt, root, group_matrix, group_times, initial, selected)
    hold_score = core.score_group(opt, root, hold_matrix, hold_times, initial, selected)
    group_score["improvable_cost_krw"] = group_score["W_optimized_krw"] - group_score["W_lower_bound_krw"]
    hold_score["improvable_cost_krw"] = hold_score["W_optimized_krw"] - hold_score["W_lower_bound_krw"]
    for score in (group_score, hold_score):
        denom = score["W_initial_krw"] - score["W_lower_bound_krw"]
        score["improved_percentage_of_theoretical"] = 100 * score["W_reduction_krw"] / denom if denom > 0 else None

    greedy_comparison = None
    if greedy is not None:
        greedy_comparison = {
            "group_optimization": core.score_group(opt, root, group_matrix, group_times, initial, greedy),
            "independent_holdout": core.score_group(opt, root, hold_matrix, hold_times, initial, greedy),
        }
    milp_comparison = None
    if milp_mask is not None:
        milp_comparison = {
            "group_optimization": core.score_group(opt, root, group_matrix, group_times, initial, milp_mask),
            "independent_holdout": core.score_group(opt, root, hold_matrix, hold_times, initial, milp_mask),
        }

    pd.DataFrame({"site_id": sites.site_id,
                  "group": np.where(selected, "G1", "G2")}).to_csv(out / "composition.csv", index=False)
    result = {
        "experiment": "mae_loss_milp",
        "validation_type": "random_delivery_day_single_split_60_20_20",
        "source_experiment": str(source),
        "seed": old["seed"],
        "model": "Existing LightGBM MAE Loss (no refit)",
        "model_metrics_holdout": old["model_metrics_holdout"],
        "group_optimization_in_sample": group_score,
        "group_evaluation_independent_holdout": hold_score,
        "solver": solver,
        "selected_algorithm": selected_algorithm,
        "comparison_original_greedy": greedy_comparison,
        "comparison_raw_milp": milp_comparison,
        "group_opt_missing_site_hours": group_missing,
        "holdout_missing_site_hours": hold_missing,
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)

    print("[MILP] " + "="*72, flush=True)
    print(f"[MILP] Selected assignment = {selected_algorithm}", flush=True)
    print(f"[MILP] HOLDOUT MODEL weighted MAE = {old['model_metrics_holdout']['weighted_MAE_kwh']:.5f} kWh", flush=True)
    print(f"[MILP] GROUP OPT initial W      = {group_score['W_initial_krw']:,.2f} KRW", flush=True)
    print(f"[MILP] GROUP OPT optimized W    = {group_score['W_optimized_krw']:,.2f} KRW", flush=True)
    print(f"[MILP] GROUP OPT lower bound    = {group_score['W_lower_bound_krw']:,.2f} KRW", flush=True)
    print(f"[MILP] HOLDOUT initial W        = {hold_score['W_initial_krw']:,.2f} KRW", flush=True)
    print(f"[MILP] HOLDOUT optimized W      = {hold_score['W_optimized_krw']:,.2f} KRW", flush=True)
    print(f"[MILP] HOLDOUT lower bound      = {hold_score['W_lower_bound_krw']:,.2f} KRW", flush=True)
    print(f"[MILP] HOLDOUT improvable cost  = {hold_score['improvable_cost_krw']:,.2f} KRW", flush=True)
    print(f"[MILP] HOLDOUT improved percent = {hold_score['improved_percentage_of_theoretical']:.4f}%", flush=True)
    if milp_comparison is not None and greedy_comparison is not None:
        print(f"[MILP] Raw MILP vs Greedy (GROUP OPT) = {milp_comparison['group_optimization']['W_optimized_krw']:,.2f} "
              f"vs {greedy_comparison['group_optimization']['W_optimized_krw']:,.2f} KRW", flush=True)
        print(f"[MILP] Raw MILP vs Greedy (HOLDOUT)   = {milp_comparison['independent_holdout']['W_optimized_krw']:,.2f} "
              f"vs {greedy_comparison['independent_holdout']['W_optimized_krw']:,.2f} KRW", flush=True)
    print(f"[MILP] Results saved: {out}", flush=True)


if __name__ == "__main__":
    main()
