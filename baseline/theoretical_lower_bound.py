from pathlib import Path
import json
import numpy as np

import baseline_optimizer as opt

# ==========================================
# 1. Paths
# ==========================================
ROOT = Path(__file__).resolve().parent.parent
RUN_NAME = "lgbm_random_day_5fold"

PRED_PATH = (
    ROOT / "output" / "baseline_validation"
    / RUN_NAME / "validation_site_predictions.parquet"
)

OPT_DIR = ROOT / "output" / "baseline_optimizer" / RUN_NAME
SUMMARY_PATH = OPT_DIR / "summary.json"

# ==========================================
# 2. Load Existing OOF Residuals
# ==========================================
sites = opt.load_sites(ROOT)
pred = opt.load_predictions(PRED_PATH, sites)
residuals = opt.load_observed_residuals(ROOT, pred)

# Uses the same missing-residual = 0 policy
matrix, times = opt.complete_matrix(
    residuals,
    sites.site_id.to_list(),
    None,
    None,
    label="LOWER BOUND"
)

weights = opt.time_weights(ROOT, times)

# ==========================================
# 3. Calculate Theoretical Lower Bound
# ==========================================
total_error = matrix.sum(axis=1)

W_LB = float(
    0.10 * np.dot(weights, np.abs(total_error))
)

# ==========================================
# 4. Load Existing Greedy Results
# ==========================================
with open(SUMMARY_PATH, encoding="utf-8") as f:
    summary = json.load(f)

W_initial = summary["W_initial_krw"]
W_greedy = summary["W_optimized_krw"]

max_improvement = (
    (W_initial - W_LB) / W_initial * 100
    if W_initial > 0 else 0.0
)

greedy_improvement = (
    (W_initial - W_greedy) / W_initial * 100
    if W_initial > 0 else 0.0
)

available_gap = W_initial - W_LB

gap_utilization = (
    (W_initial - W_greedy) / available_gap * 100
    if available_gap > 0 else 0.0
)

# ==========================================
# 5. Print Results
# ==========================================
print("=" * 65)
print("OIBC THEORETICAL LOWER BOUND")
print("=" * 65)

print(f"Evaluation hours       : {len(times):,}")
print(f"Initial W              : {W_initial:,.2f} KRW")
print(f"Greedy W               : {W_greedy:,.2f} KRW")
print(f"Theoretical Lower Bound: {W_LB:,.2f} KRW")

print("-" * 65)

print(f"Greedy improvement     : {greedy_improvement:.4f}%")
print(f"Maximum possible improvement (upper bound): {max_improvement:.4f}%")
print(f"Greedy gap utilization : {gap_utilization:.4f}%")
print(f"Remaining gap to LB    : {W_greedy - W_LB:,.2f} KRW")

print("=" * 65)

# ==========================================
# 6. Save Results
# ==========================================
results = {
    "evaluation_hours": len(times),
    "W_initial_krw": W_initial,
    "W_greedy_krw": W_greedy,
    "W_lower_bound_krw": W_LB,
    "greedy_improvement_pct": greedy_improvement,
    "max_possible_improvement_pct": max_improvement,
    "greedy_gap_utilization_pct": gap_utilization,
    "remaining_gap_to_lower_bound_krw": W_greedy - W_LB,
}

with open(
    OPT_DIR / "theoretical_lower_bound.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved: theoretical_lower_bound.json")