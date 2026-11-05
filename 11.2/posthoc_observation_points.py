#!/usr/bin/env python3
"""B: separate station-group Ridge regression corrections for WA/WB.

Fit only on saved 60/20/20 GROUP_OPT predictions/actuals; independently
score HOLDOUT. Requires posthoc_stats.py in the same codes/ folder.
"""
from posthoc_stats import run

if __name__ == "__main__":
    run("observation_points")
