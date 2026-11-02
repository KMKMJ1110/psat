#!/usr/bin/env python3
"""C: one regularized linear regression (Ridge) residual model per plant.

Fit only on saved 60/20/20 GROUP_OPT predictions/actuals; independently
score HOLDOUT. Requires posthoc_stats.py in the same codes/ folder.
"""
from posthoc_stats import run

if __name__ == "__main__":
    run("lr_model")
