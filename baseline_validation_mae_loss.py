#!/usr/bin/env python3
"""60/20/20 with LightGBM L1 (regression_l1) objective, other settings unchanged."""
from baseline_validation import run

if __name__ == "__main__":
    run("l1_loss")
