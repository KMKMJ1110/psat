#!/usr/bin/env python3
"""OIBC 60/20/20: Multiple Linear Regression (OLS) + baseline Greedy.

Place next to the user's 60/20/20 ``codes/baseline_validation.py``.
Run: python3 codes/baseline_validation_linear_regression.py

Reuses the exact same 60/20/20 date split, features, outputs, scoring and
Greedy as the user's 60/20/20 baseline. No contest data leave the local PC.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
CORE_PATH = THIS_DIR / "baseline_validation.py"
spec = importlib.util.spec_from_file_location("psat_602020_core_linear", CORE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load 60/20/20 core: {CORE_PATH}")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

if not all(hasattr(core, key) for key in ("split_602020", "run", "fit_models", "RUN_NAMES", "VARIANTS")):
    raise RuntimeError(
        f"{CORE_PATH} must be the previously supplied 60:20:20 baseline script, "
        "not the old 5-Fold version."
    )


def fit_linear_models(base, sites, fit, fa, fb, variant):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {}
    rounds = (10, 17)
    details = {
        "algorithm": "Multiple Linear Regression (ordinary least squares)",
        "imputer": "median (preserve all 24 features)",
        "scaling": "StandardScaler",
        "objective": "MSE / OLS",
        "n_features": len(base.FEATS),
        "fitting_rounds": [10, 17],
        "round_aggregation": "50:50",
    }

    for r in rounds:
        df = base.features(fit, sites, fa, fb, None, r)
        if len(df) != len(fit):
            raise ValueError(f"round={r}: feature join changed row count")
        df = df.loc[df.elev > 0]
        X = df[base.FEATS].to_numpy(dtype=np.float32)
        y = (df.energy / df.cap).clip(0, 1.2).to_numpy(dtype=np.float32)
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite normalized training label")
        # The original LightGBM handles missing weather fields natively;
        # OLS does not, so use train-only median imputation.  Keep the same FEATS.
        X[~np.isfinite(X)] = np.nan
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LinearRegression(),
        )
        core.log(f"Fitting round={r}, X={X.shape}, objective=OLS (median imputation + scale)")
        model.fit(X, y)
        core.log(f"round={r} train normalized MAE={np.abs(model.predict(X)-y).mean():.6f}")
        models[r] = model
        del df, X, y
    return models, rounds, details


if __name__ == "__main__":
    # The 60/20/20 code calls fit_models via its own module globals.
    core.VARIANTS = tuple(core.VARIANTS) + ("linear_regression",)
    core.RUN_NAMES["linear_regression"] = "split602020_linear_regression"
    core.fit_models = fit_linear_models
    # In predict_eval the unchanged 60/20/20 core calls .predict on DataFrames.
    # scikit-learn's SimpleImputer accepts NaN but not infinity, so guard
    # unexpected infinite features without altering the 24-feature definition.
    original_predict_eval = core.predict_eval

    def predict_eval_ols(base, models, rounds, sites, actuals, fa, fb):
        # Identical aggregation and night / negative clipping as baseline;
        # just normalize nonfinite input values before the sklearn pipeline.
        import pandas as pd
        combined = None
        for r in rounds:
            df = base.features(actuals, sites, fa, fb, None, r)
            if len(df) != len(actuals):
                raise ValueError(f"round {r}: feature join changed evaluation row count")
            X = df[base.FEATS].to_numpy(dtype=np.float32)
            X[~np.isfinite(X)] = np.nan
            prediction = models[r].predict(X)
            kwh = np.maximum(np.where(df.elev.to_numpy() > 0, prediction, 0.0), 0.0) * df.cap.to_numpy()
            one = df[["site_id", "ts"]].copy()
            one[f"round_{r}"] = kwh
            if one.duplicated(["site_id", "ts"]).any():
                raise ValueError("Duplicate prediction keys")
            combined = one if combined is None else combined.merge(one, on=["site_id", "ts"], how="inner", validate="one_to_one")
            core.log(f"round={r} predicted {len(one):,} unseen site-hours")
        if combined is None or len(combined) != len(actuals):
            raise ValueError("Missing evaluation predictions")
        cols = [f"round_{r}" for r in rounds]
        if not np.isfinite(combined[cols].to_numpy()).all():
            raise ValueError("Nonfinite predictions")
        combined["pred_kwh"] = combined[cols].mean(axis=1)
        return combined[["site_id", "ts", "pred_kwh"]]

    core.predict_eval = predict_eval_ols
    core.run("linear_regression")
