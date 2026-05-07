"""
Unit tests for src/features/featurize.py (Component 2 / pipeline Stage 3).
Covers: reference/production split, drift injection, and the full main() run.
No real dataset required — all inputs are synthetic DataFrames.
"""

import pathlib

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.featurize import create_reference_production_split, inject_drift

# ─── Shared params fixture ────────────────────────────────────────────────────

PARAMS = {
    "data": {
        "target_column": "Churn",
        "random_seed": 42,
        "test_size": 0.2,
        "reference_ratio": 0.7,
        "processed_path": "",   # overridden per-test
        "train_path": "",
        "test_path": "",
        "reference_path": "",
        "production_path": "",
        "drop_columns": [],
    },
    "preprocessing": {
        "pipeline_artifact": "",
        "numeric_features": ["tenure", "MonthlyCharges", "TotalCharges"],
        "categorical_features": ["gender", "Contract"],
        "imputer_strategy": "median",
        "scaler": "standard",
        "smote_random_state": 42,
        "smote_k_neighbors": 5,
    },
    "monitoring": {
        "perturbation_features": ["MonthlyCharges", "tenure"],
        "perturbation_std_multiplier": 2.0,
    },
}


def _make_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "tenure": rng.integers(0, 72, n).astype(float),
        "MonthlyCharges": rng.uniform(18, 120, n),
        "TotalCharges": rng.uniform(18, 8000, n),
        "gender": rng.choice(["Male", "Female"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "Churn": rng.choice([0, 1], n, p=[0.73, 0.27]),
    })


# ─── Test 1: reference/production split respects ratio ────────────────────────

def test_reference_production_split_ratio():
    df = _make_df(200)
    ref, prod = create_reference_production_split(df, reference_ratio=0.7)

    assert len(ref) == 140
    assert len(prod) == 60
    assert len(ref) + len(prod) == len(df)


# ─── Test 2: split is non-overlapping ─────────────────────────────────────────

def test_reference_production_split_no_overlap():
    df = _make_df(100)
    ref, prod = create_reference_production_split(df, reference_ratio=0.6)

    ref_idx = set(ref.index)
    prod_idx = set(prod.index)
    assert ref_idx.isdisjoint(prod_idx)


# ─── Test 3: inject_drift modifies target columns ─────────────────────────────

def test_inject_drift_modifies_target_columns():
    df = _make_df(300)
    original_monthly = df["MonthlyCharges"].copy()
    original_tenure = df["tenure"].copy()

    drifted = inject_drift(df, PARAMS)

    # At least some values should have changed
    assert not (drifted["MonthlyCharges"] == original_monthly).all()
    assert not (drifted["tenure"] == original_tenure).all()


# ─── Test 4: inject_drift leaves non-target columns unchanged ─────────────────

def test_inject_drift_leaves_other_columns_unchanged():
    df = _make_df(200)
    original_total = df["TotalCharges"].copy()

    params_narrow = {**PARAMS, "monitoring": {
        "perturbation_features": ["MonthlyCharges"],  # only this one
        "perturbation_std_multiplier": 2.0,
    }}
    drifted = inject_drift(df, params_narrow)

    pd.testing.assert_series_equal(drifted["TotalCharges"], original_total)


# ─── Test 5: inject_drift does not change DataFrame shape ─────────────────────

def test_inject_drift_preserves_shape():
    df = _make_df(150)
    drifted = inject_drift(df, PARAMS)

    assert drifted.shape == df.shape


# ─── Test 6: main() end-to-end with mocked filesystem ────────────────────────

def test_featurize_main_creates_all_four_splits(tmp_path, monkeypatch):
    """Run main() with a tmp_path-based filesystem — no real data needed."""
    from src.data.preprocess import build_full_pipeline
    from unittest.mock import patch

    df = _make_df(300)

    # Write a processed CSV to tmp_path
    processed_csv = tmp_path / "processed.csv"
    df.to_csv(processed_csv, index=False)

    # Build and serialise a minimal pipeline
    pipe = build_full_pipeline(PARAMS)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    pipe.fit(X, y)
    pipeline_path = tmp_path / "pipeline.joblib"
    joblib.dump(pipe, pipeline_path)

    # Build params pointing to tmp dirs
    test_params = {
        **PARAMS,
        "data": {
            **PARAMS["data"],
            "processed_path": str(processed_csv),
            "train_path": str(tmp_path / "train.csv"),
            "test_path": str(tmp_path / "test.csv"),
            "reference_path": str(tmp_path / "reference.csv"),
            "production_path": str(tmp_path / "production.csv"),
        },
        "preprocessing": {
            **PARAMS["preprocessing"],
            "pipeline_artifact": str(pipeline_path),
        },
    }

    with patch("src.features.featurize.load_params", return_value=test_params):
        from src.features import featurize
        featurize.main()

    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "test.csv").exists()
    assert (tmp_path / "reference.csv").exists()
    assert (tmp_path / "production.csv").exists()
