"""
Unit tests for preprocessing pipeline (Component 2).
At least 3 transformation tests as required by the rubric.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline


# ─── Fixtures ────────────────────────────────────────────────────────────────

PARAMS = {
    "data": {"target_column": "Churn", "random_seed": 42},
    "preprocessing": {
        "numeric_features": ["tenure", "MonthlyCharges", "TotalCharges"],
        "categorical_features": ["gender", "Contract", "PaymentMethod"],
        "imputer_strategy": "median",
        "smote_random_state": 42,
        "smote_k_neighbors": 5,
    },
}


def _make_clean_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, n).astype(float),
            "MonthlyCharges": rng.uniform(18, 120, n),
            "TotalCharges": rng.uniform(18, 8000, n),
            "gender": rng.choice(["Male", "Female"], n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)"], n
            ),
            "Churn": rng.choice([0, 1], n, p=[0.73, 0.27]),
        }
    )


def _make_df_with_nulls(n: int = 200) -> pd.DataFrame:
    df = _make_clean_df(n)
    rng = np.random.default_rng(99)
    null_idx = rng.choice(n, size=20, replace=False)
    df.loc[null_idx, "TotalCharges"] = np.nan
    return df


# ─── Test 1: Preprocessor output shape is correct ────────────────────────────

def test_preprocessor_output_shape():
    from src.data.preprocess import build_preprocessor

    df = _make_clean_df(100)
    X = df.drop(columns=["Churn"])
    preprocessor = build_preprocessor(PARAMS)
    X_t = preprocessor.fit_transform(X)

    # numeric (3) + one-hot encoded categoricals
    assert X_t.shape[0] == 100
    assert X_t.shape[1] > 3  # must have expanded categoricals


# ─── Test 2: Imputer fills missing values — no NaN after transform ────────────

def test_imputer_handles_missing_values():
    from src.data.preprocess import build_preprocessor

    df = _make_df_with_nulls(200)
    X = df.drop(columns=["Churn"])
    assert X["TotalCharges"].isnull().any(), "Expected NaN values in fixture"

    preprocessor = build_preprocessor(PARAMS)
    X_t = preprocessor.fit_transform(X)

    assert not np.isnan(X_t).any(), "NaN values should be imputed"


# ─── Test 3: StandardScaler produces zero mean / unit variance ───────────────

def test_numeric_features_are_scaled():
    from src.data.preprocess import build_preprocessor

    df = _make_clean_df(500)
    X = df.drop(columns=["Churn"])
    preprocessor = build_preprocessor(PARAMS)
    X_t = preprocessor.fit_transform(X)

    # The first 3 columns are numeric (tenure, MonthlyCharges, TotalCharges)
    numeric_cols = X_t[:, :3]
    means = numeric_cols.mean(axis=0)
    stds = numeric_cols.std(axis=0)

    np.testing.assert_allclose(means, 0.0, atol=0.1)
    np.testing.assert_allclose(stds, 1.0, atol=0.1)


# ─── Test 4: OneHotEncoder expands categorical columns ───────────────────────

def test_categorical_encoding_increases_columns():
    from src.data.preprocess import build_preprocessor

    df = _make_clean_df(100)
    X = df.drop(columns=["Churn"])
    n_original_features = X.shape[1]

    preprocessor = build_preprocessor(PARAMS)
    X_t = preprocessor.fit_transform(X)

    assert X_t.shape[1] > n_original_features


# ─── Test 5: Full pipeline serialises and deserialises correctly ──────────────

def test_pipeline_serialisation(tmp_path):
    import joblib
    from src.data.preprocess import build_full_pipeline

    df = _make_clean_df(300)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    pipeline = build_full_pipeline(PARAMS)
    pipeline.fit(X, y)

    artifact = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, artifact)
    loaded = joblib.load(artifact)

    # Both pipelines should produce identical output on new data
    df_new = _make_clean_df(50)
    X_new = df_new.drop(columns=["Churn"])
    preprocessor_orig = pipeline.named_steps["preprocessor"]
    preprocessor_loaded = loaded.named_steps["preprocessor"]

    out_orig = preprocessor_orig.transform(X_new)
    out_loaded = preprocessor_loaded.transform(X_new)

    np.testing.assert_array_almost_equal(out_orig, out_loaded)


# ─── Test 6: Clean data preparation strips target whitespace correctly ────────

def test_prepare_clean_encodes_target():
    from src.data.prepare import clean_data

    df = pd.DataFrame(
        {
            "customerID": ["A", "B", "C"],
            "tenure": [1, 2, 3],
            "MonthlyCharges": [50.0, 60.0, 70.0],
            "TotalCharges": ["100", " ", "200"],
            "Churn": ["Yes", "No", "Yes"],
        }
    )
    params = {
        "data": {"drop_columns": ["customerID"], "target_column": "Churn"}
    }
    cleaned = clean_data(df, params)

    assert "customerID" not in cleaned.columns
    assert cleaned["Churn"].tolist() == [1, 0, 1]
    assert cleaned["TotalCharges"].dtype in [float, np.float64]
