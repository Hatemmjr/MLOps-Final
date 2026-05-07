"""
Unit tests for src/training/train.py (Component 3 — experiment tracking).
All MLflow I/O and Optuna calls are mocked to run instantly without a server.
"""

import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


# ─── Shared fixtures ──────────────────────────────────────────────────────────

PARAMS = {
    "data": {
        "target_column": "Churn",
        "random_seed": 42,
        "train_path": "",
        "test_path": "",
    },
    "training": {
        "experiment_name": "test-experiment",
        "model_name": "test-churn-model",
        "cv_folds": 2,
        "scoring_metric": "roc_auc",
        "n_optuna_trials": 1,
        "optuna_timeout": 30,
        "logistic_regression": {
            "C": [1.0],
            "solver": ["lbfgs"],
            "max_iter": [100],
        },
        "random_forest": {
            "n_estimators": [10],
            "max_depth": [3],
            "min_samples_split": [2],
        },
        "gradient_boosting": {
            "n_estimators": [10],
            "learning_rate": [0.1],
            "max_depth": [3],
        },
    },
    "mlflow": {
        "tracking_uri": "sqlite:///test_mlflow.db",
        "staging_stage": "Staging",
        "production_stage": "Production",
        "model_name": "test-churn-model",
    },
}


def _make_splits(n: int = 200):
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(5, 2, n),
        "f3": rng.uniform(0, 1, n),
    })
    y = pd.Series(rng.choice([0, 1], n, p=[0.73, 0.27]), name="Churn")
    return X, y, X.copy(), y.copy()


def _make_mock_run(run_id: str = "abc123", roc_auc: float = 0.85):
    run = MagicMock()
    run.info.run_id = run_id
    run.data.metrics = {"roc_auc": roc_auc}
    return run


# ─── Test 1: compute_metrics returns all expected keys ────────────────────────

def test_compute_metrics_returns_all_keys():
    from src.training.train import compute_metrics
    from sklearn.linear_model import LogisticRegression

    X_train, y_train, X_test, y_test = _make_splits(200)
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    metrics = compute_metrics(model, X_test, y_test)

    for key in ("accuracy", "roc_auc", "f1", "precision", "recall", "log_loss"):
        assert key in metrics, f"Missing metric: {key}"


# ─── Test 2: compute_metrics values are in valid range ────────────────────────

def test_compute_metrics_values_in_range():
    from src.training.train import compute_metrics
    from sklearn.linear_model import LogisticRegression

    X_train, y_train, X_test, y_test = _make_splits(200)
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    metrics = compute_metrics(model, X_test, y_test)

    for key in ("accuracy", "roc_auc", "f1", "precision", "recall"):
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"
    assert metrics["log_loss"] >= 0.0


# ─── Test 3: load_splits reads correct columns ────────────────────────────────

def test_load_splits_returns_correct_shapes(tmp_path):
    from src.training.train import load_splits

    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({
        "f1": rng.normal(0, 1, 100),
        "f2": rng.normal(5, 2, 100),
        "Churn": rng.choice([0, 1], 100),
    })
    test_df = pd.DataFrame({
        "f1": rng.normal(0, 1, 20),
        "f2": rng.normal(5, 2, 20),
        "Churn": rng.choice([0, 1], 20),
    })
    train_path = str(tmp_path / "train.csv")
    test_path = str(tmp_path / "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    params = {**PARAMS, "data": {
        **PARAMS["data"],
        "train_path": train_path,
        "test_path": test_path,
    }}
    X_train, y_train, X_test, y_test = load_splits(params)

    assert X_train.shape == (100, 2)
    assert X_test.shape == (20, 2)
    assert len(y_train) == 100
    assert len(y_test) == 20
    assert "Churn" not in X_train.columns


# ─── Test 4: load_params reads YAML file ─────────────────────────────────────

def test_load_params_reads_yaml(tmp_path):
    from src.training.train import load_params

    content = "training:\n  experiment_name: test-exp\n"
    p = tmp_path / "params.yaml"
    p.write_text(content)

    result = load_params(str(p))
    assert result["training"]["experiment_name"] == "test-exp"


# ─── Test 5: run_logistic_regression logs metrics and returns run_id ──────────

def test_run_logistic_regression_returns_run_id():
    from src.training.train import run_logistic_regression

    X_train, y_train, X_test, y_test = _make_splits(200)
    mock_run = _make_mock_run("lr-run-001", roc_auc=0.82)

    with patch("src.training.train.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        run_id = run_logistic_regression(X_train, y_train, X_test, y_test, PARAMS)

    assert run_id == "lr-run-001"


# ─── Test 6: run_random_forest returns a run_id ───────────────────────────────

def test_run_random_forest_returns_run_id():
    from src.training.train import run_random_forest

    X_train, y_train, X_test, y_test = _make_splits(200)
    mock_run = _make_mock_run("rf-run-001", roc_auc=0.83)

    with patch("src.training.train.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        run_id = run_random_forest(X_train, y_train, X_test, y_test, PARAMS)

    assert run_id == "rf-run-001"


# ─── Test 7: run_gradient_boosting returns a run_id ──────────────────────────

def test_run_gradient_boosting_returns_run_id():
    from src.training.train import run_gradient_boosting

    X_train, y_train, X_test, y_test = _make_splits(200)
    mock_run = _make_mock_run("gb-run-001", roc_auc=0.88)

    with patch("src.training.train.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        run_id = run_gradient_boosting(X_train, y_train, X_test, y_test, PARAMS)

    assert run_id == "gb-run-001"


# ─── Test 8: register_and_promote_best picks the highest roc_auc ─────────────

def test_register_and_promote_best_picks_highest_auc():
    from src.training.train import register_and_promote_best

    run_ids = ["run-lr", "run-rf", "run-gb"]
    runs = {
        "run-lr": _make_mock_run("run-lr", roc_auc=0.80),
        "run-rf": _make_mock_run("run-rf", roc_auc=0.83),
        "run-gb": _make_mock_run("run-gb", roc_auc=0.88),  # best
    }

    mock_client = MagicMock()
    mock_client.get_run.side_effect = lambda rid: runs[rid]
    mock_mv = MagicMock()
    mock_mv.version = "1"

    with patch("src.training.train.MlflowClient", return_value=mock_client), \
         patch("src.training.train.mlflow.register_model", return_value=mock_mv) as mock_register:
        register_and_promote_best(run_ids, PARAMS)

    # Should register the GB run (highest AUC = 0.88)
    mock_register.assert_called_once_with(
        model_uri="runs:/run-gb/model",
        name="test-churn-model",
    )
    # Should transition to Staging then Production
    assert mock_client.transition_model_version_stage.call_count == 2
