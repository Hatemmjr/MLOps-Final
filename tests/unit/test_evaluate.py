"""
Unit tests for src/evaluation/evaluate.py (Component 3 / model validation gate).
All MLflow and filesystem calls are mocked — no live server or real model needed.
"""

import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


# ─── Shared fixtures ──────────────────────────────────────────────────────────

PARAMS = {
    "data": {
        "target_column": "Churn",
        "test_path": "",   # overridden per-test
    },
    "mlflow": {
        "tracking_uri": "sqlite:///test_mlflow.db",
        "production_stage": "Production",
    },
    "training": {
        "model_name": "telco-churn-model",
    },
    "ci": {
        "min_accuracy": 0.50,
        "min_roc_auc": 0.50,
    },
}


def _mock_model(pred_value: int = 0, prob_value: float = 0.8):
    """Return a sklearn-style mock model."""
    model = MagicMock()
    model.predict = MagicMock(side_effect=lambda X: np.array([pred_value] * len(X)))
    model.predict_proba = MagicMock(
        side_effect=lambda X: np.array([[1 - prob_value, prob_value]] * len(X))
    )
    return model


def _make_test_csv(tmp_path, n: int = 100, churn_rate: float = 0.3) -> str:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(5, 2, n),
        "Churn": rng.choice([0, 1], n, p=[1 - churn_rate, churn_rate]),
    })
    path = str(tmp_path / "test.csv")
    df.to_csv(path, index=False)
    return path


# ─── Test 1: evaluate_model returns required keys ─────────────────────────────

def test_evaluate_model_returns_accuracy_and_roc_auc(tmp_path):
    from src.evaluation.evaluate import evaluate_model

    test_path = _make_test_csv(tmp_path)
    params = {**PARAMS, "data": {**PARAMS["data"], "test_path": test_path}}
    model = _mock_model()

    metrics = evaluate_model(model, params)

    assert "accuracy" in metrics
    assert "roc_auc" in metrics


# ─── Test 2: metrics are in [0, 1] ────────────────────────────────────────────

def test_evaluate_model_metrics_in_valid_range(tmp_path):
    from src.evaluation.evaluate import evaluate_model

    test_path = _make_test_csv(tmp_path)
    params = {**PARAMS, "data": {**PARAMS["data"], "test_path": test_path}}
    model = _mock_model()

    metrics = evaluate_model(model, params)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


# ─── Test 3: validate_thresholds — passes when metrics above min ──────────────

def test_validate_thresholds_passes_when_above_min():
    from src.evaluation.evaluate import validate_thresholds

    metrics = {"accuracy": 0.85, "roc_auc": 0.90}
    assert validate_thresholds(metrics, PARAMS) is True


# ─── Test 4: validate_thresholds — fails on low accuracy ──────────────────────

def test_validate_thresholds_fails_on_low_accuracy():
    from src.evaluation.evaluate import validate_thresholds

    metrics = {"accuracy": 0.30, "roc_auc": 0.80}  # accuracy below 0.50
    assert validate_thresholds(metrics, PARAMS) is False


# ─── Test 5: validate_thresholds — fails on low roc_auc ──────────────────────

def test_validate_thresholds_fails_on_low_roc_auc():
    from src.evaluation.evaluate import validate_thresholds

    metrics = {"accuracy": 0.80, "roc_auc": 0.40}  # roc_auc below 0.50
    assert validate_thresholds(metrics, PARAMS) is False


# ─── Test 6: validate_thresholds — fails on BOTH low ─────────────────────────

def test_validate_thresholds_fails_when_both_below():
    from src.evaluation.evaluate import validate_thresholds

    metrics = {"accuracy": 0.30, "roc_auc": 0.40}
    assert validate_thresholds(metrics, PARAMS) is False


# ─── Test 7: load_production_model calls mlflow with correct URI ──────────────

def test_load_production_model_calls_mlflow(tmp_path):
    from src.evaluation import evaluate

    mock_model = _mock_model()
    with patch("src.evaluation.evaluate.mlflow.set_tracking_uri") as mock_uri, \
         patch(
             "src.evaluation.evaluate.mlflow.sklearn.load_model",
             return_value=mock_model,
         ) as mock_load:

        result = evaluate.load_production_model(PARAMS)

        mock_uri.assert_called_once_with(PARAMS["mlflow"]["tracking_uri"])
        mock_load.assert_called_once_with("models:/telco-churn-model/Production")
        assert result is mock_model


# ─── Test 8: load_params reads YAML correctly ─────────────────────────────────

def test_load_params_reads_yaml(tmp_path):
    from src.evaluation.evaluate import load_params

    yaml_content = "data:\n  target_column: Churn\n  test_path: data/splits/test.csv\n"
    yaml_file = tmp_path / "params.yaml"
    yaml_file.write_text(yaml_content)

    params = load_params(str(yaml_file))
    assert params["data"]["target_column"] == "Churn"
