"""
Evaluation utilities — used by CI/CD model validation gate and standalone.
Loads the Production model from MLflow and asserts minimum thresholds.
"""

import logging
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_params(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_production_model(params: dict):
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    model_name = params["training"]["model_name"]
    stage = params["mlflow"]["production_stage"]
    model_uri = f"models:/{model_name}/{stage}"
    log.info("Loading model from %s", model_uri)
    return mlflow.sklearn.load_model(model_uri)


def evaluate_model(model, params: dict) -> dict:
    dp = params["data"]
    test_df = pd.read_csv(dp["test_path"])
    target = dp["target_column"]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }
    log.info("Evaluation metrics: %s", metrics)
    return metrics


def validate_thresholds(metrics: dict, params: dict) -> bool:
    ci = params["ci"]
    passed = True
    if metrics["accuracy"] < ci["min_accuracy"]:
        log.error(
            "accuracy %.4f below threshold %.4f",
            metrics["accuracy"],
            ci["min_accuracy"],
        )
        passed = False
    if metrics["roc_auc"] < ci["min_roc_auc"]:
        log.error(
            "roc_auc %.4f below threshold %.4f",
            metrics["roc_auc"],
            ci["min_roc_auc"],
        )
        passed = False
    if passed:
        log.info("All model validation thresholds passed.")
    return passed


def main() -> None:
    params = load_params()
    model = load_production_model(params)
    metrics = evaluate_model(model, params)
    ok = validate_thresholds(metrics, params)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
