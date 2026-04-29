"""
Bonus B — Apache Airflow DAG
Training pipeline orchestration with 5 tasks:
validate_data → preprocess → train → evaluate → register_model
Failure in any task halts all downstream tasks.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

# ─── Default arguments ────────────────────────────────────────────────────────
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ─── Task functions ───────────────────────────────────────────────────────────

def validate_data_fn(**context):
    """Validate raw data schema before any processing."""
    import pandas as pd
    import pandera as pa
    import yaml
    from pandera import Check, Column, DataFrameSchema

    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)

    schema = DataFrameSchema(
        {
            "tenure": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
            "MonthlyCharges": Column(float, nullable=True),
            "Churn": Column(object, Check.isin(["Yes", "No"])),
        },
        coerce=True,
    )

    df = pd.read_csv(params["data"]["raw_path"])
    schema.validate(df[["tenure", "MonthlyCharges", "Churn"]])
    log.info("Data validation passed: %d rows", len(df))
    context["ti"].xcom_push(key="row_count", value=len(df))


def preprocess_fn(**context):
    """Run prepare + preprocess + featurize stages."""
    from src.data.prepare import load_params, load_raw_data, clean_data
    import pathlib
    import pandas as pd

    params = load_params()
    df = load_raw_data(params["data"]["raw_path"])
    df = clean_data(df, params)
    df.to_csv(params["data"]["processed_path"], index=False)

    from src.data.preprocess import fit_and_save_pipeline
    fit_and_save_pipeline(df, params, params["preprocessing"]["pipeline_artifact"])

    from src.features.featurize import main as featurize_main
    featurize_main()

    log.info("Preprocessing complete.")


def train_fn(**context):
    """Run all three experiments and log to MLflow."""
    import mlflow
    import yaml
    from src.training.train import (
        load_splits,
        run_logistic_regression,
        run_random_forest,
        run_gradient_boosting,
    )

    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["training"]["experiment_name"])

    X_train, y_train, X_test, y_test = load_splits(params)
    run_ids = [
        run_logistic_regression(X_train, y_train, X_test, y_test, params),
        run_random_forest(X_train, y_train, X_test, y_test, params),
        run_gradient_boosting(X_train, y_train, X_test, y_test, params),
    ]
    context["ti"].xcom_push(key="run_ids", value=run_ids)
    log.info("Training complete. Run IDs: %s", run_ids)


def evaluate_fn(**context):
    """Assert that the best run exceeds the minimum performance threshold."""
    import yaml
    from mlflow import MlflowClient
    import mlflow

    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)

    run_ids = context["ti"].xcom_pull(key="run_ids", task_ids="train")
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    client = MlflowClient()

    min_auc = params["mlflow"]["min_roc_auc_threshold"]
    for rid in run_ids:
        auc = client.get_run(rid).data.metrics.get("roc_auc", 0)
        log.info("Run %s: roc_auc=%.4f", rid, auc)

    best_auc = max(
        client.get_run(rid).data.metrics.get("roc_auc", 0) for rid in run_ids
    )
    if best_auc < min_auc:
        raise ValueError(
            f"Best model roc_auc ({best_auc:.4f}) below threshold ({min_auc}). "
            "Halting pipeline."
        )
    log.info("Evaluation gate passed: best roc_auc=%.4f", best_auc)
    context["ti"].xcom_push(key="best_auc", value=best_auc)


def register_model_fn(**context):
    """Register best model and promote to Production."""
    import yaml
    from src.training.train import register_and_promote_best

    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)

    run_ids = context["ti"].xcom_pull(key="run_ids", task_ids="train")
    register_and_promote_best(run_ids, params)
    log.info("Model registered and promoted to Production.")


# ─── DAG definition ───────────────────────────────────────────────────────────
with DAG(
    dag_id="telco_churn_training_pipeline",
    default_args=default_args,
    description="End-to-end MLOps training pipeline for Telco Churn prediction",
    schedule_interval="@weekly",
    catchup=False,
    tags=["mlops", "telco-churn", "ddsc611"],
) as dag:

    t1_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data_fn,
    )

    t2_preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess_fn,
    )

    t3_train = PythonOperator(
        task_id="train",
        python_callable=train_fn,
    )

    t4_evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=evaluate_fn,
    )

    t5_register = PythonOperator(
        task_id="register_model",
        python_callable=register_model_fn,
    )

    # ── DAG topology ──────────────────────────────────────────────────────────
    t1_validate >> t2_preprocess >> t3_train >> t4_evaluate >> t5_register
