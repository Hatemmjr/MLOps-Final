"""
Training — Experiment Tracking & Model Registry (Component 3)
Runs three experiments with HPO via Optuna, logs everything to MLflow,
registers the best model, and promotes it to Production.
All parameters come from configs/params.yaml.
"""

import logging

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_params(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_splits(params: dict):
    dp = params["data"]
    train_df = pd.read_csv(dp["train_path"])
    test_df = pd.read_csv(dp["test_path"])
    target = dp["target_column"]
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    return X_train, y_train, X_test, y_test


def compute_metrics(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "f1": float(f1_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "log_loss": float(log_loss(y, y_prob)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 — Logistic Regression (Grid Search via Optuna)
# ─────────────────────────────────────────────────────────────────────────────
def run_logistic_regression(X_train, y_train, X_test, y_test, params: dict) -> str:
    cfg = params["training"]["logistic_regression"]
    n_trials = params["training"]["n_optuna_trials"]
    seed = params["data"]["random_seed"]
    cv = StratifiedKFold(
        n_splits=params["training"]["cv_folds"], shuffle=True, random_state=seed
    )

    def objective(trial):
        C = trial.suggest_categorical("C", cfg["C"])
        solver = trial.suggest_categorical("solver", cfg["solver"])
        max_iter = trial.suggest_categorical("max_iter", cfg["max_iter"])
        clf = LogisticRegression(
            C=C, solver=solver, max_iter=max_iter, random_state=seed
        )
        score = cross_val_score(
            clf, X_train, y_train, cv=cv, scoring="roc_auc"
        ).mean()
        return score

    with mlflow.start_run(run_name="logistic-regression-optuna") as run:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=params["training"]["optuna_timeout"],
        )
        best = study.best_params
        mlflow.log_params(best)

        model = LogisticRegression(**best, random_state=seed)
        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        run_id = run.info.run_id
        log.info("[LR] roc_auc=%.4f run_id=%s", metrics["roc_auc"], run_id)
    return run_id


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — Random Forest (Optuna)
# ─────────────────────────────────────────────────────────────────────────────
def run_random_forest(X_train, y_train, X_test, y_test, params: dict) -> str:
    cfg = params["training"]["random_forest"]
    n_trials = params["training"]["n_optuna_trials"]
    seed = params["data"]["random_seed"]
    cv = StratifiedKFold(
        n_splits=params["training"]["cv_folds"], shuffle=True, random_state=seed
    )

    def objective(trial):
        n_estimators = trial.suggest_categorical("n_estimators", cfg["n_estimators"])
        max_depth = trial.suggest_categorical("max_depth", cfg["max_depth"])
        min_samples_split = trial.suggest_categorical(
            "min_samples_split", cfg["min_samples_split"]
        )
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=seed,
            n_jobs=-1,
        )
        score = cross_val_score(
            clf, X_train, y_train, cv=cv, scoring="roc_auc"
        ).mean()
        return score

    with mlflow.start_run(run_name="random-forest-optuna") as run:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=params["training"]["optuna_timeout"],
        )
        best = study.best_params
        mlflow.log_params(best)

        model = RandomForestClassifier(**best, random_state=seed, n_jobs=-1)
        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        run_id = run.info.run_id
        log.info("[RF] roc_auc=%.4f run_id=%s", metrics["roc_auc"], run_id)
    return run_id


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3 — Gradient Boosting (Optuna)
# ─────────────────────────────────────────────────────────────────────────────
def run_gradient_boosting(X_train, y_train, X_test, y_test, params: dict) -> str:
    cfg = params["training"]["gradient_boosting"]
    n_trials = params["training"]["n_optuna_trials"]
    seed = params["data"]["random_seed"]
    cv = StratifiedKFold(
        n_splits=params["training"]["cv_folds"], shuffle=True, random_state=seed
    )

    def objective(trial):
        n_estimators = trial.suggest_categorical("n_estimators", cfg["n_estimators"])
        learning_rate = trial.suggest_categorical("learning_rate", cfg["learning_rate"])
        max_depth = trial.suggest_categorical("max_depth", cfg["max_depth"])
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=seed,
        )
        score = cross_val_score(
            clf, X_train, y_train, cv=cv, scoring="roc_auc"
        ).mean()
        return score

    with mlflow.start_run(run_name="gradient-boosting-optuna") as run:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=params["training"]["optuna_timeout"],
        )
        best = study.best_params
        mlflow.log_params(best)

        model = GradientBoostingClassifier(**best, random_state=seed)
        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        run_id = run.info.run_id
        log.info("[GB] roc_auc=%.4f run_id=%s", metrics["roc_auc"], run_id)
    return run_id


# ─────────────────────────────────────────────────────────────────────────────
# Model Registration & Promotion
# ─────────────────────────────────────────────────────────────────────────────
def register_and_promote_best(run_ids: list[str], params: dict) -> None:
    """
    Pick the run with the highest roc_auc, register it in the MLflow Model
    Registry, then transition: None → Staging → Production.
    """
    client = MlflowClient()
    model_name = (
        params["mlflow"]["model_name"]
        if "model_name" in params["mlflow"]
        else params["training"]["model_name"]
    )
    staging_stage = params["mlflow"]["staging_stage"]
    production_stage = params["mlflow"]["production_stage"]

    # Find best run by roc_auc
    best_run_id, best_auc = None, -1.0
    for rid in run_ids:
        run_data = client.get_run(rid)
        auc = run_data.data.metrics.get("roc_auc", 0.0)
        if auc > best_auc:
            best_auc = auc
            best_run_id = rid

    log.info("Best run: %s with roc_auc=%.4f", best_run_id, best_auc)

    # Register model
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = mv.version
    log.info("Registered as '%s' version %s", model_name, version)

    # Transition: None → Staging
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=staging_stage,
        archive_existing_versions=False,
    )
    log.info("Version %s transitioned to %s", version, staging_stage)

    # Transition: Staging → Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=production_stage,
        archive_existing_versions=True,
    )
    log.info("Version %s transitioned to %s", version, production_stage)


def main() -> None:
    params = load_params()
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["training"]["experiment_name"])

    X_train, y_train, X_test, y_test = load_splits(params)
    log.info("Training on %d samples, evaluating on %d", len(X_train), len(X_test))

    run_ids = []
    run_ids.append(run_logistic_regression(X_train, y_train, X_test, y_test, params))
    run_ids.append(run_random_forest(X_train, y_train, X_test, y_test, params))
    run_ids.append(run_gradient_boosting(X_train, y_train, X_test, y_test, params))

    register_and_promote_best(run_ids, params)
    log.info("Training complete. Best model promoted to Production.")


if __name__ == "__main__":
    main()
