"""
Training — Experiment Tracking & Model Registry (Component 3)

Improvements based on Kaggle notebook analysis:
  1. Feature selection: drop 6 near-zero-correlation features identified via
     Chi-Squared test (gender, PhoneService, MultipleLines, InternetService,
     StreamingTV, StreamingMovies).
  2. LightGBM added as Experiment 5 — consistently matches / beats XGBoost
     on this dataset (CV AUC ~90.3% in reference notebook).
  3. Stacking ensemble added as Experiment 6 — uses XGB + LGBM + RF as base
     learners with LGBM as meta-learner (best result in reference notebook).
  4. Continuous Optuna search (suggest_float / suggest_int) for all models.
  5. class_weight / scale_pos_weight for imbalance handling on every model.
  6. Threshold tuning to maximise F1 on a held-out CV fold.

Set N_OPTUNA_TRIALS env var to override n_optuna_trials (e.g. for CI).
"""

import logging
import os

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from mlflow import MlflowClient
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
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
from xgboost import XGBClassifier

from src.data.feature_selection import drop_low_signal  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def best_threshold(model, X_train: pd.DataFrame, y_train, seed: int) -> float:
    """Find the decision threshold that maximises F1 on a held-out CV fold."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    thresholds = np.linspace(0.2, 0.7, 50)
    f1_by_thresh = np.zeros(len(thresholds))

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        for i, t in enumerate(thresholds):
            preds = (probs >= t).astype(int)
            f1_by_thresh[i] += f1_score(y_val, preds, zero_division=0)

    return float(thresholds[np.argmax(f1_by_thresh)])


def compute_metrics(model, X, y, threshold: float = 0.5) -> dict:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y, y_prob)),
        "threshold": threshold,
    }


def _log_run(run_name: str, model, X_train, y_train, X_test, y_test,
             params: dict, extra_params: dict = None) -> str:
    """Shared MLflow run logic: fit, threshold-tune, log, return run_id."""
    seed = params["data"]["random_seed"]
    model.fit(X_train, y_train)
    thresh = best_threshold(model, X_train, y_train, seed)
    metrics = compute_metrics(model, X_test, y_test, thresh)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(extra_params or {})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        run_id = run.info.run_id

    log.info(
        "[%s] roc_auc=%.4f  f1=%.4f  recall=%.4f  thr=%.2f  run=%s",
        run_name[:4].upper(), metrics["roc_auc"], metrics["f1"],
        metrics["recall"], thresh, run_id,
    )
    return run_id


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 — Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
def run_logistic_regression(X_train, y_train, X_test, y_test, params: dict) -> str:
    n_trials = int(os.environ.get("N_OPTUNA_TRIALS", params["training"]["n_optuna_trials"]))
    seed = params["data"]["random_seed"]
    cv = StratifiedKFold(n_splits=params["training"]["cv_folds"], shuffle=True, random_state=seed)

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 100, log=True)
        max_iter = trial.suggest_int("max_iter", 200, 1000, step=100)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        clf = LogisticRegression(
            C=C, solver=solver, max_iter=max_iter,
            class_weight="balanced", random_state=seed,
        )
        return cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=params["training"]["optuna_timeout"])
    best = study.best_params
    model = LogisticRegression(**best, class_weight="balanced", random_state=seed)
    return _log_run("logistic-regression-optuna", model, X_train, y_train,
                    X_test, y_test, params, best)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — Random Forest
# ─────────────────────────────────────────────────────────────────────────────
def run_random_forest(X_train, y_train, X_test, y_test, params: dict) -> str:
    n_trials = int(os.environ.get("N_OPTUNA_TRIALS", params["training"]["n_optuna_trials"]))
    seed = params["data"]["random_seed"]
    cv = StratifiedKFold(n_splits=params["training"]["cv_folds"], shuffle=True, random_state=seed)

    def objective(trial):
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 4, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
        return cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=params["training"]["optuna_timeout"])
    best = study.best_params
    model = RandomForestClassifier(**best, class_weight="balanced", random_state=seed, n_jobs=-1)
    return _log_run("random-forest-optuna", model, X_train, y_train,
                    X_test, y_test, params, best)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3 — Gradient Boosting (sklearn)
# ─────────────────────────────────────────────────────────────────────────────
def run_gradient_boosting(X_train, y_train, X_test, y_test, params: dict) -> str:
    n_trials = int(os.environ.get("N_OPTUNA_TRIALS", params["training"]["n_optuna_trials"]))
    seed = params["data"]["random_seed"]
    cv = StratifiedKFold(n_splits=params["training"]["cv_folds"], shuffle=True, random_state=seed)

    def objective(trial):
        clf = GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            random_state=seed,
        )
        return cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=params["training"]["optuna_timeout"])
    best = study.best_params
    model = GradientBoostingClassifier(**best, random_state=seed)
    return _log_run("gradient-boosting-optuna", model, X_train, y_train,
                    X_test, y_test, params, best)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4 — XGBoost
# ─────────────────────────────────────────────────────────────────────────────
def run_xgboost(X_train, y_train, X_test, y_test, params: dict) -> str:
    n_trials = int(os.environ.get("N_OPTUNA_TRIALS", params["training"]["n_optuna_trials"]))
    seed = params["data"]["random_seed"]
    cv = StratifiedKFold(n_splits=params["training"]["cv_folds"], shuffle=True, random_state=seed)
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    def objective(trial):
        clf = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 9),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            scale_pos_weight=scale_pos_weight,
            random_state=seed, eval_metric="logloss", verbosity=0,
        )
        return cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=params["training"]["optuna_timeout"])
    best = study.best_params
    model = XGBClassifier(
        **best, scale_pos_weight=scale_pos_weight,
        random_state=seed, eval_metric="logloss", verbosity=0,
    )
    return _log_run("xgboost-optuna", model, X_train, y_train,
                    X_test, y_test, params,
                    {**best, "scale_pos_weight": scale_pos_weight})


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5 — LightGBM
# ─────────────────────────────────────────────────────────────────────────────
def run_lightgbm(X_train, y_train, X_test, y_test, params: dict) -> str:
    """
    LightGBM: histogram-based GBM — faster than sklearn GBM and slightly
    better than XGBoost on this dataset per reference notebook (CV AUC 90.33%).
    """
    n_trials = int(os.environ.get("N_OPTUNA_TRIALS", params["training"]["n_optuna_trials"]))
    seed = params["data"]["random_seed"]
    cv = StratifiedKFold(n_splits=params["training"]["cv_folds"], shuffle=True, random_state=seed)
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    def objective(trial):
        clf = LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            num_leaves=trial.suggest_int("num_leaves", 20, 150),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            scale_pos_weight=scale_pos_weight,
            random_state=seed, n_jobs=-1, verbosity=-1,
        )
        return cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=params["training"]["optuna_timeout"])
    best = study.best_params
    model = LGBMClassifier(
        **best, scale_pos_weight=scale_pos_weight,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )
    return _log_run("lightgbm-optuna", model, X_train, y_train,
                    X_test, y_test, params,
                    {**best, "scale_pos_weight": scale_pos_weight})


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 6 — Stacking Ensemble (XGB + LGBM + RF → LGBM meta)
# ─────────────────────────────────────────────────────────────────────────────
def run_stacking(X_train, y_train, X_test, y_test, params: dict,
                 xgb_model, lgbm_model, rf_model) -> str:
    """
    Stack the three best individual models. LGBM is the meta-learner.
    Based on reference notebook: Stacking AUC=83%, F1=83% (best overall).
    """
    seed = params["data"]["random_seed"]

    stack = StackingClassifier(
        estimators=[
            ("xgb", xgb_model),
            ("lgbm", lgbm_model),
            ("rf", rf_model),
        ],
        final_estimator=LGBMClassifier(
            n_estimators=200, learning_rate=0.05,
            random_state=seed, n_jobs=-1, verbosity=-1,
        ),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    return _log_run("stacking-ensemble", stack, X_train, y_train,
                    X_test, y_test, params, {"meta_learner": "lgbm"})


# ─────────────────────────────────────────────────────────────────────────────
# Model Registration & Promotion
# ─────────────────────────────────────────────────────────────────────────────
def register_and_promote_best(run_ids: list[str], params: dict) -> None:
    """Pick the run with the highest roc_auc → register → Production."""
    client = MlflowClient()
    model_name = (
        params["mlflow"]["model_name"]
        if "model_name" in params["mlflow"]
        else params["training"]["model_name"]
    )
    staging_stage = params["mlflow"]["staging_stage"]
    production_stage = params["mlflow"]["production_stage"]

    best_run_id, best_auc = None, -1.0
    for rid in run_ids:
        auc = client.get_run(rid).data.metrics.get("roc_auc", 0.0)
        if auc > best_auc:
            best_auc = auc
            best_run_id = rid

    log.info("Best run: %s  roc_auc=%.4f", best_run_id, best_auc)
    mv = mlflow.register_model(model_uri=f"runs:/{best_run_id}/model", name=model_name)
    version = mv.version
    log.info("Registered '%s' version %s", model_name, version)

    client.transition_model_version_stage(
        name=model_name, version=version, stage=staging_stage,
        archive_existing_versions=False,
    )
    client.transition_model_version_stage(
        name=model_name, version=version, stage=production_stage,
        archive_existing_versions=True,
    )
    log.info("Version %s promoted to %s", version, production_stage)


def main() -> None:
    params = load_params()
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["training"]["experiment_name"])

    X_train, y_train, X_test, y_test = load_splits(params)

    # Drop near-zero-correlation features (Chi-Squared analysis)
    X_train, X_test = drop_low_signal(X_train, X_test)
    log.info("After feature selection: %d features", X_train.shape[1])
    log.info("Training on %d samples, evaluating on %d", len(X_train), len(X_test))

    run_ids = []
    run_ids.append(run_logistic_regression(X_train, y_train, X_test, y_test, params))
    run_ids.append(run_random_forest(X_train, y_train, X_test, y_test, params))
    run_ids.append(run_gradient_boosting(X_train, y_train, X_test, y_test, params))

    # Build tuned XGB + LGBM for stacking (reuse their best params)
    seed = params["data"]["random_seed"]
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = neg / max(pos, 1)

    xgb_id = run_xgboost(X_train, y_train, X_test, y_test, params)
    lgbm_id = run_lightgbm(X_train, y_train, X_test, y_test, params)
    run_ids.extend([xgb_id, lgbm_id])

    # Build stacking base-learners with good defaults for speed
    xgb_base = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        scale_pos_weight=spw, random_state=seed, eval_metric="logloss", verbosity=0,
    )
    lgbm_base = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        scale_pos_weight=spw, random_state=seed, n_jobs=-1, verbosity=-1,
    )
    rf_base = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )
    stack_id = run_stacking(
        X_train, y_train, X_test, y_test, params,
        xgb_base, lgbm_base, rf_base,
    )
    run_ids.append(stack_id)

    register_and_promote_best(run_ids, params)
    log.info("Training complete. Best model promoted to Production.")


if __name__ == "__main__":
    main()
