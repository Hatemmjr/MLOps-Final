"""
Shared feature selection utilities.
Used by both src/training/train.py and src/evaluation/evaluate.py
to guarantee identical column sets at training and evaluation time.

Features identified as near-zero correlation with Churn via Chi-Squared test
(reference: Kaggle Telco Churn notebook EDA — Pearson |r| < 0.1).
"""

import logging

import pandas as pd

log = logging.getLogger(__name__)

LOW_SIGNAL_FEATURES = [
    "gender",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "StreamingTV",
    "StreamingMovies",
]


def drop_low_signal(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop one-hot encoded columns whose base feature name is in
    LOW_SIGNAL_FEATURES.  Works whether columns are raw ('gender')
    or one-hot prefixed ('cat__gender_Male').
    """
    drop_cols = [
        c for c in X_train.columns
        if any(feat in c for feat in LOW_SIGNAL_FEATURES)
    ]
    if drop_cols:
        log.info(
            "Dropping %d low-signal columns: %s", len(drop_cols), drop_cols
        )
    return (
        X_train.drop(columns=drop_cols),
        X_test.drop(columns=drop_cols),
    )


def drop_low_signal_single(X: pd.DataFrame) -> pd.DataFrame:
    """
    Single-DataFrame variant — used during evaluation / serving
    where there is no train/test pair.
    """
    drop_cols = [
        c for c in X.columns
        if any(feat in c for feat in LOW_SIGNAL_FEATURES)
    ]
    if drop_cols:
        log.info(
            "Dropping %d low-signal columns: %s", len(drop_cols), drop_cols
        )
    return X.drop(columns=drop_cols)
