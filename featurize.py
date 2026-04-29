"""
Stage 3 — featurize
Applies the fitted preprocessing pipeline to produce final feature matrices,
then splits into train / test / reference / production sets.
All parameters come from configs/params.yaml.
"""

import logging
import pathlib

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.preprocess import load_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_params(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def create_reference_production_split(
    df: pd.DataFrame, reference_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal-style split: first `reference_ratio` fraction = reference set,
    remainder = production set.  Drift is injected later in monitoring.
    """
    cut = int(len(df) * reference_ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def inject_drift(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Simulate sensor / distribution drift on the production set by adding
    Gaussian noise scaled by `perturbation_std_multiplier` × feature std.
    """
    perturb_features = params["monitoring"]["perturbation_features"]
    multiplier = params["monitoring"]["perturbation_std_multiplier"]
    seed = params["data"]["random_seed"]
    rng = np.random.default_rng(seed)

    df_perturbed = df.copy()
    for col in perturb_features:
        if col in df_perturbed.columns:
            std = df_perturbed[col].std()
            noise = rng.normal(0, multiplier * std, size=len(df_perturbed))
            df_perturbed[col] = df_perturbed[col] + noise
    log.info("Drift injected on features: %s", perturb_features)
    return df_perturbed


def main() -> None:
    params = load_params()
    dp = params["data"]
    pp = params["preprocessing"]

    df = pd.read_csv(dp["processed_path"])
    target = dp["target_column"]
    seed = dp["random_seed"]

    # ── Train / Test split ──────────────────────────────────────────────────
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=dp["test_size"], random_state=seed, stratify=y
    )

    # Reload pipeline (fitted in preprocess stage) for transform-only use
    pipeline = load_pipeline(pp["pipeline_artifact"])
    preprocessor = pipeline.named_steps["preprocessor"]

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Reconstruct DataFrames with feature names
    feature_names = preprocessor.get_feature_names_out()
    train_df = pd.DataFrame(X_train_t, columns=feature_names)
    train_df[target] = y_train.values
    test_df = pd.DataFrame(X_test_t, columns=feature_names)
    test_df[target] = y_test.values

    # ── Reference / Production split (on raw df for drift monitoring) ───────
    reference_ratio = dp["reference_ratio"]
    ref_df, prod_df = create_reference_production_split(df, reference_ratio)
    prod_df = inject_drift(prod_df, params)

    # ── Persist splits ───────────────────────────────────────────────────────
    for path, data in [
        (dp["train_path"], train_df),
        (dp["test_path"], test_df),
        (dp["reference_path"], ref_df),
        (dp["production_path"], prod_df),
    ]:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        log.info("Written %d rows to %s", len(data), path)


if __name__ == "__main__":
    main()
