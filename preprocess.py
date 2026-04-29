"""
Stage 2 — preprocess
Builds and fits a scikit-learn Pipeline (imputer + scaler + encoder + SMOTE)
on the cleaned data and serialises it as a DVC-tracked artifact.
All parameters come from configs/params.yaml.
"""

import logging
import pathlib

import joblib
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_params(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_preprocessor(params: dict) -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles numeric and categorical features
    separately.  No values are hardcoded — everything is read from params.yaml.
    """
    p = params["preprocessing"]
    numeric_features = p["numeric_features"]
    categorical_features = p["categorical_features"]
    imputer_strategy = p["imputer_strategy"]

    numeric_transformer = ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def build_full_pipeline(params: dict) -> ImbPipeline:
    """
    Full imbalanced-learn Pipeline: preprocessing → SMOTE.
    Note: SMOTE is only applied during fit (training); it does not affect
    transform-only calls used at serving time.
    """
    p = params["preprocessing"]
    preprocessor = build_preprocessor(params)
    smote = SMOTE(
        random_state=p["smote_random_state"],
        k_neighbors=p["smote_k_neighbors"],
    )
    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", smote),
        ]
    )
    return pipeline


def fit_and_save_pipeline(
    df: pd.DataFrame, params: dict, artifact_path: str
) -> ImbPipeline:
    """Fit the pipeline on cleaned data and persist it."""
    target = params["data"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    pipeline = build_full_pipeline(params)
    pipeline.fit(X, y)
    log.info("Pipeline fitted. Classes after SMOTE resampling logged.")

    pathlib.Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_path)
    log.info("Pipeline serialised to %s", artifact_path)
    return pipeline


def load_pipeline(artifact_path: str) -> ImbPipeline:
    """Load a previously fitted pipeline from disk."""
    return joblib.load(artifact_path)


def get_preprocessor_only(pipeline: ImbPipeline):
    """
    Extract just the ColumnTransformer step (no SMOTE) for use at serving time.
    Returns a fitted ColumnTransformer that can call .transform().
    """
    return pipeline.named_steps["preprocessor"]


def main() -> None:
    params = load_params()
    processed_path = params["data"]["processed_path"]
    artifact_path = params["preprocessing"]["pipeline_artifact"]

    df = pd.read_csv(processed_path)
    log.info("Loaded cleaned data from %s (%d rows)", processed_path, len(df))

    fit_and_save_pipeline(df, params, artifact_path)


if __name__ == "__main__":
    main()
