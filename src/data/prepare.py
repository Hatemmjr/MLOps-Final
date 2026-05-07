"""
Stage 1 — prepare
Reads real Telco Churn CSV data, applies basic cleaning,
and writes a single cleaned CSV to data/processed/.
All parameters come from configs/params.yaml.
"""

import logging
import pathlib

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_params(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw_data(raw_path: str) -> pd.DataFrame:
    """Load real raw CSV. Raises FileNotFoundError if the dataset is missing."""
    p = pathlib.Path(raw_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Real dataset not found at '{raw_path}'.\n"
            "Place the IBM Telco Customer Churn CSV at that path before running.\n"
            "Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
        )
    df = pd.read_csv(p)
    log.info("Loaded raw data from %s (%d rows)", raw_path, len(df))
    return df


def clean_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply lightweight cleaning steps."""
    drop_cols = params["data"].get("drop_columns", [])
    target = params["data"]["target_column"]

    # Drop identifier columns
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # TotalCharges is sometimes read as object due to whitespace
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Encode binary target
    if df[target].dtype == object:
        df[target] = df[target].map({"Yes": 1, "No": 0})

    log.info(
        "After cleaning: %d rows, %d cols, %d missing values",
        len(df),
        len(df.columns),
        df.isnull().sum().sum(),
    )
    return df


def main() -> None:
    params = load_params()
    raw_path = params["data"]["raw_path"]
    out_path = params["data"]["processed_path"]

    df = load_raw_data(raw_path)
    df = clean_data(df, params)

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Cleaned data written to %s", out_path)


if __name__ == "__main__":
    main()
