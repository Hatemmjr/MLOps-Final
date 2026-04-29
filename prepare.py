"""
Stage 1 — prepare
Downloads (or reads) raw Telco Churn data, applies basic cleaning,
and writes a single cleaned CSV to data/processed/.
All parameters come from configs/params.yaml.
"""

import logging
import pathlib

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_params(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw_data(raw_path: str) -> pd.DataFrame:
    """Load raw CSV. If the file doesn't exist, download a synthetic version."""
    p = pathlib.Path(raw_path)
    if not p.exists():
        log.warning(
            "Raw data not found at %s — generating synthetic dataset for CI.", raw_path
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        df = _generate_synthetic_telco()
        df.to_csv(p, index=False)
        log.info("Synthetic dataset written to %s (%d rows)", raw_path, len(df))
    else:
        df = pd.read_csv(p)
        log.info("Loaded raw data from %s (%d rows)", raw_path, len(df))
    return df


def _generate_synthetic_telco(n: int = 7043) -> pd.DataFrame:
    """Generate a synthetic Telco-like dataset for reproducibility in CI."""
    rng = np.random.default_rng(42)
    yes_no = ["Yes", "No"]
    df = pd.DataFrame(
        {
            "customerID": [f"CUST-{i:05d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": rng.choice(yes_no, n),
            "Dependents": rng.choice(yes_no, n, p=[0.7, 0.3]),
            "tenure": rng.integers(0, 73, n),
            "PhoneService": rng.choice(yes_no, n, p=[0.1, 0.9]),
            "MultipleLines": rng.choice(["No phone service", "No", "Yes"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(["No internet service", "No", "Yes"], n),
            "OnlineBackup": rng.choice(["No internet service", "No", "Yes"], n),
            "DeviceProtection": rng.choice(["No internet service", "No", "Yes"], n),
            "TechSupport": rng.choice(["No internet service", "No", "Yes"], n),
            "StreamingTV": rng.choice(["No internet service", "No", "Yes"], n),
            "StreamingMovies": rng.choice(["No internet service", "No", "Yes"], n),
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21]
            ),
            "PaperlessBilling": rng.choice(yes_no, n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": rng.uniform(18, 119, n).round(2),
            "TotalCharges": (rng.uniform(18, 8000, n)).round(2),
            "Churn": rng.choice(yes_no, n, p=[0.7349, 0.2651]),
        }
    )
    # Introduce a few missing values in TotalCharges (mirrors real dataset)
    mask = rng.choice([True, False], n, p=[0.0016, 0.9984])
    df.loc[mask, "TotalCharges"] = np.nan
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
