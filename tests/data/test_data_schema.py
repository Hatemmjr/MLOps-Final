"""
Data validation tests using pandera.
Ensures the cleaned data conforms to the expected schema and value ranges.
"""

import pathlib

import pandas as pd
import pandera as pa
import pytest
import yaml


def load_params(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Define a pandera schema to validate the cleaned data
schema = pa.DataFrameSchema(
    {
        "gender": pa.Column(str, pa.Check.isin(["Male", "Female"])),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
        "Partner": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "Dependents": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "tenure": pa.Column(float, pa.Check.ge(0), coerce=True),
        "PhoneService": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "MultipleLines": pa.Column(str, pa.Check.isin(["No phone service", "No", "Yes"])),
        "InternetService": pa.Column(str, pa.Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": pa.Column(str, pa.Check.isin(["No internet service", "No", "Yes"])),
        "OnlineBackup": pa.Column(str, pa.Check.isin(["No internet service", "No", "Yes"])),
        "DeviceProtection": pa.Column(str, pa.Check.isin(["No internet service", "No", "Yes"])),
        "TechSupport": pa.Column(str, pa.Check.isin(["No internet service", "No", "Yes"])),
        "StreamingTV": pa.Column(str, pa.Check.isin(["No internet service", "No", "Yes"])),
        "StreamingMovies": pa.Column(str, pa.Check.isin(["No internet service", "No", "Yes"])),
        "Contract": pa.Column(str, pa.Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "PaymentMethod": pa.Column(
            str,
            pa.Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        ),
        "MonthlyCharges": pa.Column(float, pa.Check.ge(0), coerce=True),
        "TotalCharges": pa.Column(float, pa.Check.ge(0), coerce=True, nullable=True),
        "Churn": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
    }
)


def test_processed_data_schema():
    """Test that the processed CSV matches the expected Pandera schema."""
    params = load_params()
    processed_path = params["data"]["processed_path"]

    if not pathlib.Path(processed_path).exists():
        pytest.skip(f"Processed data not found at {processed_path}")

    df = pd.read_csv(processed_path)
    schema.validate(df)
