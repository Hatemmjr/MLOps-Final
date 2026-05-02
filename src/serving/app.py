"""
Component 4 — Model Serving (FastAPI)
Loads the Production model from MLflow Registry at startup.
Exposes GET /health and POST /predict (+ POST /predict/batch).
"""

import logging
import pathlib
from contextlib import asynccontextmanager
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from pydantic import BaseModel, validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def _load_params(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


PARAMS = _load_params()
SERVING_CFG = PARAMS["serving"]
MLFLOW_CFG = PARAMS["mlflow"]

# ─────────────────────────────────────────────────────────────────────────────
# Prometheus metrics (Component 6 — defined here, used in /predict)
# ─────────────────────────────────────────────────────────────────────────────
CONFIDENCE_HISTOGRAM = Histogram(
    "prediction_confidence",
    "Histogram of model prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
MONTHLY_CHARGES_HISTOGRAM = Histogram(
    "feature_monthly_charges",
    "Distribution of MonthlyCharges in incoming requests",
    buckets=[20, 40, 60, 80, 100, 120],
)
TENURE_HISTOGRAM = Histogram(
    "feature_tenure",
    "Distribution of tenure (months) in incoming requests",
    buckets=[0, 12, 24, 36, 48, 60, 72],
)
MODEL_VERSION_GAUGE = Gauge("model_version", "Current model version in production")
INFERENCE_COUNTER = Counter(
    "inference_count",
    "Number of inference requests by predicted class",
    ["predicted_class"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Global model state
# ─────────────────────────────────────────────────────────────────────────────
_model_state: dict[str, Any] = {
    "model": None,
    "preprocessor": None,   # sklearn ColumnTransformer (fitted)
    "model_name": None,
    "model_version": None,
    "status": "not_loaded",
}


def _load_preprocessor() -> None:
    """Load the fitted sklearn ColumnTransformer from disk."""
    pipeline_path = pathlib.Path(PARAMS["preprocessing"]["pipeline_artifact"])
    if pipeline_path.exists():
        pipe = joblib.load(pipeline_path)
        # The pipeline is imblearn Pipeline: [('preprocessor', CT), ('smote', SMOTE)]
        # At inference time we only need the ColumnTransformer, not SMOTE
        if hasattr(pipe, "named_steps") and "preprocessor" in pipe.named_steps:
            _model_state["preprocessor"] = pipe.named_steps["preprocessor"]
        elif hasattr(pipe, "transform"):  # already a plain ColumnTransformer
            _model_state["preprocessor"] = pipe
        log.info("Preprocessing pipeline loaded from %s", pipeline_path)
    else:
        log.warning(
            "Preprocessing pipeline not found at %s — "
            "raw features will be passed directly.",
            pipeline_path,
        )


def _load_model() -> None:
    _load_preprocessor()
    mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])
    model_name = SERVING_CFG["model_name"]
    stage = SERVING_CFG["model_stage"]
    model_uri = f"models:/{model_name}/{stage}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        _model_state["model"] = model
        _model_state["model_name"] = model_name
        _model_state["model_version"] = stage
        _model_state["status"] = "ready"

        log.info("Model '%s' (%s) loaded successfully.", model_name, stage)

        # Try to fetch numeric version for gauge
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        if versions:
            MODEL_VERSION_GAUGE.set(int(versions[0].version))
    except Exception as e:
        log.warning("MLflow model load failed (%s). Falling back to local artifact.", e)
        _load_local_fallback()


def _load_local_fallback() -> None:
    """Fallback: load the latest joblib artifact from disk for CI environments."""
    # Search both mlartifacts/ and mlruns/ for model.pkl
    search_dirs = [pathlib.Path("mlartifacts"), pathlib.Path("mlruns")]
    artifacts = []
    for d in search_dirs:
        if d.exists():
            artifacts.extend(d.rglob("model.pkl"))
    if not artifacts:
        log.error("No local model artifact found. /predict will return 503.")
        _model_state["status"] = "unavailable"
        return
    chosen = sorted(artifacts)[-1]
    model = joblib.load(chosen)
    _model_state["model"] = model
    _model_state["model_name"] = "local-fallback"
    _model_state["model_version"] = "local"
    _model_state["status"] = "ready"
    log.info("Loaded local fallback model from %s", chosen)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────
class ChurnRecord(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    @validator("tenure", "MonthlyCharges", "TotalCharges")
    @classmethod
    def must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("Feature must be non-negative")
        return v

    @validator("SeniorCitizen")
    @classmethod
    def senior_binary(cls, v):
        if v not in (0, 1):
            raise ValueError("SeniorCitizen must be 0 or 1")
        return v


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    label: str


class BatchRequest(BaseModel):
    records: list[ChurnRecord]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: model loading on startup
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting model server …")
    _load_model()
    # Start Prometheus metrics server on a separate port
    try:
        prom_port = PARAMS["monitoring"]["prometheus_port"]
        start_http_server(prom_port)
        log.info("Prometheus metrics available on port %d", prom_port)
    except Exception as e:
        log.warning("Could not start Prometheus server: %s", e)
    yield
    log.info("Shutting down model server.")


app = FastAPI(
    title="Telco Churn Prediction API",
    description="MLOps Final Project — DDSC611",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": _model_state["status"],
        "model_name": _model_state["model_name"],
        "model_version": _model_state["model_version"],
    }


def _predict_single(record: ChurnRecord) -> PredictionResponse:
    if _model_state["model"] is None or _model_state["status"] != "ready":
        raise HTTPException(status_code=503, detail="Model not available")

    df = pd.DataFrame([record.dict()])

    # Apply preprocessing (ColumnTransformer: scale numerics + one-hot encode categoricals)
    preprocessor = _model_state["preprocessor"]
    if preprocessor is not None:
        try:
            df = pd.DataFrame(
                preprocessor.transform(df),
                columns=preprocessor.get_feature_names_out(),
            )
        except Exception as e:
            log.error("Preprocessing failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")

    model = _model_state["model"]
    pred = int(model.predict(df)[0])
    confidence = float(model.predict_proba(df)[0][pred])

    # Update Prometheus metrics
    CONFIDENCE_HISTOGRAM.observe(confidence)
    MONTHLY_CHARGES_HISTOGRAM.observe(record.MonthlyCharges)
    TENURE_HISTOGRAM.observe(record.tenure)
    INFERENCE_COUNTER.labels(predicted_class=str(pred)).inc()

    label = "Churn" if pred == 1 else "No Churn"
    return PredictionResponse(prediction=pred, confidence=confidence, label=label)


@app.post("/predict", response_model=PredictionResponse)
def predict(record: ChurnRecord):
    return _predict_single(record)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    return BatchResponse(predictions=[_predict_single(r) for r in batch.records])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.serving.app:app",
        host=SERVING_CFG["host"],
        port=SERVING_CFG["port"],
        reload=SERVING_CFG["reload"],
    )
