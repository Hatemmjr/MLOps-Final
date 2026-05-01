"""
Component 4 — API test script.
Uses httpx (ASGI test client) so no live server is needed.
Run with: pytest tests/test_api.py
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ─── Patch model loading so tests don't require a live MLflow server ─────────
import sys
import types

# Create a minimal mock model
class MockModel:
    def predict(self, X):
        return np.array([0] * len(X))

    def predict_proba(self, X):
        return np.array([[0.8, 0.2]] * len(X))


def _mock_load_model():
    from src.serving import app as app_module
    app_module._model_state["model"] = MockModel()
    app_module._model_state["model_name"] = "telco-churn-model"
    app_module._model_state["model_version"] = "Production"
    app_module._model_state["status"] = "ready"


# ─── Sample payload ───────────────────────────────────────────────────────────
SAMPLE_RECORD = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 1025.45,
}


@pytest.fixture(scope="module")
def client():
    from src.serving.app import app
    _mock_load_model()
    with TestClient(app) as c:
        yield c


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_health_endpoint_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_has_required_fields(client):
    response = client.get("/health")
    body = response.json()
    assert "status" in body
    assert "model_name" in body
    assert "model_version" in body


def test_health_status_is_ready(client):
    response = client.get("/health")
    assert response.json()["status"] == "ready"


def test_predict_returns_200(client):
    response = client.post("/predict", json=SAMPLE_RECORD)
    assert response.status_code == 200


def test_predict_response_schema(client):
    response = client.post("/predict", json=SAMPLE_RECORD)
    body = response.json()
    assert "prediction" in body
    assert "confidence" in body
    assert "label" in body


def test_predict_binary_output(client):
    response = client.post("/predict", json=SAMPLE_RECORD)
    pred = response.json()["prediction"]
    assert pred in (0, 1)


def test_predict_confidence_in_range(client):
    response = client.post("/predict", json=SAMPLE_RECORD)
    confidence = response.json()["confidence"]
    assert 0.0 <= confidence <= 1.0


def test_predict_invalid_senior_citizen(client):
    bad_record = {**SAMPLE_RECORD, "SeniorCitizen": 5}
    response = client.post("/predict", json=bad_record)
    assert response.status_code == 422


def test_predict_negative_tenure_rejected(client):
    bad_record = {**SAMPLE_RECORD, "tenure": -5}
    response = client.post("/predict", json=bad_record)
    assert response.status_code == 422


def test_predict_missing_field_rejected(client):
    incomplete = {k: v for k, v in SAMPLE_RECORD.items() if k != "MonthlyCharges"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_batch_returns_list(client):
    batch_payload = {"records": [SAMPLE_RECORD, SAMPLE_RECORD]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 2


def test_label_maps_to_prediction(client):
    response = client.post("/predict", json=SAMPLE_RECORD)
    body = response.json()
    if body["prediction"] == 1:
        assert body["label"] == "Churn"
    else:
        assert body["label"] == "No Churn"
