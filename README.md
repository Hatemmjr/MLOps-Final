# Telco Churn MLOps Pipeline
### DDSC611 – Machine Learning Engineering Practices · Spring 2026
**ESLSCA University** | Team Project | 40% of Final Grade

[![CI Pipeline](https://github.com/Hatemmjr/MLOps-Final/actions/workflows/ci.yml/badge.svg)](https://github.com/Hatemmjr/MLOps-Final/actions)

---

## Quickstart 

```bash
pip install -r requirements.txt
dvc repro
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

The API is live at **http://localhost:8000/docs**

---

## Full Setup Guide

### 1. Clone & Install

```bash
git clone https://github.com/Hatemmjr/MLOps-Final.git
cd MLOps-Final
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=.
```

### 2. Set Up DVC

```bash
dvc init
dvc remote add -d myremote /tmp/dvc-remote
dvc repro
```

This generates all data artifacts automatically (synthetic Telco dataset included).

### 3. Train Models

```bash
python -m src.training.train
```

Runs 3 experiments (Logistic Regression, Random Forest, Gradient Boosting) with Optuna HPO.
Best model is automatically promoted to **Production** in the MLflow registry.

### 4. View MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open **http://localhost:5000** → Experiments → `telco-churn-experiment`

### 5. Start the API

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

**Health check:**
```bash
curl http://localhost:8000/health
```

**Make a prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender":"Male","SeniorCitizen":0,"Partner":"Yes",
    "Dependents":"No","tenure":12,"PhoneService":"Yes",
    "MultipleLines":"No","InternetService":"Fiber optic",
    "OnlineSecurity":"No","OnlineBackup":"No",
    "DeviceProtection":"No","TechSupport":"No",
    "StreamingTV":"Yes","StreamingMovies":"Yes",
    "Contract":"Month-to-month","PaperlessBilling":"Yes",
    "PaymentMethod":"Electronic check",
    "MonthlyCharges":85.50,"TotalCharges":1025.45
  }'
```

### 6. Run Tests

```bash
pytest tests/ --cov=src --cov-report=term-missing -v
```

Coverage

### 7. Generate Monitoring Reports

```bash
python monitoring/run_monitoring.py
open monitoring/evidently_reports/baseline_report.html
open monitoring/evidently_reports/drift_report.html
```

**Drift simulation:** Gaussian noise (σ = 2× feature std) is injected into
`MonthlyCharges`, `tenure`, and `TotalCharges` in the production set to simulate
distribution shift. If >20% of features drift, a retrain alert is triggered and
logged to `monitoring/drift.log`.

### 8. Export Experiment Log

```bash
python docs/experiment_log_generator.py
# Output: docs/experiment_log.csv
```

### 9. Docker 

```bash
docker compose up --build
# API:        http://localhost:8000
# MLflow UI:  http://localhost:5000
# Prometheus: http://localhost:9090
```

---

## Architecture

```
Raw Data (DVC)
     │
     ▼
prepare → preprocess → featurize → train
                                      │
                          MLflow Registry (Production)
                                      │
                              FastAPI /predict
                                      │
                         Evidently Monitoring Reports
```

## 📁 Repository Structure

```
├── .github/workflows/ci.yml     # CI: lint → test → data-val → model-val
├── configs/params.yaml          # ALL parameters — zero hardcoded values
├── data/                        # DVC-tracked (not committed to Git)
├── src/
│   ├── data/prepare.py          # Stage 1: clean raw data
│   ├── data/preprocess.py       # Stage 2: sklearn Pipeline + SMOTE
│   ├── features/featurize.py    # Stage 3: splits + drift injection
│   ├── training/train.py        # MLflow + Optuna HPO (3 experiments)
│   ├── evaluation/evaluate.py   # Model validation gate
│   └── serving/app.py           # FastAPI /health /predict /predict/batch
├── monitoring/run_monitoring.py # Evidently reports + Prometheus metrics
├── tests/                       # Unit + API + schema tests
├── docs/model_card.md           # Model Card
├── docs/data_card.md            # Data Card
├── docs/experiment_log.csv      # MLflow runs export
├── dvc.yaml                     # Pipeline: prepare→preprocess→featurize→train
├── Dockerfile                   # Bonus A: serving container
├── docker-compose.yml           # Bonus A: serving + mlflow + prometheus
└── dags/training_pipeline.py    # Bonus B: Airflow 5-task DAG
```

---

## Key Commands Reference

| Task | Command |
|------|---------|
| Run DVC pipeline | `dvc repro` |
| View pipeline DAG | `dvc dag` |
| Start MLflow UI | `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000` |
| Train & register | `python -m src.training.train` |
| Validate model | `python -m src.evaluation.evaluate` |
| Start API | `uvicorn src.serving.app:app --port 8000` |
| Run monitoring | `python monitoring/run_monitoring.py` |
| Run all tests | `pytest tests/ --cov=src` |
| Export experiment log | `python docs/experiment_log_generator.py` |
| Docker stack | `docker compose up --build` |

---

## Team

| Name | Student ID |
|------|-----------|
| TBD | TBD |
| TBD | TBD |
| TBD | TBD |

---

## Dataset

**IBM Telco Customer Churn** — 7,043 rows × 20 features  
Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
License: IBM Community Data License Agreement (educational use)

---

## Demo Video & Report

- **Demo Video:** [Link TBD]
- **Technical Report:** [Link TBD]

---

© 2026 ESLSCA University · DDSC611 · Mohamed Tharwat, PhD, SM IEEE
