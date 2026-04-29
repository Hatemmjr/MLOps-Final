# Telco Churn MLOps Pipeline
### DDSC611 – Machine Learning Engineering Practices · Spring 2026
**ESLSCA University** | Team Project | 40% of Final Grade

[![CI Pipeline](https://github.com/Hatemmjr/MLOps-Final/actions/workflows/ci.yml/badge.svg)](https://github.com/Hatemmjr/MLOps-Final/actions)

---

## ⚡ Quickstart (3 commands)

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Run the full DVC pipeline (prepare → preprocess → featurize)
dvc repro

# 3. Start the serving application
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

The API is now live at **http://localhost:8000**. Test it:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Male","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
       "tenure":12,"PhoneService":"Yes","MultipleLines":"No",
       "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
       "DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes",
       "StreamingMovies":"Yes","Contract":"Month-to-month",
       "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
       "MonthlyCharges":85.50,"TotalCharges":1025.45}'
```

---

## 🏗 Architecture Overview

```
Raw Data (DVC)
     │
     ▼
┌─────────────┐    ┌──────────────────┐    ┌───────────────────┐
│   prepare   │───▶│   preprocess     │───▶│    featurize      │
│ (cleaning)  │    │ (sklearn pipeline│    │ (train/test split)│
│             │    │  + SMOTE)        │    │                   │
└─────────────┘    └──────────────────┘    └───────────────────┘
                                                    │
                                                    ▼
                              ┌─────────────────────────────────┐
                              │    Training (3 experiments)      │
                              │  LR · Random Forest · GBM        │
                              │  HPO via Optuna                  │
                              │  MLflow Tracking + Registry      │
                              └──────────────┬──────────────────┘
                                             │ Best model → Production
                                             ▼
                              ┌──────────────────────────────────┐
                              │  FastAPI Serving App              │
                              │  GET /health  POST /predict       │
                              │  POST /predict/batch             │
                              │  Prometheus metrics on :8001      │
                              └──────────────┬───────────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────────┐
                              │  Monitoring (Evidently + Prom)   │
                              │  Baseline report · Drift report  │
                              │  Drift threshold → retrain loop  │
                              └──────────────────────────────────┘
```

---

## 📁 Repository Structure

```
project-root/
├── .github/workflows/ci.yml        # CI/CD (lint → test → validate → model gate)
├── configs/params.yaml             # ALL pipeline parameters (zero hardcoded values)
├── data/
│   ├── raw/                        # DVC-tracked (not committed to Git)
│   ├── processed/                  # DVC-tracked
│   └── splits/                     # DVC-tracked (train / test / ref / prod)
├── src/
│   ├── data/
│   │   ├── prepare.py              # Stage 1: clean raw data
│   │   └── preprocess.py           # Stage 2: sklearn pipeline + SMOTE
│   ├── features/featurize.py       # Stage 3: split + drift injection
│   ├── training/train.py           # MLflow + Optuna HPO (3 experiments)
│   ├── evaluation/evaluate.py      # Model validation gate
│   └── serving/app.py              # FastAPI application
├── monitoring/
│   ├── run_monitoring.py           # Evidently reports + Prometheus metrics
│   ├── evidently_reports/          # baseline_report.html, drift_report.html
│   └── prometheus/
│       ├── prometheus.yml          # Scrape config
│       └── metrics.prom            # Written by monitoring script
├── dags/training_pipeline.py       # Airflow DAG (Bonus B)
├── tests/
│   ├── unit/test_preprocessing.py  # 6 unit tests
│   ├── data/test_schema.py         # pandera schema tests
│   └── test_api.py                 # FastAPI endpoint tests
├── docs/
│   ├── model_card.md
│   └── data_card.md
├── dvc.yaml                        # DVC pipeline (prepare → preprocess → featurize)
├── dvc.lock                        # Reproducibility lock (auto-generated)
├── Dockerfile                      # Serving app container (Bonus A)
├── docker-compose.yml              # serving + mlflow + prometheus (Bonus A)
└── requirements.txt                # Pinned dependencies
```

---

## 🛠 Full Setup Guide

### Prerequisites
- Python 3.11+
- Git
- (Optional) Docker & Docker Compose for Bonus A

### Step-by-step

```bash
# Clone the repo
git clone https://github.com/Hatemmjr/MLOps-Final.git
cd MLOps-Final

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt

# Initialise DVC (first time only)
dvc init
dvc remote add -d myremote /tmp/dvc-remote

# Run the full DVC pipeline
dvc repro

# Start the MLflow tracking server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --port 5000 &

# Train all experiments and register the best model
python src/training/train.py

# Start the API
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

### Docker (Bonus A)

```bash
docker compose up --build
# API: http://localhost:8000
# MLflow UI: http://localhost:5000
# Prometheus: http://localhost:9090
```

---

## 🧪 Running Tests

```bash
# All tests with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Unit tests only
pytest tests/unit/ -v

# API tests only
pytest tests/test_api.py -v
```

---

## 📊 Running the Monitoring Script

```bash
# Generates baseline_report.html and drift_report.html in monitoring/evidently_reports/
python monitoring/run_monitoring.py
```

---

## 📋 Key Commands Reference

| Task | Command |
|------|---------|
| Run DVC pipeline | `dvc repro` |
| View DVC DAG | `dvc dag` |
| Start MLflow UI | `mlflow server --port 5000` |
| Train & register | `python src/training/train.py` |
| Validate model | `python src/evaluation/evaluate.py` |
| Start API | `uvicorn src.serving.app:app --port 8000` |
| Run monitoring | `python monitoring/run_monitoring.py` |
| Run all tests | `pytest tests/ --cov=src` |
| Docker stack | `docker compose up --build` |

---

## 👥 Team

| Name | Student ID | Primary Focus |
|------|-----------|---------------|
| TBD | TBD | Data Pipeline + DVC |
| TBD | TBD | Training + MLflow |
| TBD | TBD | Serving + Monitoring |

*All members contributed to and understand the entire pipeline end-to-end.*

---

## 📜 License & Academic Integrity

This project was developed for DDSC611 at ESLSCA University (Spring 2026).  
AI coding assistants were used as productivity aids; all team members fully understand every component.  
External references are cited in the technical report.
