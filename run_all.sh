#!/usr/bin/env bash
set -e

echo "======================================"
echo "      MLOps End-to-End Pipeline       "
echo "======================================"

echo "1. Cleaning up any running MLFlow servers..."
lsof -ti:5000 | xargs kill -9 2>/dev/null || true

# Activate virtual environment
source venv/bin/activate

echo -e "\n2. Starting MLFlow tracking server in background..."
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000 &
MLFLOW_PID=$!
sleep 5

echo -e "\n3. Running DVC Data Pipeline (Prepare -> Preprocess -> Featurize)..."
dvc repro

echo -e "\n4. Running Model Training and HPO..."
python src/training/train.py

echo -e "\n5. Running Model Evaluation..."
python src/evaluation/evaluate.py

echo -e "\n6. Running Drift Monitoring and Prometheus Metrics Generation..."
python monitoring/run_monitoring.py

echo -e "\n7. Running Pytest Suite (Unit and API tests)..."
pytest tests/

echo -e "\n======================================"
echo "✅ Pipeline Executed Successfully!"
echo "======================================"

echo "Cleaning up MLflow background process..."
kill -9 $MLFLOW_PID 2>/dev/null || true
