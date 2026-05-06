#!/usr/bin/env bash
# =====================================================================
#  MLOps Pipeline — Full Demonstration Runner
#  Runs every component in sequence and explains what each does.
# =====================================================================
set -e
PYTHON="venv/bin/python"
PYTEST="venv/bin/pytest"
MLFLOW="venv/bin/mlflow"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         MLOps Final Project — Full Pipeline Demo            ║"
echo "║         Telco Customer Churn Prediction                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Kill any existing MLflow server on port 5000 ─────────────────────
echo "► Cleaning up any old MLflow server on port 5000..."
lsof -ti:5000 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# ── Step 1: Start MLflow Tracking Server ─────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1 — MLflow Tracking Server"
echo "  WHY: MLflow records all experiments, parameters, metrics, and"
echo "       model artifacts. The UI lives at http://127.0.0.1:5000"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$MLFLOW server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000 &
MLFLOW_PID=$!
echo "  MLflow PID: $MLFLOW_PID — waiting 5s for it to start..."
sleep 5
echo "  ✅ MLflow server is running at http://127.0.0.1:5000"

# ── Step 2: Data Preparation ──────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2 — Data Preparation  (src/data/prepare.py)"
echo "  WHY: Loads raw Telco CSV, cleans it (encodes Churn 0/1,"
echo "       coerces TotalCharges to float, handles nulls),"
echo "       and writes data/processed/telco_churn_clean.csv"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON src/data/prepare.py
echo "  ✅ Cleaned CSV written to data/processed/telco_churn_clean.csv"

# ── Step 3: Preprocessing Pipeline ───────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3 — Preprocessing Pipeline  (src/data/preprocess.py)"
echo "  WHY: Builds and fits a sklearn ColumnTransformer:"
echo "       - StandardScaler on numeric features (tenure, charges)"
echo "       - OneHotEncoder on 15 categorical features"
echo "       - SMOTE to balance the imbalanced churn classes"
echo "       Saves as data/processed/preprocessing_pipeline.joblib"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON src/data/preprocess.py
echo "  ✅ Pipeline artifact saved to data/processed/preprocessing_pipeline.joblib"

# ── Step 4: Featurization ─────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 4 — Featurization  (src/features/featurize.py)"
echo "  WHY: Applies the fitted pipeline to produce 4 stratified splits:"
echo "       - train.csv   (model training)"
echo "       - test.csv    (model evaluation)"
echo "       - reference.csv  (monitoring baseline)"
echo "       - production.csv (simulated live traffic for drift detection)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON src/features/featurize.py
echo "  ✅ Data splits written to data/splits/"

# ── Step 5: Model Training + HPO ─────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 5 — Model Training + HPO  (src/training/train.py)"
echo "  WHY: Runs 3 experiments via Optuna (Logistic Regression,"
echo "       Random Forest, Gradient Boosting), logs params + metrics"
echo "       to MLflow, picks the best ROC-AUC model, and promotes it"
echo "       through Staging → Production in the MLflow Model Registry."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON src/training/train.py
echo "  ✅ Best model registered and promoted to Production"

# ── Step 6: Model Evaluation ─────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 6 — Model Evaluation  (src/evaluation/evaluate.py)"
echo "  WHY: Loads the Production model from MLflow, evaluates it on"
echo "       the held-out test set, and enforces CI gates:"
echo "       accuracy >= 0.50 and roc_auc >= 0.50"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON src/evaluation/evaluate.py
echo "  ✅ Model passed all performance thresholds"

# ── Step 7: Monitoring + Drift Detection ─────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 7 — Monitoring & Drift Detection  (monitoring/run_monitoring.py)"
echo "  WHY: Uses Evidently to compare reference vs production data,"
echo "       generates HTML reports, checks >20% drift threshold,"
echo "       and writes 5 custom Prometheus metrics to a .prom file."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON monitoring/run_monitoring.py
echo "  ✅ Evidently reports saved to monitoring/evidently_reports/"

# ── Step 8: Test Suite ────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 8 — Pytest Test Suite  (tests/)"
echo "  WHY: Validates every component with:"
echo "       - Unit tests: preprocessing transformations (6 tests)"
echo "       - Data tests: pandera schema validation"
echo "       - API tests: FastAPI endpoints with mock model (11 tests)"
echo "       - Integration: end-to-end pipeline test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTEST tests/ --cov=src --cov-report=term-missing --cov-fail-under=70 -v 2>&1
echo ""

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              ✅ PIPELINE COMPLETE                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  MLflow UI  →  http://127.0.0.1:5000                        ║"
echo "║  Reports    →  monitoring/evidently_reports/                 ║"
echo "║  Metrics    →  monitoring/prometheus/metrics.prom            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Keeping MLflow server running. To stop it: kill $MLFLOW_PID"
echo ""
