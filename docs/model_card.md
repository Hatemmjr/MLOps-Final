# Model Card — Telco Customer Churn Classifier

*Prepared for DDSC611 Final Project · Spring 2026 · ESLSCA University*

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | `telco-churn-model` |
| **Version** | Registered in MLflow; see Production tag |
| **Type** | Binary Classification (Churn / No Churn) |
| **Algorithm** | Best of: Gradient Boosting / Random Forest / Logistic Regression (selected by Optuna HPO) |
| **Framework** | scikit-learn 1.4.2 |
| **Training date** | See MLflow run metadata |
| **Owner** | DDSC611 Team |

---

## Intended Use

### Primary use case
Predicting which Telco customers are likely to churn within the next billing period, enabling proactive retention interventions (discount offers, service upgrades, customer success calls).

### Intended users
- Retention/CRM teams acting on churn scores
- Business analysts monitoring customer health
- MLOps engineers operating the pipeline

### Out-of-scope uses
- Credit scoring or financial risk decisions
- Legal, employment, or housing eligibility decisions
- Any use where model error could cause direct personal harm

---

## Training Data

- **Dataset**: IBM Telco Customer Churn (Kaggle / IBM Sample Data)
- **Size**: 7,043 rows × 20 features
- **Date range**: Cross-sectional (not time-series)
- **Train / Test split**: 80 / 20 stratified by target
- **Class distribution**: ~26.5% churn (positive class)
- **SMOTE**: Applied to training set to address class imbalance

Full details in `docs/data_card.md`.

---

## Evaluation Metrics

> Metrics reported on the held-out **test set** (20% of data, never seen during training or HPO).

| Metric | Score |
|--------|-------|
| ROC-AUC | ≥ 0.80 (target) |
| Accuracy | ≥ 0.78 (target) |
| F1 (positive class) | ≥ 0.60 (target) |
| Precision | TBD (see MLflow run) |
| Recall | TBD (see MLflow run) |

*Exact values are recorded in `docs/experiment_log.csv` and the MLflow UI.*

### Per-subgroup performance

| Subgroup | ROC-AUC | Notes |
|---------|---------|-------|
| Senior Citizens (SeniorCitizen=1) | TBD | Smaller subgroup; evaluate for disparity |
| Fiber Optic customers | TBD | Higher churn base rate |
| Month-to-month contracts | TBD | Dominant churn driver |
| Long-tenure customers (≥48 months) | TBD | Lower churn; check calibration |

---

## Limitations

1. **Cross-sectional data**: The model cannot capture temporal dynamics or individual customer trajectories.
2. **Geography**: The dataset reflects a single (unnamed) US telecom provider; performance may degrade on data from other regions or providers.
3. **Feature drift**: Monthly charges and tenure are the most drift-sensitive features; regular monitoring is required (see Component 6).
4. **SMOTE-induced risk**: Synthetic minority oversampling may cause over-optimistic training metrics; always validate on unmodified held-out data.
5. **Concept drift**: Customer behaviour (and the definition of "churn") may shift over time; the model should be retrained at least quarterly.

---

## Ethical Considerations

- **Sensitive attributes**: `gender` and `SeniorCitizen` are present in the data. The model may reflect historical patterns that disadvantage certain groups. Subgroup fairness analysis (see above) should be reviewed before deployment.
- **Retention actions**: Automated outreach based on churn scores should be reviewed by a human before triggering legal or contractual actions (e.g., early-termination fee waivers).
- **Transparency**: Customers should not be denied service based solely on a churn score.
- **Data minimisation**: Only features necessary for churn prediction are used; no personally identifiable information (PII) beyond `customerID` (dropped before training) is processed.

---

## Monitoring & Maintenance

- **Drift monitoring**: Evidently reports generated weekly; alert triggered if >20% of features drift.
- **Retraining trigger**: Drift alert or ROC-AUC drop below 0.75 on a fresh evaluation batch.
- **Model registry**: All versions tracked in MLflow; rollback is possible by re-promoting an earlier version to Production.
- **Prometheus metrics**: Confidence histogram, tenure & MonthlyCharges distributions, inference counts — scraped every 15 seconds.

---

## How to Cite

```
DDSC611 Team (2026). Telco Customer Churn MLOps Pipeline.
ESLSCA University, Machine Learning Engineering Practices.
GitHub: https://github.com/Hatemmjr/MLOps-Final
```
