# Data Card — IBM Telco Customer Churn Dataset

*Prepared for MLOps Final Project*

---

## Dataset Overview

| Field | Value |
|-------|-------|
| **Name** | IBM Telco Customer Churn |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) / IBM Sample Data |
| **License** | Open / Community Data License Agreement |
| **Version used** | v1 (7,043 rows) |
| **Format** | CSV (tabular) |
| **Task** | Binary Classification — predict customer churn |

---

## Source & Collection

The dataset was originally published by IBM as a sample dataset for Watson Analytics. It contains demographic, account, and service information for fictional Telco customers. It is widely used in academic and industry tutorials for churn modelling.

- **Original publisher**: IBM Corporation
- **Public URL**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Date of access**: 2026

---

## Schema

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `customerID` | string | Unique customer identifier | **Dropped before training** |
| `gender` | categorical | Male / Female | Sensitive attribute |
| `SeniorCitizen` | int (0/1) | Whether the customer is a senior | Sensitive attribute |
| `Partner` | categorical | Yes / No | |
| `Dependents` | categorical | Yes / No | |
| `tenure` | int | Months as a customer | Range: 0–72 |
| `PhoneService` | categorical | Yes / No | |
| `MultipleLines` | categorical | Yes / No / No phone service | |
| `InternetService` | categorical | DSL / Fiber optic / No | |
| `OnlineSecurity` | categorical | Yes / No / No internet service | |
| `OnlineBackup` | categorical | Yes / No / No internet service | |
| `DeviceProtection` | categorical | Yes / No / No internet service | |
| `TechSupport` | categorical | Yes / No / No internet service | |
| `StreamingTV` | categorical | Yes / No / No internet service | |
| `StreamingMovies` | categorical | Yes / No / No internet service | |
| `Contract` | categorical | Month-to-month / One year / Two year | Strong churn predictor |
| `PaperlessBilling` | categorical | Yes / No | |
| `PaymentMethod` | categorical | 4 categories | |
| `MonthlyCharges` | float | Monthly bill amount (USD) | Range: ~$18–$119 |
| `TotalCharges` | float | Total billed to date (USD) | Contains missing values (~11 rows) |
| `Churn` | categorical → int | Target: Yes=1 / No=0 | ~26.5% positive |

---

## Preprocessing Decisions

1. **customerID dropped**: Not predictive; acts as a primary key only.
2. **TotalCharges coerced to float**: A small number of rows contain whitespace strings (`" "`); these are converted to `NaN` and imputed with the column median.
3. **Target encoding**: `Churn` is mapped from `{Yes, No}` to `{1, 0}` for binary classification.
4. **Missing value imputation**: Numeric features → median; categorical features → most frequent value.
5. **Scaling**: `StandardScaler` applied to all numeric features to centre and normalise.
6. **Feature Selection**: 6 low-signal categorical features (`gender`, `PhoneService`, `MultipleLines`, `InternetService`, `StreamingTV`, `StreamingMovies`) are dropped dynamically based on Chi-Squared statistical testing to reduce noise.
7. **Categorical encoding**: `OneHotEncoder(handle_unknown="ignore")` applied to all remaining categorical features.
8. **SMOTE**: Applied to the training set only to address the ~73/27 class imbalance.

---

## Train / Test / Reference / Production Split

| Split | Size | Purpose |
|-------|------|---------|
| Train | 80% of full data | Model training + HPO |
| Test | 20% of full data | Held-out evaluation; model validation gate |
| Reference | First 70% of full data | Evidently baseline distribution |
| Production | Last 30% of full data + injected drift | Drift monitoring simulation |

**Drift simulation**: Gaussian noise (μ=0, σ = 2× feature std) is added to `MonthlyCharges`, `tenure`, and `TotalCharges` in the production set to simulate sensor/distribution drift. This is clearly documented and visible in the Evidently drift report.

---

## Known Biases

1. **Gender balance**: The dataset is approximately 50/50 male/female; churn rates may differ by gender.
2. **Senior citizen underrepresentation**: Only ~16% of customers are senior citizens; subgroup metrics may be unreliable for this group.
3. **Geographic homogeneity**: All customers are from a single (unnamed) provider; results may not generalise across markets.
4. **Snapshot bias**: The data is cross-sectional; it does not capture the customer journey over time.
5. **Synthetic origin**: As a vendor-generated sample, the data may not reflect real-world noise, edge cases, or regulatory constraints.

---

## Privacy & Licensing Notes

- The dataset does **not** contain real customer PII; `customerID` values are synthetic.
- The dataset is licensed for educational and research use under the IBM Community Data License Agreement.
- No special data processing agreements are required for academic use.
- For production deployment against real customer data, a full Data Protection Impact Assessment (DPIA) would be required under GDPR / local equivalents.

---

## How to Access

```bash
# Download via Kaggle CLI
kaggle datasets download -d blastchar/telco-customer-churn
unzip telco-customer-churn.zip -d data/raw/

# Track with DVC
dvc add data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
git add data/raw/.gitignore data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.dvc
```

*Raw data is tracked by DVC and must never be committed directly to Git.*
