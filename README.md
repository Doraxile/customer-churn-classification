# Customer Churn Classification
### Predicting Telco Customer Churn with Random Forest

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.x-150458?style=flat-square&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-10b981?style=flat-square)

---

## Overview

An end-to-end machine learning pipeline that predicts whether a telecom customer will churn, built on the **IBM Telco Customer Churn** dataset from Kaggle. The project follows a structured 9-unit data science methodology — from data collection to model evaluation and business recommendation.

**Key results (optimized model on held-out test set):**

| Metric | Score | Target | Status |
|---|---|---|---|
| Accuracy | 0.7665 | ≥ 0.75 | ✅ Met |
| Precision (Churn) | 0.5431 | ≥ 0.55 | ❌ Slightly below |
| **Recall (Churn)** | **0.7446** | ≥ 0.55 | ✅ Met |
| **F1-Score (Churn)** | **0.6281** | ≥ 0.55 | ✅ Met |

> Recall is the primary metric here — in churn prevention, missing an at-risk customer (false negative) is more costly than flagging a non-churner for retention outreach.

---

## Dataset

**Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|---|---|
| File | `Telco-Customer-Churn.csv` |
| Rows | 7,043 customers |
| Columns | 21 attributes |
| Target | `Churn` (Yes / No) |
| Class distribution | 73.5% Not Churn · 26.5% Churn |

Place the CSV in the project root before running the notebook. The notebook auto-detects it from both `./` and `./data/`.

---

## Project Structure

```
customer-churn-classification/
│
├── Prediksi_Customer_Churn_...ipynb   # Main notebook (9 units)
├── Telco-Customer-Churn.csv           # Dataset (download from Kaggle)
└── README.md
```

---

## Methodology — 9-Unit Pipeline

| Unit | Stage | Key Output |
|---|---|---|
| #1 | Data Collection | `df_integrated` — 7,043 × 21 |
| #2 | Data Exploration | EDA, crosstabs, 5 business hypotheses |
| #3 | Data Validation | Identified 11 blank strings in `TotalCharges` |
| #4 | Data Object Selection | Removed `customerID`; 20 columns retained |
| #5 | Data Cleaning | Fixed `TotalCharges` dtype, stripped whitespace |
| #6 | Feature Engineering | 9 new features → 29 total columns |
| #7 | Label Encoding | `Churn`: Yes→1, No→0; separated X and y |
| #8 | Model Building | Pipeline + RandomizedSearchCV (12 iter, CV=3) |
| #9 | Model Evaluation | Confusion matrix, classification report, feature importance |

---

## Feature Engineering

9 new features were created from existing columns:

| New Feature | Type | Description |
|---|---|---|
| `tenure_group` | Categorical | Tenure binned into 4 groups: 0–12, 13–24, 25–48, 49–72 months |
| `monthly_charge_group` | Categorical | MonthlyCharges quantile-binned: Low / Medium / High |
| `avg_charges_per_tenure` | Numeric | `TotalCharges / tenure` — average historical spend per active month |
| `has_internet_service` | Binary (0/1) | Whether customer has any internet service |
| `has_tech_support` | Binary (0/1) | Whether customer subscribed to tech support |
| `has_online_security` | Binary (0/1) | Whether customer subscribed to online security |
| `has_online_backup` | Binary (0/1) | Whether customer subscribed to online backup |
| `has_streaming_service` | Binary (0/1) | Whether customer uses any streaming service |
| `is_auto_payment` | Binary (0/1) | Whether payment method is automatic (bank transfer / credit card) |

---

## Model

**Algorithm:** Random Forest Classifier (scikit-learn)

**Pipeline:**
```
ColumnTransformer
├── Numeric features   → passthrough
└── Categorical features → OneHotEncoder(handle_unknown='ignore')
         ↓
RandomForestClassifier(class_weight='balanced')
```

**Hyperparameter tuning** via `RandomizedSearchCV`:
- `n_iter=12`, `cv=3`, `scoring='f1'`
- Search space: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`

**Best parameters found:**
```python
{
    'model__n_estimators': 200,
    'model__max_depth': 30,
    'model__min_samples_split': 10,
    'model__min_samples_leaf': 4,
    'model__max_features': 'log2'
}
```
Best CV F1-Score: **0.6301**

**Train-test split:** 80 / 20 with `stratify=y` to preserve class distribution.

---

## Results

**Baseline vs Optimized:**

| Metric | Baseline | Optimized |
|---|---|---|
| Accuracy | 0.7616 | **0.7665** |
| Precision (Churn) | 0.5466 | 0.5431 |
| Recall (Churn) | 0.5833 | **0.7446** |
| F1-Score (Churn) | 0.5644 | **0.6281** |

The optimized model improved Recall by +16 percentage points — meaning significantly more at-risk customers are now correctly identified.

**Classification Report (Optimized):**
```
              precision    recall  f1-score   support

 Tidak Churn       0.89      0.77      0.83      1033
       Churn       0.54      0.74      0.63       372

    accuracy                           0.77      1405
   macro avg       0.72      0.76      0.73      1405
weighted avg       0.80      0.77      0.78      1405
```

**Top 5 Most Important Features:**

| Rank | Feature | Importance |
|---|---|---|
| 1 | `tenure` | 0.1025 |
| 2 | `TotalCharges` | 0.0807 |
| 3 | `Contract_Month-to-month` | 0.0764 |
| 4 | `avg_charges_per_tenure` | 0.0598 |
| 5 | `MonthlyCharges` | 0.0564 |

---

## Business Insights

Based on feature importance and EDA, high-risk churn customers tend to share these characteristics:

- **Short tenure** (0–12 months) — the critical early loyalty window
- **Month-to-month contract** — no long-term commitment, easy to leave
- **High monthly charges** — perceived value mismatch
- **Electronic check payment** — no automatic payment commitment
- **No tech support or online security** — low engagement with add-on services

**Recommended retention strategies:**
1. Offer contract upgrade incentives (monthly → annual) for customers with tenure < 6 months
2. Review pricing structure for Fiber Optic customers — high cost, high churn
3. Run upselling campaigns for TechSupport and OnlineSecurity to increase stickiness
4. Incentivize auto-payment enrollment (bank transfer / credit card)
5. Implement proactive onboarding check-ins at month 3 and 6

---

## How to Run

**Requirements:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**Steps:**
1. Download `Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Place it in the same folder as the notebook
3. Open and run all cells in `Prediksi_Customer_Churn_...ipynb`

> Python 3.10+ recommended. All cells include learning notes and output interpretation.

---

## Tech Stack

`Python 3.12` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Jupyter Notebook`

---

## Author

**Doraxile**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-atdb-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/atdb)
[![GitHub](https://img.shields.io/badge/GitHub-Doraxale-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Doraxile)

---

*Final project — Data Science Course, Universitas Gunadarma, Semester 8 (June 2026)*
*Scheme: Associate Data Scientist (Ilmuwan Data Madya)*
