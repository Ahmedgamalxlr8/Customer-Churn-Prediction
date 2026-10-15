# Telecom Customer Churn Prediction 📊📞

An end-to-end supervised machine learning classification pipeline built to predict customer attrition (churn) for a telecommunications provider, enabling proactive, data-driven customer retention campaigns.

---

## 📌 1. Business Problem & Objective

Customer churn occurs when subscribers cancel their telecom services and switch to a competitor. Because acquiring a new customer is significantly more expensive than retaining an existing one, proactive retention is a primary business priority.

This project develops a binary classification pipeline (`Churn = Yes (1)` vs. `Churn = No (0)`) using historical customer demographic, account, and service subscription data.

* **Primary Metric Priority:** Failing to identify a departing customer (**False Negative**) results in the permanent loss of their remaining Customer Lifetime Value (CLV) and recurring revenue, which is far costlier than offering a minor promotional incentive to a loyal customer (**False Positive**). Therefore, this project prioritizes **Recall (Churn)** alongside **F1-Score** to capture as many at-risk subscribers as possible while keeping false alarms manageable.

---

## 📂 2. Dataset Overview

* **Dataset:** [IBM Telco Customer Churn Dataset](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)
* **Dimensions:** `7,043` rows and `21` columns (`1` identifier, `3` continuous numerical features, `16` categorical/binary features, and `1` binary target).
* **Data Quality Fixes:**
  * Removed the non-predictive identifier column (`customerID`).
  * Identified `11` blank string records (`" "`) in `TotalCharges` (corresponding to new customers with `tenure == 0`), coerced them to `NaN`, and imputed them using the training fold median inside a leakage-free `Pipeline`.

---

## 🔍 3. Key Exploratory Data Analysis (EDA) Findings

1. **Moderate Class Imbalance:** `73.46%` of customers stayed (`No`), while `26.54%` churned (`Yes`). Relying on raw accuracy alone is misleading, as a naive majority-class predictor would achieve `73.5%` accuracy while catching `0%` of churners.
2. **Contract Type Vulnerability:** Customers on **Month-to-month** contracts account for the vast majority of attrition (`~42.7%` churn rate), whereas **One-year** (`<12%`) and **Two-year** (`<3%`) contracts exhibit strong customer lock-in.
3. **Early-Tenure Risk:** Churn is heavily concentrated within the **first 1 to 12 months** of service. Once a customer remains past the two-year mark, their probability of canceling drops sharply.
4. **Pricing & Service Friction:** Churning customers carry a noticeably higher median monthly bill (`~$79.65`) compared to retained customers (`~$64.43`), driven largely by Fiber Optic internet plans and manual Electronic Check billing.

> **Note on Causation:** While high monthly charges and month-to-month contracts correlate strongly with churn, correlation does not prove causation. Confounding factors—such as local competitor promotions, temporary housing, or unrecorded regional service outages—may simultaneously influence plan choice and cancellation.

---

## ⚙️ 4. Leakage-Free Preprocessing & Modeling Methodology

To guarantee **zero data leakage**:

1. The dataset was split into an **80% training set (`5,634` rows)** and a **20% held-out test set (`1,409` rows)** using `stratify=y` and `random_state=42` *before* any preprocessing.
2. A Scikit-Learn `ColumnTransformer` and `Pipeline` were constructed to fit all transformations strictly on the training folds:
   * **Numerical Pipeline (`tenure`, `MonthlyCharges`, `TotalCharges`):** `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`
   * **Categorical Pipeline (`16` features):** `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(drop='first', handle_unknown='ignore')`

---

## 📈 5. Model Comparison & Hyperparameter Tuning

### 5-Fold Stratified Cross-Validation (Training Set)

A balanced **Logistic Regression** model was established as the linear baseline and benchmarked against three alternative classifiers:

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Logistic Regression)** | 0.7497 | 0.5185 | **0.8013** | **0.6294** |
| **Decision Tree (`max_depth=6`)** | 0.7244 | 0.4886 | 0.8000 | 0.6064 |
| **Random Forest (Untuned)** | **0.7827** | **0.5812** | 0.6475 | 0.6124 |
| **SVM (RBF Kernel)** | 0.7483 | 0.5168 | 0.7826 | 0.6223 |

### Hyperparameter Tuning (`GridSearchCV`)

* **Random Forest Regularization:** The untuned Random Forest overfitted the majority class (`Recall = 0.6475`). Tuning tree depth and leaf size (`max_depth=8`, `min_samples_leaf=10`, `n_estimators=100`, `class_weight='balanced'`) produced a **meaningful improvement**, lifting 5-fold CV F1 from **`0.6124` to `0.6332`** and matching the `80%` recall of the linear baseline.

### Final Held-Out Test Set Evaluation (`n = 1,409`)

Evaluating the winning **Tuned Balanced Random Forest** once on the unseen test set achieved:

* **Accuracy:** `75.0%`
* **Precision (`Churn = Yes`):** `52.0%`
* **Recall (`Churn = Yes`):** `80.0%`
* **F1-Score (`Churn = Yes`):** `0.63`

#### Test Set Confusion Matrix Breakdown

* **True Positives (`TP = 299`):** Actual churners correctly flagged (`80.0%` of all departing customers), enabling early retention outreach.
* **False Negatives (`FN = 75`):** Churners missed by the model (`20.0%`), representing lost revenue opportunities.
* **False Positives (`FP = 279`):** Loyal customers flagged as at-risk; they may receive a low-cost contract upgrade offer.
* **True Negatives (`TN = 756`):** Loyal customers correctly predicted to stay (`73.0%` of non-churners), requiring no intervention budget.

---

## 💡 6. Business Recommendations & Limitations

### Actionable Retention Strategies

1. **Targeted Contract Migration Offers:** Proactively target flagged high-risk **Month-to-month** subscribers with a **10–15% monthly discount** or a complimentary month of service when they transition to a 1-year contract.
2. **First-Year Onboarding & Autopay Incentives:** For flagged customers in their **first 12 months** on high-tier **Fiber Optic** plans, trigger a proactive technical support check-in and offer a **$5/month billing credit** to switch from Electronic Check to automatic payment methods.

### Key Limitations

* **False Alarm Overhead (`52%` Precision):** Nearly half of flagged accounts would have stayed without intervention. Consequently, retention campaigns must use **value-retaining incentives** (contract lock-in discounts or service upgrades) rather than unconditional cash credits.
* **Static Data Snapshot:** The dataset lacks longitudinal behavioral signals (such as recent drops in usage, support ticket sentiment, or competitor pricing changes), requiring periodic retraining in production to mitigate data drift.

---

## 🚀 7. Repository Structure & How to Run

```text
├── Data/
│   └── Telco-Customer-Churn.csv          # Raw IBM Telco dataset
├── customer_churn_prediction.ipynb       # Full Jupyter Notebook (EDA, Pipeline, CV, Tuning)
├── churn_prediction_pipeline.joblib      # Saved end-to-end Scikit-Learn Pipeline
└── README.md                             # Project documentation & Executive Report
