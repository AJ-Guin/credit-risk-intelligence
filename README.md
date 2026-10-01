# Credit Risk Intelligence

An end-to-end **Machine Learning Credit Risk Assessment** application built using **XGBoost** to predict loan default risk. The project combines model optimization, probability calibration, classification-threshold tuning, and **SHAP explainability** to make credit-risk predictions more interpretable.

The goal is not only to predict whether an applicant is likely to default, but also to understand **why the model made a particular prediction**.

---

## 🌐 Live Application

The model has been deployed as a web application.

👉 **[Launch Credit Risk Intelligence](https://credit-risk-intelligence-vu45.onrender.com)**

### What you can do

- Enter applicant information
- Submit a loan application
- Get the predicted risk probability
- View the final risk classification

---

## 🚀 Project Overview

Credit risk assessment is a binary classification problem where the model estimates whether a loan applicant is likely to default.

This project takes applicant and loan information such as:

- Age
- Income
- Home ownership
- Employment length
- Loan intent
- Loan grade
- Loan amount
- Interest rate
- Loan-to-income ratio
- Previous default history
- Credit history length

and predicts the applicant's **loan status**.

### Prediction Target

| Value | Meaning |
|---|---|
| `0` | Lower-risk / No default |
| `1` | Higher-risk / Default |

The project uses **XGBoost** as the primary machine learning algorithm because it performs well on structured/tabular data and can capture nonlinear relationships between borrower and loan characteristics.

---

## ✨ Key Features

- Exploratory Data Analysis (EDA)
- Data validation and cleaning
- Duplicate removal
- Outlier and invalid-value handling
- Numerical and categorical feature separation
- Missing-value imputation
- One-hot encoding for categorical features
- Class-imbalance handling using `scale_pos_weight`
- Logistic Regression baseline
- XGBoost classification
- Stratified 5-fold cross-validation
- XGBoost hyperparameter optimization using `RandomizedSearchCV`
- Precision-Recall based evaluation
- Classification threshold analysis
- Probability calibration using sigmoid calibration
- Global SHAP feature importance
- Local SHAP waterfall explanations
- False-positive and false-negative analysis
- Model serialization using `joblib`

---

# 📊 Dataset

The project uses the `credit_risk_dataset.csv` dataset.

The original dataset contains:

- **32,581 records**
- **12 columns**

After removing duplicate records, the dataset contained **32,416 records**.

The target variable is:

```text
loan_status
```

### Dataset Features

| Feature | Description |
|---|---|
| `person_age` | Applicant's age |
| `person_income` | Applicant's annual income |
| `person_home_ownership` | Home ownership status |
| `person_emp_length` | Employment length |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan grade |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Loan interest rate |
| `loan_status` | Target variable |
| `loan_percent_income` | Loan amount as a percentage of income |
| `cb_person_default_on_file` | Previous default recorded in credit history |
| `cb_person_cred_hist_length` | Length of credit history |

---

# 🔎 Exploratory Data Analysis

The project performs several EDA and data-validation steps, including:

- Dataset shape inspection
- Missing-value analysis
- Statistical summaries
- Target-class distribution
- Numerical feature identification
- Box plots for numerical variables
- Correlation analysis
- Correlation heatmap
- Detection of unrealistic ages
- Detection of extreme employment lengths

For example, the dataset contained some unrealistic age values such as `123` and `144`, which were removed during data validation.

---

# 🧹 Data Cleaning

The following validation rules were applied:

### Duplicate removal

Duplicate records were removed from the dataset.

### Age validation

Applicants were restricted to:

```text
18 <= person_age <= 100
```

### Employment validation

Employment length was restricted so that:

```text
person_emp_length <= person_age
```

and:

```text
person_emp_length <= 60
```

### Loan amount validation

Only records with:

```text
loan_amnt > 0
```

were retained.

---

# ⚙️ Data Preprocessing

The project uses Scikit-learn `Pipeline` and `ColumnTransformer` to keep preprocessing consistent between training and prediction.

### Numerical Features

Numerical missing values are handled using:

```python
SimpleImputer(strategy="median")
```

For Logistic Regression, numerical features are additionally standardized using:

```python
StandardScaler()
```

For XGBoost, scaling is not applied because tree-based models do not require feature scaling.

### Categorical Features

Categorical missing values are replaced using:

```python
SimpleImputer(
    strategy="constant",
    fill_value="missing"
)
```

Categorical variables are then transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

This also allows the model to safely handle previously unseen categorical values.

---

# ⚖️ Handling Class Imbalance

The target variable is imbalanced:

| Class | Records |
|---|---:|
| `0` | 25,473 |
| `1` | 7,108 |

Because the positive class is significantly smaller, the project calculates:

```python
scale_weight = negative_samples / positive_samples
```

The resulting weight was approximately:

```text
3.63
```

This value is passed to XGBoost through:

```python
scale_pos_weight
```

This helps the model pay more attention to the minority class.

---

# 🤖 Machine Learning Models

## 1. Logistic Regression

Logistic Regression was implemented as the baseline model.

Its preprocessing pipeline includes:

```text
Numerical Features
        ↓
Median Imputation
        ↓
Standard Scaling

Categorical Features
        ↓
Missing Value Imputation
        ↓
One-Hot Encoding
```

The baseline provides a reference point for evaluating the more complex XGBoost model.

---

## 2. XGBoost

The main model used in the project is:

```python
XGBClassifier
```

The XGBoost pipeline performs:

```text
Raw Data
   ↓
Missing Value Handling
   ↓
One-Hot Encoding
   ↓
XGBoost
   ↓
Risk Prediction
```

The model also uses:

```python
scale_pos_weight
```

to address class imbalance.

---

# 🔍 Hyperparameter Optimization

Instead of relying only on default XGBoost parameters, the project uses:

```python
RandomizedSearchCV
```

The following hyperparameters are searched:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `gamma`

The search performs:

```text
150 parameter combinations
×
5-fold cross-validation
=
750 model fits
```

The optimization uses:

```python
scoring="average_precision"
```

which is useful for imbalanced classification problems.

---

# 📈 Model Evaluation

The optimized XGBoost model achieved the following results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | **92%** |
| Precision | **82%** |
| Recall | **80%** |
| F1 Score | **81%** |

### Classification Report

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0 | 0.95 | 0.95 | 0.95 |
| 1 | 0.82 | 0.80 | 0.81 |

The test set contained **6,305 records**.

> These metrics correspond to the tuned XGBoost model evaluated before the final calibrated model was serialized.

---

# 🔄 Cross-Validation

The project uses:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Stratification ensures that the class distribution is approximately maintained across the folds.

### Cross-validation results

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 87.05% | 81.16% | 54.48% | 77.81% | 64.08% |
| XGBoost | 90.71% | 88.10% | 71.41% | 74.97% | 73.13% |

---

# 🎯 Probability Calibration

A classification model's predicted probability does not always represent a well-calibrated probability.

For example:

```text
Predicted probability = 0.70
```

should ideally correspond to approximately a 70% event frequency among similar predictions.

To improve probability calibration, the project uses:

```python
CalibratedClassifierCV
```

with:

```python
method="sigmoid"
```

and:

```python
cv=5
```

The final model is therefore based on the optimized XGBoost estimator with sigmoid probability calibration.

---

# 🎚️ Classification Threshold Optimization

Instead of blindly using the default classification threshold of:

```text
0.50
```

the project analyzes the Precision-Recall relationship and searches for a threshold that maximizes F1 score.

The final threshold calculated from the calibrated model was approximately:

```text
0.6014
```

This threshold is saved separately as:

```text
best_threshold.pkl
```

The idea is to convert the predicted probability into a final risk classification based on the selected business/modeling threshold.

---

# 🧠 SHAP Explainability

One of the major features of this project is **SHAP (SHapley Additive exPlanations)**.

SHAP is used to understand how individual features contribute to model predictions.

The project uses:

```python
shap.TreeExplainer
```

because XGBoost is a tree-based model.

---

## 🌍 Global Explainability

A SHAP summary plot is generated to understand:

- Which features are important overall
- How strongly each feature contributes
- Whether a feature tends to increase or decrease the model output

```python
shap.summary_plot(
    shap_values,
    X_test_df
)
```

This provides a global view of the model's behavior across the test dataset.

---

## 👤 Local Explainability

The project also provides applicant-level explanations using a SHAP waterfall plot.

For a particular applicant, the waterfall plot shows:

```text
Base Prediction
       ↓
Feature Contribution
       ↓
Feature Contribution
       ↓
Feature Contribution
       ↓
Final Prediction
```

For example, instead of simply saying:

```text
High Risk
```

the system can provide an explanation of which applicant characteristics pushed the prediction toward higher or lower risk.

This makes the model much more interpretable than treating XGBoost as a complete black box.

---

# 🚨 False Positive & False Negative Analysis

The project separately analyzes incorrect predictions.

### False Positive

```text
Actual = 0
Predicted = 1
```

The model incorrectly identifies a lower-risk applicant as risky.

### False Negative

```text
Actual = 1
Predicted = 0
```

The model incorrectly identifies a risky applicant as lower-risk.

For the evaluated tuned XGBoost model:

```text
False Positives: 244
False Negatives: 268
```

This analysis is particularly important in credit-risk problems because different types of classification errors can have different practical consequences.

---

# 💾 Model Saving

The final calibrated model is serialized using `joblib`:

```python
joblib.dump(
    calibrated_model,
    "credit_risk_model.pkl"
)
```

The optimized classification threshold is also saved:

```python
joblib.dump(
    best_threshold,
    "best_threshold.pkl"
)
```

### Generated Model Files

```text
credit_risk_model.pkl
best_threshold.pkl
```

---

# 🏗️ Project Architecture

```text
                    Credit Risk Assessment
                             │
                             ▼
                    Applicant Information
                             │
                             ▼
                     Data Preprocessing
                             │
                ┌────────────┴────────────┐
                │                         │
          Numerical Data            Categorical Data
                │                         │
         Median Imputation          Missing Imputation
                │                         │
                │                   One-Hot Encoding
                │                         │
                └────────────┬────────────┘
                             │
                             ▼
                         XGBoost
                             │
                             ▼
                  Hyperparameter Tuning
                             │
                             ▼
                   Probability Calibration
                             │
                             ▼
                  Risk Probability Score
                             │
                             ▼
                    Threshold Optimization
                             │
                             ▼
                     Risk Classification
                             │
                             ▼
                       SHAP Explanation
```

---

# 🛠️ Tech Stack

### Programming Language

- Python

### Machine Learning

- XGBoost
- Scikit-learn
- SciPy

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Explainable AI

- SHAP

### Model Persistence

- Joblib

---

# 📁 Suggested Project Structure

```text
Credit-Risk-Intelligence/
│
├── data/
│   └── credit_risk_dataset.csv
│
├── models/
│   ├── credit_risk_model.pkl
│   └── best_threshold.pkl
│
├── notebooks/
│   └── Credit_risk.ipynb
│
├── app/
│   └── ...
│
├── requirements.txt
├── README.md
└── .gitignore
```

> The exact application files may differ depending on how the deployment layer is organized.

---

# 📦 Main Dependencies

A typical environment for reproducing the notebook includes:

```text
numpy
pandas
scikit-learn
xgboost
scipy
matplotlib
seaborn
shap
joblib
```

---

# 💡 Example Prediction Workflow

Once the trained model is loaded:

```python
import joblib

model = joblib.load("credit_risk_model.pkl")
threshold = joblib.load("best_threshold.pkl")

probability = model.predict_proba(input_data)[:, 1]

prediction = (probability >= threshold).astype(int)
```

The resulting probability represents the model's estimated probability for class `1`, while the saved threshold is used to convert that probability into a binary classification.

---

# 🎯 Project Objectives

The project was designed to demonstrate an end-to-end machine learning workflow rather than simply training a single classifier.

The main objectives were:

1. Understand and clean real-world credit-risk data.
2. Handle missing values and categorical variables.
3. Address class imbalance.
4. Establish a baseline using Logistic Regression.
5. Train an XGBoost classifier.
6. Optimize model hyperparameters.
7. Evaluate the model using multiple classification metrics.
8. Calibrate predicted probabilities.
9. Optimize the classification threshold.
10. Analyze false positives and false negatives.
11. Use SHAP to explain model predictions.
12. Save the final model for application use.

---

# 🔬 Key Machine Learning Concepts Demonstrated

This project demonstrates practical knowledge of:

- Exploratory Data Analysis
- Feature engineering and preprocessing
- `Pipeline`
- `ColumnTransformer`
- `SimpleImputer`
- `OneHotEncoder`
- `StandardScaler`
- Train/Test Split
- Stratified Cross-Validation
- Class Imbalance
- XGBoost
- Hyperparameter Optimization
- `RandomizedSearchCV`
- Precision
- Recall
- F1 Score
- ROC-AUC
- Precision-Recall Curve
- Probability Calibration
- Classification Threshold Optimization
- SHAP
- Model Serialization

---

# ⚠️ Important Note

This project is intended as a **machine learning demonstration and educational project**.

Credit decisions in real-world financial systems involve additional factors such as regulatory requirements, fairness assessments, data governance, human oversight, validation, monitoring, and institution-specific credit policies.

The model's predictions should therefore not be treated as a standalone financial decision.

---

# 👨‍💻 Author

**AJ**

Built as an end-to-end Machine Learning project focused on:

```text
Machine Learning
        +
XGBoost
        +
Explainable AI
        +
Credit Risk Assessment
```

---

## ⭐ If you find this project useful

Feel free to ⭐ star the repository and explore the implementation.
