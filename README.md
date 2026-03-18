## Interactive App

This project includes a Streamlit web application that allows users to input applicant details and generate a real-time default risk prediction using the trained XGBoost model.

To run locally:

```bash
streamlit run app.py

# Credit Default Prediction using Machine Learning

## Overview
This project builds a machine learning pipeline to predict whether a borrower will default on a loan, using the Home Credit dataset from Kaggle.

The objective is to compare a baseline statistical model with a more advanced machine learning model, and evaluate performance in the presence of class imbalance.

---

## Dataset
- Source: Home Credit Default Risk (Kaggle)
- Observations: ~307,000
- Target variable: `TARGET`
  - 0 = Non-default
  - 1 = Default

The dataset is highly imbalanced, with approximately 8% defaults.

---

## Approach

### Data Preparation
- Removed columns with >60% missing values
- Replaced anomalous values (e.g. `DAYS_EMPLOYED = 365243`) with NaN
- Dropped ID column (`SK_ID_CURR`)
- Split data into training and validation sets (80/20)

### Preprocessing
- Numerical features:
  - Median imputation
  - Standard scaling
- Categorical features:
  - Most frequent imputation
  - One-hot encoding

A unified preprocessing pipeline was implemented using `ColumnTransformer`.

---

## Models

### 1. Logistic Regression
- Baseline model
- Class-weighted to handle imbalance

### 2. XGBoost
- Gradient boosting model
- Captures nonlinear relationships and feature interactions

---

## Evaluation Metrics
Due to class imbalance, the following metrics were used:

- ROC-AUC
- Precision-Recall AUC
- Confusion Matrix

---

## Results

| Model | ROC-AUC | PR-AUC |
|------|--------|--------|
| Logistic Regression | ~0.75 | ~0.23 |
| XGBoost | ~0.76 | ~0.25 |

XGBoost improved predictive performance over the baseline model.

---

## Threshold Analysis

The default classification threshold of 0.5 resulted in a very conservative model, detecting very few defaults.

By lowering the threshold, the model captured more defaulting borrowers at the cost of increased false positives.

Example:

- Threshold = 0.1:
  - ~60% of defaults detected
  - Moderate increase in false positives

- Threshold = 0.05:
  - ~85% of defaults detected
  - Significant increase in false positives

This demonstrates that threshold selection must align with business objectives and risk tolerance.

---

## Feature Importance

The most important predictors were:

- External risk scores (`EXT_SOURCE_1/2/3`)
- Education level
- Income type
- Age and employment duration
- Asset ownership indicators

These results are consistent with real-world credit risk modeling, where external scores and socioeconomic factors are key drivers.

---

## Key Takeaways

- Class imbalance requires careful metric selection
- Simple models provide strong baselines
- Tree-based models capture nonlinear relationships
- Threshold tuning is critical in real-world decision-making
- Feature importance aligns with expected drivers of credit risk

---

## Future Improvements

- Incorporate additional relational datasets (bureau, previous applications)
- Hyperparameter tuning
- Model calibration
- Deployment via a web app (e.g., Streamlit)

---

## Tools Used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib

---

## Author

Numair Khan  
Actuarial / Risk / Machine Learning
