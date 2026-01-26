# Bank-Financial-Health-Analysis

# Banking Financial Health Classification
## ARB Apex Bank - Rural Banking Risk Assessment System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning project for predicting financial health status of rural banks to identify institutions at risk of liquidity failure and capital inadequacy. This portfolio project recreates real-world banking supervision analysis using synthetic data.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Results & Insights](#results--insights)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## 🎯 Project Overview

This project builds a classification system to predict bank financial health status, enabling early identification of banks requiring regulatory intervention. The model analyzes 25+ financial indicators including liquidity ratios, capital adequacy, asset quality, and profitability metrics to classify banks as **HEALTHY** or **HIGH_RISK**.

**Context:** Originally developed during a Data Science internship at ARB Apex Bank Ltd (Ghana) analyzing 224+ rural banks. This portfolio version uses synthetic data to demonstrate the same analytical methodology while maintaining confidentiality.

### Success Criteria
- ✅ **Primary:** Recall ≥ 80% (minimize false negatives - catch risky banks)
- ✅ **Secondary:** Precision ≥ 70%, F1-Score ≥ 75%, ROC-AUC ≥ 85%

---

## 💼 Business Problem

Rural and community banks face unique challenges including limited capital bases, concentrated loan portfolios, and operational inefficiencies. Banking regulators need early warning systems to identify institutions requiring intervention before failure occurs.

**Objective:** Predict which banks are at high risk of:
1. **Liquidity Crisis** - Unable to meet short-term obligations
2. **Capital Inadequacy** - Insufficient equity buffer to absorb losses  
3. **Regulatory Intervention Need** - Requiring supervisory action

**Impact:** Early identification enables proactive measures (capital injection, management changes, mergers) preventing bank failures and protecting depositors.

---

## 📊 Dataset

### Specifications
- **Banks:** 224 rural banking institutions
- **Time Period:** 12 quarters (Q1 2021 - Q4 2023)
- **Total Records:** 2,688 bank-quarter observations
- **Features:** 35 features (25+ original + engineered features)
- **Target Distribution:** 33% HIGH_RISK, 67% HEALTHY

### Feature Categories

**Liquidity Metrics**
- `liquidity_ratio` - Liquid assets / Total assets (%)
- `cash_reserve_ratio` - Cash reserves / Deposits (%)
- `loan_to_deposit_ratio` - Total loans / Deposits (%)

**Capital Metrics**
- `capital_adequacy_ratio` - Equity / Risk-weighted assets (%)
- `leverage_ratio` - Assets / Equity

**Asset Quality**
- `npl_ratio` - Non-performing loans / Total loans (%)
- `loan_loss_provisions` - Reserves for bad loans

**Profitability**
- `return_on_assets` (ROA) - Net income / Assets (%)
- `return_on_equity` (ROE) - Net income / Equity (%)
- `net_interest_margin` - Interest income spread (%)

**Operations**
- `cost_to_income_ratio` - Operating costs / Income (%)
- `staff_count`, `branches` - Operational scale

**Risk Indicators**
- `concentration_risk` - Loan portfolio concentration (%)
- `deposit_volatility` - Deposit fluctuation measure (%)

### Target Variable Definition
Banks classified as **HIGH_RISK** if meeting ANY criteria:
- Liquidity ratio < 15%
- Capital adequacy ratio < 12%
- NPL ratio > 15%
- Negative net income for 2+ consecutive quarters
- ROE < -5%

**Note:** This is synthetic data created for portfolio purposes. Original analysis used confidential banking data from ARB Apex Bank Ltd.

---

## 🏆 Model Performance

### Best Model: **XGBoost Classifier**

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Recall** | **100.0%** | ≥80% | ✅ **EXCEEDED** |
| **Precision** | **99.5%** | ≥70% | ✅ **EXCEEDED** |
| **F1-Score** | **99.7%** | ≥75% | ✅ **EXCEEDED** |
| **ROC-AUC** | **99.99%** | ≥85% | ✅ **EXCEEDED** |
| **Accuracy** | **99.8%** | - | ✅ **EXCELLENT** |

### Confusion Matrix (Test Set: 538 banks)

|  | Predicted HEALTHY | Predicted HIGH_RISK |
|---|---|---|
| **Actual HEALTHY** | 353 (TN) | 1 (FP) |
| **Actual HIGH_RISK** | 0 (FN) | 184 (TP) |

**Key Outcomes:**
- ✅ **All 184 high-risk banks correctly identified** (0 missed)
- ✅ Only 1 false alarm (1 healthy bank flagged as risky)
- ✅ 100% of actual high-risk banks caught by the model

### Hyperparameters (Best Configuration)
```python
{
    'n_estimators': 100,
    'max_depth': 7,
    'learning_rate': 0.2,
    'subsample': 0.6,
    'colsample_bytree': 0.8
}
```

### All Models Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time |
|-------|----------|-----------|--------|----------|---------|------------|
| **XGBoost** | **99.8%** | **99.5%** | **100.0%** | **99.7%** | **99.99%** | 0.13s |
| Random Forest | 99.1% | 98.4% | 98.9% | 98.6% | 99.96% | 0.26s |
| SVC | 94.6% | 90.2% | 94.6% | 92.3% | 99.08% | 0.82s |
| KNN | 94.8% | 91.1% | 94.0% | 92.5% | 98.20% | 0.002s |
| Logistic Regression | 90.3% | 84.4% | 88.0% | 86.2% | 96.46% | 2.27s |

**All models exceeded the 80% recall threshold**, with XGBoost achieving perfect recall.

---

## 🔑 Key Features

### Technical Highlights
- **End-to-End ML Pipeline:** Data generation → EDA → Feature engineering → Model training → Evaluation
- **Class Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Hyperparameter Optimization:** GridSearchCV with 5-fold stratified cross-validation
- **Model Interpretability:** Feature importance analysis and SHAP values
- **Production-Ready Code:** Modular design, comprehensive documentation, reproducible results

### Feature Engineering
Created 7 interaction features to capture complex relationships:
- Liquidity × Capital adequacy interaction
- NPL × Leverage interaction (risk amplification)
- ROE × Capital adequacy interaction
- Loan-to-Deposit × NPL interaction
- Cost-to-Income × Deposit volatility interaction
- Efficiency score composite
- Risk composite score

### Preprocessing Pipeline
- **Missing Value Imputation:** Median imputation (5% missingness in operational metrics)
- **Outlier Treatment:** Winsorization at 1st and 99th percentiles
- **Categorical Encoding:** One-hot encoding for region, bank size, lending style
- **Feature Scaling:** StandardScaler for all numerical features
- **Train-Test Split:** 80-20 stratified split (2,150 train / 538 test)

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/bank-financial-health-classification.git
cd bank-financial-health-classification
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
xgboost==1.7.6
imbalanced-learn==0.11.0
shap==0.42.1
jupyter==1.0.0
```

---

## 💻 Usage

### Running Notebooks Sequentially

1. **Data Generation**
```bash
jupyter notebook notebooks/01_data_generation.ipynb
```
Generates synthetic banking dataset (224 banks × 12 quarters)

2. **Exploratory Data Analysis**
```bash
jupyter notebook notebooks/02_eda_analysis.ipynb
```
Analyzes distributions, correlations, temporal trends, and regional patterns

3. **Feature Engineering**
```bash
jupyter notebook notebooks/03_feature_engineering.ipynb
```
Preprocesses data, creates interaction features, handles class imbalance

4. **Model Training**
```bash
jupyter notebook notebooks/04_model_training.ipynb
```
Trains 5 models, tunes hyperparameters, evaluates performance

### Using the Trained Model

```python
import joblib
import numpy as np

# Load the best model
model = joblib.load('models/best_model.pkl')
preprocessor = joblib.load('data/processed/preprocessor.pkl')

# Load new bank data
new_data = pd.read_csv('new_bank_data.csv')

# Preprocess
X_new = preprocessor.transform(new_data)

# Predict
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)

# Results
risk_banks = new_data[predictions == 1]
print(f"High-risk banks identified: {len(risk_banks)}")
```

---



## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11 |
| **ML Frameworks** | scikit-learn, XGBoost, imbalanced-learn |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Interpretation** | SHAP |
| **Development** | Jupyter Notebook |
| **Version Control** | Git, GitHub |

---

## 📈 Results & Insights

### Model Performance Highlights

1. **Perfect Recall (100%):** The XGBoost model identified all 184 high-risk banks in the test set with zero false negatives. This is critical for regulatory risk assessment where missing a failing bank has severe consequences.

2. **High Precision (99.5%):** Only 1 false positive out of 354 healthy banks, minimizing unnecessary regulatory interventions and resource waste.

3. **Exceptional Discrimination (ROC-AUC: 99.99%):** Near-perfect ability to distinguish between healthy and high-risk banks across all probability thresholds.

4. **Fast Training (<1 second):** Model training and prediction are computationally efficient, suitable for frequent retraining as new data arrives.

### Top Risk Indicators

Based on feature importance analysis, the most predictive features are:
1. **NPL Ratio** - Non-performing loans directly indicate credit quality problems
2. **Liquidity Ratio** - Low liquidity signals inability to meet obligations
3. **Capital Adequacy Ratio** - Insufficient capital buffer increases failure risk
4. **Return on Equity** - Persistent negative ROE shows unsustainable operations
5. **Consecutive Losses** - Multiple loss quarters indicate systemic issues

### Business Impact

**Cost-Benefit Analysis:**
- **Cost of False Negative (FN):** Very High - Bank failures, depositor losses, regulatory penalties
- **Cost of False Positive (FP):** Moderate - Unnecessary supervisory resources, potential reputation impact
- **Model Performance:** 0 FN, 1 FP → Optimal for risk-averse regulatory environment

**Deployment Recommendation:**
With 100% recall and 99.5% precision, this model is ready for production deployment with human oversight for final intervention decisions.

---

## ⚠️ Limitations

1. **Synthetic Data:** This project uses artificially generated data. Real-world banking data may have additional complexities, noise, and edge cases not captured in the synthetic dataset.

2. **Temporal Assumptions:** The model assumes stable relationships between features and risk status over time. Economic shocks, regulatory changes, or market conditions may alter these relationships.

3. **Regional Factors:** While region is included as a feature, local economic conditions, political factors, and infrastructure differences may require region-specific models.

4. **Class Imbalance Handling:** SMOTE generates synthetic minority class samples which may not perfectly represent real high-risk bank characteristics.

5. **External Factors:** The model does not account for macroeconomic indicators (GDP growth, inflation, interest rates) or external shocks (pandemics, political instability).

6. **Model Interpretability:** While feature importance and SHAP values provide insights, XGBoost remains a "black box" compared to simpler models like logistic regression.

7. **Sample Size:** 224 banks may be insufficient to capture the full diversity of banking institutions, especially rare failure modes.

---

## Future Improvements

### Model Enhancements
- [ ] **Time Series Features:** Incorporate trend analysis (3-month moving averages, quarter-over-quarter growth rates)
- [ ] **Ensemble Methods:** Combine XGBoost with Random Forest and Neural Networks for robust predictions
- [ ] **Anomaly Detection:** Add unsupervised learning to identify unusual patterns not captured by historical labels
- [ ] **Calibration:** Improve probability estimates for risk scoring and intervention priority ranking

### Data Enhancements
- [ ] **Macroeconomic Variables:** Add GDP growth, inflation rates, central bank policy rates
- [ ] **Network Effects:** Model interbank lending and contagion risks
- [ ] **Qualitative Factors:** Incorporate management quality scores, governance metrics
- [ ] **Real-Time Data:** Integrate daily/weekly indicators for faster early warning

### Deployment & Monitoring
- [ ] **API Development:** Build REST API for real-time predictions
- [ ] **Dashboard:** Create interactive visualization dashboard for supervisors
- [ ] **Model Monitoring:** Implement drift detection and automated retraining pipelines
- [ ] **A/B Testing:** Compare model recommendations against expert judgments

### Regulatory Alignment
- [ ] **Basel III Compliance:** Ensure model aligns with international banking standards
- [ ] **Stress Testing:** Simulate economic downturns and test model robustness
- [ ] **Explainability Reports:** Generate human-readable reports for each prediction
- [ ] **Audit Trail:** Log all predictions and decisions for regulatory review

---

## 👨‍💻 Author

**Godwin Addetsi**  
MSc Data Science, Leiden University

**LinkedIn:** [linkedin.com/in/godwin-addetsi](https://linkedin.com/in/godwin-addetsi)  
**Email:** godwinaddetsi12@gmail.com  

### About the Project

This project was developed as part of a Data Science portfolio to demonstrate:
- End-to-end machine learning project execution
- Banking domain knowledge and regulatory understanding
- Technical proficiency in classification, imbalanced learning, and model interpretation
- Professional code documentation and project organization

**Original Experience:** Data Science Intern at ARB Apex Bank Ltd (Sep 2022 – Aug 2023), where I conducted predictive analysis on 224+ rural banks to identify institutions at risk of liquidity failure and capital inadequacy.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ If you find this project useful, please consider giving it a star!**

---

