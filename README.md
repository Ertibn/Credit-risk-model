# Credit Risk Probability Model for Alternative Data
## End-to-End Implementation for Bati Bank

**Challenge**: 10 Academy Week 4 Challenge  
**Dates**: 28 May – 03 Jun 2026  
**Interim Submission**: 31 May 2026, 8:00 PM UTC  
**Final Submission**: 03 Jun 2026, 8:00 PM UTC

---

## Project Overview

This project builds an end-to-end credit risk scoring system for Bati Bank's buy-now-pay-later (BNPL) service. Using transaction data from an eCommerce platform, we engineer a credit risk proxy variable from Recency, Frequency, and Monetary (RFM) patterns, train predictive models, and deploy a containerized REST API for real-time risk scoring.

### Business Problem

Bati Bank is partnering with an eCommerce company to enable buy-now-pay-later services. The challenge is to build a defensible credit scoring model that:

1. **Defines a proxy target variable** for default risk using behavioral data (no explicit default labels available)
2. **Selects predictive features** that correlate with the risk proxy
3. **Trains and compares models** that assign risk probability scores
4. **Deploys as a REST API** for real-time loan origination decisions
5. **Operates under Basel II regulatory constraints** emphasizing interpretability and documentation

---

## Credit Scoring Business Understanding

### 1. Basel II Accord's Influence on Model Interpretability

**Basel II Context**: The Basel II Capital Accord requires financial institutions to maintain adequate capital reserves based on credit risk exposure. It mandates:
- **Risk measurement**: Precise quantification of default probability (PD), Loss Given Default (LGD), and Exposure at Default (EAD)
- **Model validation**: Rigorous backtesting and stress testing
- **Documentation**: Detailed rationale for modeling choices
- **Interpretability**: Model decisions must be explainable to regulators and stakeholders

**Impact on Our Model Design**:
- We prioritize **interpretable models** (Logistic Regression, Decision Trees) that can explain why a customer is classified as high-risk
- We document **every modeling choice** against Basel II requirements
- We implement **comprehensive monitoring** and backtesting frameworks
- We use **Weight of Evidence (WoE)** and **Information Value (IV)** to validate feature predictiveness
- We maintain **audit trails** of model decisions for regulatory review

**Regulatory Trade-off**: While complex models (deep learning, ensemble methods) may achieve higher accuracy, Basel II requires that we prioritize explainability. A 2% drop in accuracy for a fully interpretable model is preferable to a 5% improvement in accuracy with a "black box" model that regulators cannot validate.

---

### 2. Proxy Variable Necessity and Business Risks

#### Why a Proxy Variable is Necessary

The raw eCommerce transaction dataset contains **no explicit default labels**. We face two options:

**Option A: Traditional Approach (Not Available)**
- Obtain historical loan data with actual defaults
- Problem: Bati Bank has no existing BNPL history; no loans have defaulted yet

**Option B: Proxy Variable (Our Approach)**
- Use behavioral patterns to infer credit-worthiness
- Assume customers with low engagement (low recency, frequency, monetary value) are **proxy high-risk**
- Rationale: Disengaged customers may lack financial stability or intent to repay

#### Business Risks of Proxy-Based Prediction

**1. Proxy-Reality Mismatch**
- **Risk**: Behavioral disengagement ≠ Default probability
- **Example**: A customer may be inactive because they switched to competitors, not due to credit risk
- **Mitigation**: Post-deployment monitoring to compare predicted risk vs. actual defaults

**2. Self-Fulfilling Prophecy**
- **Risk**: Denying credit to "high-risk" customers may make them more likely to default (if they can access credit elsewhere)
- **Mitigation**: Stratified evaluation; test lower-risk tiers to validate calibration

**3. Bias Against New Customers**
- **Risk**: RFM-based segmentation penalizes customers with short histories
- **Example**: A new customer with 1 transaction cannot be scored fairly
- **Mitigation**: Implement minimum engagement thresholds before scoring

**4. Proxy Decay Over Time**
- **Risk**: Behavioral patterns change; what predicts default today may not hold next year
- **Mitigation**: Quarterly re-clustering and model retraining

**Regulatory Disclosure**: Our final report must explicitly state that this model uses a **proxy target variable**, not observed defaults. Regulators and business stakeholders must understand the assumptions and limitations.

---

### 3. Interpretable vs. High-Performance Models in Regulated Contexts

#### Model Options Comparison

| Aspect | Logistic Regression (WoE) | Gradient Boosting (XGBoost) |
|--------|---------------------------|---------------------------|
| **Interpretability** | ✅ High (coefficient-based) | ❌ Low (complex interactions) |
| **Explainability** | ✅ Each feature has clear impact | ❌ Feature interactions hidden |
| **Regulatory Approval** | ✅ Basel II compliant | ⚠️ Requires extensive validation |
| **Accuracy** | ⚠️ 75-80% typical | ✅ 85-92% typical |
| **Computational Cost** | ✅ Low | ⚠️ High |
| **Monitoring** | ✅ Easy (coefficients stable) | ⚠️ Complex (drift hard to diagnose) |
| **Model Stability** | ✅ Stable over time | ⚠️ May degrade with data drift |

#### Our Strategy: Hybrid Approach

1. **Task 5 (Model Training)**: Train **both** Logistic Regression (with WoE) and Gradient Boosting models
2. **Comparison**: Evaluate on accuracy, AUC, F1, calibration, and stability
3. **Selection Criteria**:
   - If accuracy difference < 5%: Choose Logistic Regression (interpretability wins)
   - If accuracy difference > 5%: Use ensemble or staged rollout (high-performance wins, but with enhanced monitoring)
4. **Documentation**: Justify the choice with explicit Basel II considerations

#### Why This Matters for Bati Bank

- **Loan Approvals**: Credit officers need to explain decisions to customers
- **Regulatory Audits**: Regulators need to validate the model's logic
- **Business Accountability**: Risk officers must understand and trust the model
- **Market Confidence**: Transparent scoring builds customer and investor confidence

---

## Project Deliverables

### Task 1: Business Understanding ✅ (COMPLETE)
- [x] GitHub repository initialized with standard structure
- [x] README.md with Credit Scoring Business Understanding section
- [x] task-1 branch created and PR prepared

### Task 2: Exploratory Data Analysis 🔄 (IN PROGRESS)
- [ ] notebooks/eda.ipynb with comprehensive data exploration
- [ ] Top 3-5 insights documented
- [ ] task-2 branch created and PR prepared

### Task 3: Feature Engineering (PENDING)
- [ ] src/data_processing.py with sklearn Pipeline
- [ ] RFM feature engineering implemented
- [ ] WoE and IV transformations

### Task 4: Proxy Target Variable (PENDING)
- [ ] RFM clustering to define high-risk customers
- [ ] is_high_risk binary column created
- [ ] Processed dataset exported

### Task 5: Model Training & Tracking (PENDING)
- [ ] src/train.py with MLflow experiment tracking
- [ ] Multiple models trained and compared
- [ ] Best model registered in MLflow Registry
- [ ] Unit tests in tests/test_data_processing.py

### Task 6: Deployment & CI/CD (PENDING)
- [ ] src/api/main.py with FastAPI service
- [ ] Dockerfile and docker-compose.yml
- [ ] .github/workflows/ci.yml with linting and testing
- [ ] Deployed API with /predict endpoint

---

## Interim Submission Checklist (Due 31 May)

- [ ] GitHub repository with merged Task 1 & 2
- [ ] README.md with Business Understanding section
- [ ] EDA notebook with 3-5 key insights
- [ ] Interim report summarizing findings

---

## Final Submission Checklist (Due 03 Jun)

- [ ] Complete GitHub repository with all tasks merged
- [ ] Medium-style blog post report
- [ ] Model comparison results and best model selection
- [ ] API deployment demonstration
- [ ] Screenshots: MLflow UI, CI/CD pipeline, Docker running
- [ ] Limitations and future work section

---

## Repository Structure

```
credit-risk-model/
├── .github/
│   └── workflows/
│       └── ci.yml                    # CI/CD pipeline
├── data/
│   ├── raw/                          # Raw transaction data
│   └── processed/                    # Model-ready datasets
├── notebooks/
│   └── eda.ipynb                     # Exploratory analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py            # Feature engineering pipeline
│   ├── train.py                      # Model training & tracking
│   ├── predict.py                    # Inference utilities
│   └── api/
│       ├── main.py                   # FastAPI application
│       └── pydantic_models.py        # Request/response schemas
├── tests/
│   └── test_data_processing.py       # Unit tests
├── Dockerfile                         # Container image
├── docker-compose.yml                # Local deployment
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── README.md                         # This file
└── INTERIM_REPORT.md                 # Interim submission report
```

---

## Key Technologies

- **Data Processing**: pandas, numpy, scikit-learn
- **Feature Engineering**: xverse (WoE/IV), scikit-learn Pipeline
- **ML Modeling**: scikit-learn, XGBoost, LightGBM
- **Experiment Tracking**: MLflow
- **API**: FastAPI, Pydantic, uvicorn
- **Containerization**: Docker, docker-compose
- **CI/CD**: GitHub Actions
- **Testing**: pytest

---

## References

### Credit Risk Fundamentals
- Basel II Capital Accord
- Weight of Evidence (WoE) and Information Value (IV)
- RFM Analysis for Customer Segmentation
- Alternative Credit Scoring Methods

### Technical Resources
- MLflow Documentation
- FastAPI Documentation
- Docker Best Practices
- scikit-learn Pipeline Tutorial

---

## Contact & Support

**Instructors**: Kerod, Mahbubah, Feven  
**Slack Channel**: #all-week-4  
**Office Hours**: Mon–Fri, 08:00–15:00 UTC  
**Challenge Dates**: 28 May – 03 Jun 2026

---

**Status**: Task 1 Complete ✅ | Task 2 In Progress 🔄 | Tasks 3-6 Pending ⏳

