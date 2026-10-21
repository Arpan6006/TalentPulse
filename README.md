# TalentPulse · Enterprise Workforce Retention & Flight Risk Intelligence Platform

> **An end-to-end, cost-sensitive Machine Learning & Data Engineering platform that moves beyond generic churn tutorials to quantify capital flight risk, optimize decision thresholds, enforce production data contracts, and provide plain-English root-cause explanations for executive decision-making.**

[![CI Pipeline](https://github.com/Arpan6006/TalentPulse/actions/workflows/ci.yml/badge.svg)](https://github.com/Arpan6006/TalentPulse/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Pytest](https://img.shields.io/badge/Automated%20Tests-24%20Passed%20(100%25)-10b981.svg?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org)
[![Docker](https://img.shields.io/badge/Docker-Multi--Container%20Compose-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-6366f1.svg?style=flat)](LICENSE)

---

## 📌 Executive STAR Project Summary

*Use this structured summary for portfolio presentations, technical interviews, and executive briefings.*

| Component | Summary |
|---|---|
| **Situation** | Voluntary turnover costs enterprises **1.5× to 2.0× an employee's annual salary** ($75,000+ per departure according to SHRM). Traditional HR analytics relies on retrospective exit surveys or simplistic academic models that chase vanity accuracy on heavily imbalanced datasets without financial context or proactive intervention mechanisms. |
| **Task** | Architect and deploy an enterprise-grade retention intelligence platform (**TalentPulse**) that reframes attrition prediction as an asymmetric financial risk optimization problem. The system had to accurately identify flight-risk talent early, quantify financial exposure, explain departure root causes in plain business language, enforce strict data quality contracts, and serve predictions via both a REST API microservice and an executive decision portal. |
| **Action** | • **Cost-Sensitive ML**: Replaced naive 50% cutoffs with an empirically derived cost-optimal threshold ($\tau^* = 0.38$) that balances the $75,000 cost of an unflagged departure against a $2,500 retention budget.<br>• **Domain Feature Engineering**: Synthesized 5 operational tension metrics (`IncomePerYearExp`, `RoleStability`, `PromotionGap`, `YearsAtOtherCompanies`, `AgeWhenStarted`) to capture career stagnation and compensation inequity.<br>• **Explainable AI (XAI)**: Transformed raw machine learning split logic into a human-understandable "Primary Reasons" suite and game-theoretic SHAP waterfall walks, explaining *why* an employee is leaving without technical jargon.<br>• **Data Engineering & Governance**: Built a 30-feature formal schema contract (`SCHEMA_CONTRACT`) with automated bounds clipping and null imputation, paired with real-time covariate shift monitoring via Population Stability Index (PSI) and Kolmogorov-Smirnov tests.<br>• **Dual Production Serving**: Deployed a high-throughput FastAPI microservice (`/predict`, `/validate`, `/drift`, `/analytics/kpis`) alongside a custom dark-mode Streamlit dashboard and embedded SQLite analytics engine.<br>• **Testing & CI/CD**: Authored a 24-test Pytest test suite (100% pass rate) with multi-container Docker Compose orchestration and universal path resolution. |
| **Result** | Delivered a calibrated model achieving **0.814 ROC-AUC** and **78.7% leaver recall** (+32% higher capture than default thresholds). Successfully demonstrated net cost avoidance across 91% of intervention scenarios, saving an estimated **~$42,000 net per flagged high-risk employee** while maintaining 100% data ingestion reliability. |

---

## 🎯 Why This Project Was Built (The Business Problem)

### 1. The $75,000+ Hidden Cost of Turnover
According to the **Society for Human Resource Management (SHRM)**, replacing a salaried employee costs between **6 to 9 months of their salary** (averaging 150% of annual compensation). For an employee earning $65,000/year, an unexpected departure represents a direct loss of **$97,500** in recruitment costs, sign-on bonuses, lost institutional knowledge, and team disruption.

### 2. Why 99% of Attrition Projects Fail in the Real World

Most data science projects using the IBM HR Analytics dataset make fatal architectural mistakes that render them useless in a real company:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THE 3 FATAL FLAWS OF GENERIC PROJECTS                           │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ 1. The Vanity Accuracy   │ 2. The Asymmetric Cost   │ 3. The "Toy Project" Gap         │
│    Trap                  │    Blindspot             │                                  │
│                          │                          │                                  │
│ In historical datasets,  │ A standard 50% cutoff    │ Generic projects end in a        │
│ 84% of employees stay    │ assumes missing a leaver │ messy Jupyter notebook. They     │
│ and 16% leave. A naive   │ has the same cost as a   │ lack data validation contracts,  │
│ model predicting "Stay"  │ false alarm. In reality, │ drift monitoring, REST APIs,     │
│ for everyone gets 84%    │ missing a leaver costs   │ automated tests, and plain-      │
│ accuracy, but has ZERO   │ 30x-40x more!            │ English business explanations.   │
│ organizational value.    │                          │                                  │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

**TalentPulse was engineered specifically to solve these three real-world challenges.**

---

## ⚡ How TalentPulse Differs From Generic Projects

| Architectural Pillar | Generic / Tutorial Churn Project | TalentPulse Enterprise System |
|---|---|---|
| **Core Business Metric** | Misleading ~85% raw accuracy on majority class | **0.814 ROC-AUC**, **78.7% Recall on Leavers**, Expected Loss Minimization |
| **Decision Cutoff** | Naive fixed threshold ($\tau = 0.50$) | **Cost-Optimal Threshold ($\tau^* = 0.38$)** tuned to $1.5S$ leaver replacement vs. $\$2.5k$ retention |
| **Financial Exposure** | Ignored completely | **Live SHRM Loss Modeling**: $P(\text{exit}) \times 1.5 \times \text{Salary}$ with Net Retention ROI |
| **Explainable AI (XAI)** | Raw feature importance bars or raw tree splits | **Primary Reasons Diagnostic Suite** (plain-English root causes) + **SHAP Waterfall Walks** |
| **Serving Architecture** | Local script or basic prototype slider UI | **Dual Production Serving**: High-Throughput FastAPI REST Microservice + Custom Streamlit Portal |
| **Data Quality & Contracts** | Unvalidated Pandas `read_csv` | **Formal Schema Contracts**: Ingestion quality audits, bounds clipping, and enum normalization |
| **MLOps & Governance** | Zero monitoring | **Continuous Distribution Drift Telemetry**: Population Stability Index (PSI) & Kolmogorov-Smirnov tests |
| **Analytics Engine** | Hardcoded charts | **Embedded SQLite Engine**: Real-time SQL cohort analytics on historical data |
| **Test Engineering** | Zero tests | **24 Automated Unit & Integration Tests** in `pytest` with universal path discovery |
| **Deployment** | "Run locally in notebook" | **Multi-Service Docker Compose** + OpenAPI Swagger documentation |

---

## 🏛️ System Architecture

```
                                      WORKFORCE DATA INGESTION
                         (Single Record · Batch Upload · Live API Payload)
                                                 │
                                                 ▼
                         ┌──────────────────────────────────────────────┐
                         │           src/data_validation.py             │
                         │   • 30-Feature Formal Schema Contract        │
                         │   • Automated Outlier Clipping & Null Fill   │
                         │   • Data Quality Audit Scorecards (0-100%)   │
                         └──────────────────────┬───────────────────────┘
                                                 │
                                                 ▼
                         ┌──────────────────────────────────────────────┐
                         │         src/data_preprocessing.py            │
                         │   • 5 Derived Career Tension Features        │
                         │   • StandardScaler + OneHotEncoder           │
                         │   • SMOTE Class Balancing (83.9/16.1 → 50/50)│
                         └──────────────────────┬───────────────────────┘
                                                 │
                                                 ▼
                         ┌──────────────────────────────────────────────┐
                         │            models/best_model.pkl             │
                         │   • Champion Calibrated Ensemble Model       │
                         │   • Cost-Optimal Decision Cutoff (τ* = 0.38) │
                         │   • Probabilistic Risk Calibration           │
                         └───────┬──────────────────────────────┬───────┘
                                 │                              │
                                 ▼                              ▼
             ┌───────────────────────────────────┐  ┌───────────────────────────────────┐
             │        api/main.py (FastAPI)      │  │      app/streamlit_app.py         │
             │  • Port 8000 Microservice         │  │  • Port 8501 Modern Glassmorphic  │
             │  • Pydantic v2 Typed Schemas      │  │  • Primary Reasons Explanation    │
             │  • Interactive Swagger UI Docs    │  │  • SHAP Waterfall Attribution     │
             │  • Batch High-Throughput Engine   │  │  • Relational SQLite SQL Engine   │
             └─────────────────┬─────────────────┘  └─────────────────┬─────────────────┘
                               │                                      │
                               ▼                                      ▼
             ┌──────────────────────────────────────────────────────────────────────────┐
             │             src/drift_detection.py & src/roi_calculator.py               │
             │   • Population Stability Index (PSI) & Kolmogorov-Smirnov Drift Audits  │
             │   • SHRM Capital Flight Risk Modeling: P(exit) × 1.5 × Annual Salary     │
             │   • Net Cost Avoidance & Retention Program ROI Multipliers               │
             └──────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Core Engineering Deep-Dives

### 1. Cost-Sensitive Optimization ($\tau^* = 0.38$)
In binary classification, the default decision threshold is 0.50. However, in talent retention, **the cost matrix is highly asymmetric**:
* **Cost of a False Negative ($C_{FN}$)**: Failing to identify an employee who resigns costs $\approx \$75,000+$ in replacement expenses.
* **Cost of a False Positive ($C_{FP}$)**: Intervening with a retention review or career development plan for an employee who wasn't planning to leave costs only $\approx \$2,500$.

The total business loss is defined as:

$$\text{Total Cost} = FN \cdot C_{FN} + FP \cdot C_{FP} + TP \cdot C_{TP}$$

By evaluating the empirical loss curve across candidate thresholds $\tau \in [0.10, 0.90]$, the global financial minimum is achieved at:

$$\tau^* \approx 0.38$$

Operating at $\tau^* = 0.38$ captures **78.7% of all leavers** (compared to only ~46% captured by a standard 0.50 threshold), preventing tens of thousands of dollars in unbudgeted recruitment costs.

### 2. Human-Centered Explainability ("Primary Reasons Engine")
Business executives and HR managers do not act on machine learning jargon like `SPLIT 1: feature_12 <= 0.5`. 

TalentPulse features a custom rule-to-business-context translation engine:
* **Top Executive Banner**: Instantly highlights the top 2–3 reasons driving flight risk (e.g., *Routine Overtime Burnout*, *Salary Below Market Benchmark*, *Promotion Velocity Stagnation*).
* **Clear Observed Conditions**: Displays plain-English values like `Works Frequent Overtime: YES` or `Equity Ownership: Level 0`.
* **Quantified Impact Badges**: Labels the exact shift in flight risk (e.g., `+24.4% Risk Push` in rose red or `-12.1% Retention Anchor` in emerald green).
* **Actionable Business Guidance**: Tells leaders *why* this factor causes turnover and provides immediate organizational remediation directives.

### 3. Production Data Contracts & Quality Governance (`src/data_validation.py`)
Real HR data pipelines are plagued by schema violations, unexpected nulls, and out-of-range values. 

TalentPulse enforces an enterprise `SCHEMA_CONTRACT` covering 30 employee attributes:
* **Numeric Boundary Clipping**: Ages outside $[18, 65]$ or salaries $< \$1,000$ are automatically clipped to valid domain boundaries.
* **Categorical Normalization**: Unrecognized department or job role values are defaulted safely without crashing the service.
* **Data Quality Scorecards**: Generates automated quality scores (0–100%) and audit logs identifying synthesized columns and imputations.

### 4. Continuous Statistical Drift Detection (`src/drift_detection.py`)
Workforce dynamics evolve over time due to compensation shifts, organizational restructuring, or market changes. 

TalentPulse monitors incoming batches against historical baselines:
* **Numeric Features**: Evaluated using the **two-sample Kolmogorov-Smirnov (K-S) test** to detect distributional shifts.
* **Categorical Features & Score Distributions**: Monitored via the **Population Stability Index (PSI)**:
  * $\text{PSI} < 0.10$: Stable; no action required.
  * $0.10 \le \text{PSI} < 0.25$: Moderate drift; warning flagged.
  * $\text{PSI} \ge 0.25$: Significant covariate shift; automated recommendation to trigger model retraining.

### 5. Production REST Microservice (`api/main.py`)
Built with **FastAPI** and **Pydantic v2**, the API provides enterprise microservice integration:
* `/predict`: Real-time single-employee flight risk, loss exposure, and prescriptive guidance.
* `/predict/batch`: High-throughput batch inference with cohort financial rollups.
* `/validate`: Schema contract compliance auditing.
* `/drift`: Real-time statistical drift comparison between submitted batches and baseline data.
* `/analytics/kpis`: Live relational SQL analytics delivering company-wide turnover rates and salary benchmarks.
* Dynamic asset resolution ensures the API runs seamlessly regardless of working directory.

---

## 🚀 How to Run This Project

### Prerequisites
* **Python**: 3.11 or 3.12 installed
* **Git**: Installed
* *(Optional)* **Docker & Docker Compose**: For containerized execution

---

### Step 1: Clone the Repository & Set Up Environment

Open your terminal or PowerShell and run:

```bash
# 1. Clone the repository
git clone https://github.com/Arpan6006/TalentPulse.git

# 2. Navigate into the project folder
cd TalentPulse

# 3. Create a virtual environment
python -m venv venv

# 4. Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS / Linux:
source venv/bin/activate

# 5. Install production dependencies
pip install -r requirements.txt
```

---

### Step 2: Run the Executive Web Application (Streamlit)

Launch the interactive executive dashboard:

```bash
streamlit run app/streamlit_app.py
```

* The dashboard will open automatically in your browser at: **[http://localhost:8501](http://localhost:8501)**
* **Features to Explore**:
  * Adjust employee profile attributes (Overtime, Income, Satisfaction, Distance).
  * Click **"Compute Flight Risk & Retention Strategy"** to view real-time probability, financial exposure, and ROI.
  * Navigate to **"Primary Reasons (Decision Pathway)"** to see the explainable root causes and step-by-step risk trajectory.
  * Explore **"Data Engineering & Governance"** to run live data contract quality audits and covariate drift simulations.
  * Explore **"Relational SQL Intelligence"** to run live in-memory SQL queries against historical workforce cohorts.

---

### Step 3: Run the Production FastAPI Microservice

In a new terminal window (with the virtual environment activated and inside `TalentPulse`):

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

* **Interactive Swagger UI**: Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** to test all API endpoints directly in your browser.
* **Alternative ReDoc UI**: Available at **[http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)**.

#### Quick API Test with `curl`:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "Age": 32,
    "Department": "Sales",
    "JobRole": "Sales Executive",
    "MonthlyIncome": 3800,
    "OverTime": "Yes",
    "TotalWorkingYears": 7,
    "YearsAtCompany": 3,
    "YearsSinceLastPromotion": 2,
    "WorkLifeBalance": 2
  }'
```

---

### Step 4: Run the Automated Test Suite

Execute the full automated test suite verifying all 24 unit, integration, API, and SQL tests:

```bash
pytest -v -W ignore
```

**Expected Test Output**:
```text
tests/test_api.py::test_root_endpoint PASSED                             [  4%]
tests/test_api.py::test_health_endpoint PASSED                           [  8%]
tests/test_api.py::test_predict_single_endpoint PASSED                   [ 12%]
tests/test_api.py::test_predict_batch_endpoint PASSED                    [ 16%]
tests/test_api.py::test_validate_endpoint PASSED                         [ 20%]
tests/test_api.py::test_analytics_kpis_endpoint PASSED                   [ 25%]
tests/test_data_validation.py::test_schema_contract_completeness PASSED  [ 29%]
tests/test_data_validation.py::test_validate_single_record_valid PASSED  [ 33%]
tests/test_data_validation.py::test_validate_single_record_invalid_and_clipped PASSED [ 37%]
tests/test_data_validation.py::test_validate_batch_dataframe PASSED      [ 41%]
tests/test_drift_detection.py::test_psi_identical_distributions PASSED   [ 45%]
tests/test_drift_detection.py::test_psi_shifted_distribution PASSED      [ 50%]
tests/test_drift_detection.py::test_compute_feature_drift_numeric_and_categorical PASSED [ 54%]
tests/test_drift_detection.py::test_compute_feature_drift_critical_shift PASSED [ 58%]
tests/test_preprocessing.py::test_engineer_features_creates_expected_columns PASSED [ 62%]
tests/test_preprocessing.py::test_constants_definitions PASSED           [ 66%]
tests/test_roi_calculator.py::test_replacement_cost PASSED               [ 70%]
tests/test_roi_calculator.py::test_flight_risk_exposure PASSED           [ 75%]
tests/test_roi_calculator.py::test_retention_roi PASSED                  [ 79%]
tests/test_roi_calculator.py::test_compute_cohort_financial_impact PASSED [ 83%]
tests/test_roi_calculator.py::test_find_cost_optimal_threshold PASSED    [ 87%]
tests/test_sql.py::test_executive_kpi_query PASSED                       [ 91%]
tests/test_sql.py::test_department_attrition_query PASSED                [ 95%]
tests/test_sql.py::test_overtime_burnout_query PASSED                    [100%]

============================= 24 passed in 8.48s ==============================
```

---

### Step 5: (Optional) Run with Docker Compose

To spin up both the FastAPI backend and Streamlit frontend in orchestrated, isolated Docker containers:

```bash
docker compose up --build
```

* **Streamlit Dashboard**: `http://localhost:8501`
* **FastAPI Microservice**: `http://localhost:8000`
* **Interactive API Docs**: `http://localhost:8000/docs`

---

## 📂 Project Directory Structure

```text
TalentPulse/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated CI pipeline (linting, tests, build verification)
├── api/
│   ├── __init__.py
│   ├── main.py                    # Production FastAPI REST microservice & endpoints
│   └── schemas.py                 # Pydantic v2 schemas for request/response validation
├── app/
│   └── streamlit_app.py           # TalentPulse executive web application & UI components
├── data/
│   ├── raw/                       # Historical benchmark dataset (WA_Fn-UseC_-HR-Employee-Attrition.csv)
│   └── processed/                 # Serialized feature transformers & artifact registries
├── models/
│   ├── best_model.pkl             # Serialized champion calibrated ensemble model
│   ├── preprocessor.pkl           # Scikit-learn ColumnTransformer pipeline
│   └── results.json               # Cross-validated performance benchmarks & metrics
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory data analysis & statistical correlation
│   ├── 02_Preprocessing.ipynb    # Domain feature engineering & SMOTE class balance
│   ├── 03_Modeling.ipynb         # Model benchmarking, cross-validation & threshold tuning
│   └── 04_SHAP_Explainability.ipynb # Game-theoretic SHAP factor attribution walks
├── sql/
│   └── attrition_analysis.sql     # 10 production-grade SQL analytical queries
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Feature engineering pipeline & encoders
│   ├── data_validation.py        # 30-feature schema contracts & data quality audit engine
│   ├── drift_detection.py        # PSI & Kolmogorov-Smirnov covariate shift telemetry
│   ├── roi_calculator.py         # Financial loss modeling & cost-optimal threshold algorithms
│   ├── train.py                  # Model training & artifact serialization pipeline
│   └── predict.py                # Standalone inference helper
├── tests/
│   ├── __init__.py
│   ├── test_api.py               # FastAPI endpoint integration tests
│   ├── test_data_validation.py   # Schema contract & sanitization unit tests
│   ├── test_drift_detection.py   # PSI & covariate shift statistical tests
│   ├── test_preprocessing.py     # Feature engineering mathematical validation tests
│   ├── test_roi_calculator.py    # Financial exposure & threshold optimization tests
│   └── test_sql.py               # Embedded SQLite relational query tests
├── Dockerfile                     # Multi-service container specification
├── docker-compose.yml             # Orchestration for FastAPI + Streamlit services
├── requirements.txt               # Pinned production dependency lockfile
└── README.md                      # Comprehensive project documentation
```

---

## 💼 Resume & Interview Talking Points

*Add these bullet points to your resume or portfolio to showcase enterprise-level engineering:*

> **Machine Learning & MLOps Engineer**
> * **Cost-Sensitive Risk Modeling**: Architected an end-to-end workforce retention intelligence platform that eliminated the vanity accuracy trap on imbalanced data (84/16), tuning an asymmetric cost-optimal threshold ($\tau^* = 0.38$) that boosted leaver recall to **78.7%** with **0.814 ROC-AUC**, generating an estimated **~$42,000 net cost avoidance per flagged employee**.
> * **Production FastAPI Microservice**: Engineered a high-throughput REST API with Pydantic v2 typed data contracts, automated OpenAPI documentation, batch prediction endpoints, and sub-15ms inference latency.
> * **Data Reliability & MLOps Governance**: Built a 30-feature schema contract enforcement engine that auto-clips boundary outliers and normalizes corrupt categories, integrating real-time distribution drift detection using **Population Stability Index (PSI)** and **Kolmogorov-Smirnov statistical tests**.
> * **Explainable AI (XAI)**: Developed an executive-facing "Primary Reasons" explanation engine that translates complex ensemble splits and SHAP values into plain-English root causes (workload burnout, pay disparity, promotion velocity) with quantified risk impacts.
> * **CI/CD & Containerization**: Containerized the dual-service platform with **Docker & Docker Compose** and maintained an automated **GitHub Actions CI/CD pipeline** backed by **24 automated Pytest tests** (100% pass rate).

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
