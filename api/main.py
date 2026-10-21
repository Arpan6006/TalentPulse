# ============================================================
# Employee Attrition Prediction - Production REST API Microservice
# ============================================================
# Production FastAPI service providing:
#   - Real-time probabilistic flight risk inference
#   - High-throughput batch scoring
#   - Financial cost modeling & retention ROI calculation
#   - Data contract schema validation & distribution drift audits
#   - Relational workforce KPI aggregation
# ============================================================

import os
import sqlite3
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from api.schemas import (
    EmployeeInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ValidationResponse
)
from src.data_preprocessing import engineer_features
from src.data_validation import validate_single_record, validate_batch_dataframe
from src.roi_calculator import (
    calculate_flight_risk_exposure,
    calculate_retention_roi,
    compute_cohort_financial_impact
)
from src.drift_detection import compute_feature_drift

# ──────────────────────────────────────────────────────────
# Global Model & Preprocessor State
# ──────────────────────────────────────────────────────────
MODEL = None
PREPROCESSOR = None
FEATURE_NAMES = None
BASELINE_DF = None


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_asset_path(rel_path: str) -> str:
    candidates = [
        rel_path,
        os.path.join(PROJECT_ROOT, rel_path),
        os.path.join(os.getcwd(), "TalentPulse", rel_path),
        os.path.join(os.getcwd(), "employee-attrition-prediction", rel_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return rel_path


def load_assets():
    global MODEL, PREPROCESSOR, FEATURE_NAMES, BASELINE_DF
    try:
        model_path = _resolve_asset_path(os.path.join("models", "best_model.pkl"))
        preprocessor_path = _resolve_asset_path(os.path.join("models", "preprocessor.pkl"))
        features_path = _resolve_asset_path(os.path.join("data", "processed", "feature_names.pkl"))
        raw_data_path = _resolve_asset_path(os.path.join("data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv"))

        if os.path.exists(model_path):
            MODEL = joblib.load(model_path)
        if os.path.exists(preprocessor_path):
            PREPROCESSOR = joblib.load(preprocessor_path)
        if os.path.exists(features_path):
            FEATURE_NAMES = joblib.load(features_path)
        if os.path.exists(raw_data_path):
            BASELINE_DF = pd.read_csv(raw_data_path)
    except Exception as e:
        print(f"Warning: Could not preload all assets at startup: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_assets()
    yield


# ──────────────────────────────────────────────────────────
# Application Setup & Metadata
# ──────────────────────────────────────────────────────────
app = FastAPI(
    title="TalentPulse · Workforce Retention & Flight Risk API",
    description=(
        "Production-grade Machine Learning microservice for predictive employee turnover analysis, "
        "cost-sensitive financial risk exposure modeling, and data quality validation."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────
# Core Endpoints
# ──────────────────────────────────────────────────────────

@app.get("/", tags=["Overview"])
def root_overview():
    """Service landing endpoint with API documentation and status links."""
    return {
        "service": "TalentPulse Workforce Retention API",
        "version": "2.0.0",
        "documentation": "/docs",
        "alternative_docs": "/redoc",
        "health_check": "/health",
        "endpoints": {
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "validate_data": "POST /validate",
            "drift_audit": "POST /drift",
            "model_metrics": "GET /metrics",
            "workforce_kpis": "GET /analytics/kpis"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """Diagnostic health check verifying model artifact availability."""
    is_loaded = MODEL is not None and PREPROCESSOR is not None
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "model_name": type(MODEL).__name__ if MODEL is not None else "None",
        "feature_count": len(FEATURE_NAMES) if FEATURE_NAMES is not None else 0,
        "api_version": "2.0.0"
    }


@app.get("/metrics", tags=["Diagnostics"])
def get_metrics():
    """Return champion model evaluation metrics and comparison benchmarks."""
    results_path = os.path.join("models", "results.json")
    if os.path.exists(results_path):
        import json
        with open(results_path) as f:
            return {"benchmarks": json.load(f)}
    return {"message": "Benchmark results file not found."}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_employee(employee: EmployeeInput):
    """
    Generate real-time flight risk probability, financial exposure,
    and prescriptive retention guidance for a single employee.
    """
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded on server."
        )

    data = employee.model_dump() if hasattr(employee, "model_dump") else employee.dict()
    df_raw = pd.DataFrame([data])
    df_feat = engineer_features(df_raw)
    x_proc = PREPROCESSOR.transform(df_feat)

    prob = float(MODEL.predict_proba(x_proc)[0][1])
    pred = int(MODEL.predict(x_proc)[0])
    retention_conf = round((1.0 - prob) * 100.0, 1)

    # Risk tier classification
    if prob >= 0.65:
        tier = "Critical Flight Risk"
        directive = "Immediate priority 1-on-1 retention session; conduct salary equity audit and workload review."
    elif prob >= 0.40:
        tier = "Moderate Risk"
        directive = "Schedule career progression check-in; evaluate promotion velocity and overtime load."
    else:
        tier = "Stable Retention"
        directive = "Standard mentorship and engagement cadence."

    verdict = "Flight Risk Flagged" if pred == 1 else "Retention Anchor"

    # Financial Exposure & ROI Modeling
    exposure = calculate_flight_risk_exposure(prob, data["MonthlyIncome"])
    roi = calculate_retention_roi(prob, data["MonthlyIncome"])

    return {
        "probability": round(prob, 4),
        "retention_confidence": retention_conf,
        "prediction": pred,
        "verdict": verdict,
        "risk_tier": tier,
        "financial_exposure": exposure,
        "retention_roi": roi,
        "retention_directive": directive
    }


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch: BatchPredictionRequest):
    """
    Perform high-throughput batch scoring across an entire workforce cohort.
    Computes cohort-level financial capital flight exposure and net ROI.
    """
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded on server."
        )

    records = [
        r.model_dump() if hasattr(r, "model_dump") else r.dict()
        for r in batch.records
    ]
    if not records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch list must contain at least 1 record."
        )

    df_raw = pd.DataFrame(records)
    df_feat = engineer_features(df_raw.copy())
    x_proc = PREPROCESSOR.transform(df_feat)

    probs = MODEL.predict_proba(x_proc)[:, 1]
    preds = MODEL.predict(x_proc)

    df_raw["Attrition_Probability"] = probs.round(4)
    df_raw["Attrition_Prediction"] = preds

    def get_tier(p):
        if p >= 0.65: return "Critical Flight Risk"
        if p >= 0.40: return "Moderate Risk"
        return "Stable Retention"

    df_raw["Risk_Tier"] = [get_tier(p) for p in probs]

    high = int((df_raw["Risk_Tier"] == "Critical Flight Risk").sum())
    med = int((df_raw["Risk_Tier"] == "Moderate Risk").sum())
    low = int((df_raw["Risk_Tier"] == "Stable Retention").sum())

    financial_summary = compute_cohort_financial_impact(df_raw)

    return {
        "total_records": len(df_raw),
        "high_risk_count": high,
        "moderate_risk_count": med,
        "low_risk_count": low,
        "cohort_financial_summary": financial_summary,
        "predictions": df_raw[["Attrition_Probability", "Attrition_Prediction", "Risk_Tier", "MonthlyIncome"]].to_dict(orient="records")
    }


@app.post("/validate", response_model=ValidationResponse, tags=["Data Quality"])
def validate_record(employee: Dict[str, Any]):
    """
    Audit an employee dictionary against the formal schema contract.
    Identifies out-of-range boundaries, missing fields, or unrecognized categories.
    """
    is_valid, errors, sanitized = validate_single_record(employee)
    return {
        "is_valid": is_valid,
        "errors": errors,
        "sanitized_record": sanitized
    }


@app.post("/drift", tags=["Data Quality"])
def evaluate_drift(batch: BatchPredictionRequest):
    """
    Evaluate Population Stability Index (PSI) and Kolmogorov-Smirnov distribution
    shifts between the baseline training cohort and the submitted batch.
    """
    if BASELINE_DF is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Baseline training data is not available on server for drift comparison."
        )

    current_df = pd.DataFrame([
        r.model_dump() if hasattr(r, "model_dump") else r.dict()
        for r in batch.records
    ])
    drift_report = compute_feature_drift(BASELINE_DF, current_df)
    return drift_report


@app.get("/analytics/kpis", tags=["Workforce Analytics"])
def get_workforce_kpis():
    """Execute live SQL analytical queries returning top-line executive headcount and turnover KPIs."""
    raw_path = _resolve_asset_path(os.path.join("data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv"))
    if not os.path.exists(raw_path):
        raise HTTPException(status_code=404, detail="Raw workforce dataset not found.")

    df = pd.read_csv(raw_path)
    conn = sqlite3.connect(":memory:")
    df.to_sql("employees", conn, index=False, if_exists="replace")

    query = """
    SELECT 
        COUNT(*) AS total_headcount,
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS total_leavers,
        ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct,
        ROUND(AVG(MonthlyIncome), 0) AS company_avg_monthly_income,
        ROUND(AVG(TotalWorkingYears), 1) AS company_avg_tenure_years
    FROM employees;
    """
    res = pd.read_sql_query(query, conn).to_dict(orient="records")[0]
    conn.close()
    return res
