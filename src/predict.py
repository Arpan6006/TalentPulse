# ============================================================
# Employee Attrition Prediction - Inference / Prediction Module
# ============================================================
# Use this for:
#   - Single employee prediction (dict → probability)
#   - Batch prediction (CSV file → CSV with predictions)
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import prepare_single_employee, engineer_features

# ──────────────────────────────────────────────────────────
# Default paths (relative to project root)
# ──────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH      = os.path.join("models", "best_model.pkl")
DEFAULT_PREPROCESSOR    = os.path.join("models", "preprocessor.pkl")
DEFAULT_EXPLAINER_PATH  = os.path.join("models", "shap_explainer.pkl")
DEFAULT_FEATURE_NAMES   = os.path.join("data", "processed", "feature_names.pkl")


# ──────────────────────────────────────────────────────────
# Load artifacts (cached)
# ──────────────────────────────────────────────────────────

def load_artifacts(
    model_path:      str = DEFAULT_MODEL_PATH,
    preprocessor_path: str = DEFAULT_PREPROCESSOR,
    explainer_path:  str = DEFAULT_EXPLAINER_PATH,
    feature_names_path: str = DEFAULT_FEATURE_NAMES,
):
    """
    Load and return model, preprocessor, SHAP explainer, feature names.
    All paths are relative to the project root.
    """
    model         = joblib.load(model_path)
    preprocessor  = joblib.load(preprocessor_path)
    explainer     = joblib.load(explainer_path)
    feature_names = joblib.load(feature_names_path)
    return model, preprocessor, explainer, feature_names


# ──────────────────────────────────────────────────────────
# Risk Classification
# ──────────────────────────────────────────────────────────

def classify_risk(probability: float) -> tuple[str, str]:
    """
    Convert a raw attrition probability into a human-readable
    risk level and recommended HR action.

    Returns:
        (risk_label, advice)
    """
    if probability >= 0.65:
        return (
            "🔴 High Risk",
            "Immediate attention recommended. "
            "Consider 1-on-1 retention conversation, salary review, "
            "or role change."
        )
    elif probability >= 0.40:
        return (
            "🟠 Medium Risk",
            "Monitor closely. Schedule a check-in and explore "
            "career development opportunities."
        )
    else:
        return (
            "🟢 Low Risk",
            "Employee appears stable. Continue normal engagement "
            "and recognition."
        )


# ──────────────────────────────────────────────────────────
# Single Prediction
# ──────────────────────────────────────────────────────────

def predict_single(
    input_dict: dict,
    model=None,
    preprocessor=None,
) -> dict:
    """
    Predict attrition risk for a single employee.

    Args:
        input_dict:   Raw HR feature dictionary (before preprocessing).
        model:        Trained classifier (loaded externally or auto-loaded).
        preprocessor: Fitted ColumnTransformer (auto-loaded if None).

    Returns:
        Dict with keys: probability, prediction, risk_level, advice.
    """
    if model is None or preprocessor is None:
        model, preprocessor, _, _ = load_artifacts()

    X = prepare_single_employee(input_dict, preprocessor)
    probability = float(model.predict_proba(X)[0][1])
    prediction  = int(model.predict(X)[0])
    risk_level, advice = classify_risk(probability)

    return {
        "probability": round(probability, 4),
        "prediction":  "Will Leave" if prediction == 1 else "Will Stay",
        "risk_level":  risk_level,
        "advice":      advice,
    }


# ──────────────────────────────────────────────────────────
# Batch Prediction (CSV → CSV)
# ──────────────────────────────────────────────────────────

def predict_batch(
    input_csv_path:  str,
    output_csv_path: str = "batch_predictions.csv",
    model=None,
    preprocessor=None,
) -> pd.DataFrame:
    """
    Run predictions on a CSV file of employee records.

    The CSV must contain the same columns as the IBM HR dataset
    (minus Attrition). Four extra engineered columns are added
    automatically.

    Args:
        input_csv_path:  Path to input CSV.
        output_csv_path: Path to save the results CSV.
        model:           Trained classifier (auto-loaded if None).
        preprocessor:    Fitted ColumnTransformer (auto-loaded if None).

    Returns:
        DataFrame with original columns + Attrition_Probability,
        Attrition_Prediction, and Risk_Level columns appended.
    """
    if model is None or preprocessor is None:
        model, preprocessor, _, _ = load_artifacts()

    df = pd.read_csv(input_csv_path)
    df_feat = engineer_features(df.copy())

    X_processed  = preprocessor.transform(df_feat)
    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions   = model.predict(X_processed)

    df["Attrition_Probability"] = probabilities.round(4)
    df["Attrition_Prediction"]  = ["Will Leave" if p == 1 else "Will Stay" for p in predictions]
    df["Risk_Level"]            = [classify_risk(p)[0] for p in probabilities]

    df.to_csv(output_csv_path, index=False)
    print(f"✅ Batch predictions saved → {output_csv_path}")
    print(f"   Total employees: {len(df)}")
    print(f"   High Risk:    {(df['Risk_Level'].str.startswith('🔴')).sum()}")
    print(f"   Medium Risk:  {(df['Risk_Level'].str.startswith('🟠')).sum()}")
    print(f"   Low Risk:     {(df['Risk_Level'].str.startswith('🟢')).sum()}")

    return df


# ──────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Batch:  python src/predict.py <input_csv> [output_csv]")
        print("  Single: python src/predict.py --demo")
        sys.exit(1)

    if sys.argv[1] == "--demo":
        demo_employee = {
            "Age": 32, "BusinessTravel": "Travel_Rarely", "DailyRate": 800,
            "Department": "Sales", "DistanceFromHome": 15, "Education": 3,
            "EducationField": "Life Sciences", "EnvironmentSatisfaction": 2,
            "Gender": "Male", "HourlyRate": 65, "JobInvolvement": 2,
            "JobLevel": 2, "JobRole": "Sales Executive", "JobSatisfaction": 1,
            "MaritalStatus": "Single", "MonthlyIncome": 4000, "MonthlyRate": 14000,
            "NumCompaniesWorked": 5, "OverTime": "Yes", "PercentSalaryHike": 11,
            "PerformanceRating": 3, "RelationshipSatisfaction": 2,
            "StockOptionLevel": 0, "TotalWorkingYears": 8, "TrainingTimesLastYear": 2,
            "WorkLifeBalance": 1, "YearsAtCompany": 3, "YearsInCurrentRole": 1,
            "YearsSinceLastPromotion": 2, "YearsWithCurrManager": 1,
        }
        result = predict_single(demo_employee)
        print(json.dumps(result, indent=2))
    else:
        input_csv  = sys.argv[1]
        output_csv = sys.argv[2] if len(sys.argv) > 2 else "batch_predictions.csv"
        predict_batch(input_csv, output_csv)
