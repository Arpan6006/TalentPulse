# ============================================================
# Employee Attrition Prediction - Data Quality & Contract Validation
# ============================================================
# Production data integrity layer ensuring:
#   - Strict schema enforcement (field presence and data types)
#   - Value domain and boundary constraints
#   - Categorical vocabulary verification
#   - Automated missing value detection and imputation audits
#   - Data quality scorecard generation for batch workloads
# ============================================================

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


# Schema Contract Specification
SCHEMA_CONTRACT = {
    "Age": {"type": "int", "min": 18, "max": 65, "default": 35},
    "BusinessTravel": {"type": "category", "allowed": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], "default": "Travel_Rarely"},
    "DailyRate": {"type": "numeric", "min": 100, "max": 1500, "default": 800},
    "Department": {"type": "category", "allowed": ["Research & Development", "Sales", "Human Resources"], "default": "Research & Development"},
    "DistanceFromHome": {"type": "numeric", "min": 1, "max": 50, "default": 8},
    "Education": {"type": "int", "min": 1, "max": 5, "default": 3},
    "EducationField": {"type": "category", "allowed": ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"], "default": "Life Sciences"},
    "EnvironmentSatisfaction": {"type": "int", "min": 1, "max": 4, "default": 3},
    "Gender": {"type": "category", "allowed": ["Male", "Female"], "default": "Male"},
    "HourlyRate": {"type": "numeric", "min": 30, "max": 120, "default": 65},
    "JobInvolvement": {"type": "int", "min": 1, "max": 4, "default": 3},
    "JobLevel": {"type": "int", "min": 1, "max": 5, "default": 2},
    "JobRole": {
        "type": "category",
        "allowed": [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"
        ],
        "default": "Research Scientist"
    },
    "JobSatisfaction": {"type": "int", "min": 1, "max": 4, "default": 3},
    "MaritalStatus": {"type": "category", "allowed": ["Single", "Married", "Divorced"], "default": "Married"},
    "MonthlyIncome": {"type": "numeric", "min": 1000, "max": 25000, "default": 5000},
    "MonthlyRate": {"type": "numeric", "min": 1000, "max": 30000, "default": 14000},
    "NumCompaniesWorked": {"type": "int", "min": 0, "max": 12, "default": 2},
    "OverTime": {"type": "category", "allowed": ["Yes", "No"], "default": "No"},
    "PercentSalaryHike": {"type": "numeric", "min": 10, "max": 30, "default": 15},
    "PerformanceRating": {"type": "int", "min": 1, "max": 4, "default": 3},
    "RelationshipSatisfaction": {"type": "int", "min": 1, "max": 4, "default": 3},
    "StockOptionLevel": {"type": "int", "min": 0, "max": 3, "default": 1},
    "TotalWorkingYears": {"type": "numeric", "min": 0, "max": 45, "default": 8},
    "TrainingTimesLastYear": {"type": "int", "min": 0, "max": 10, "default": 2},
    "WorkLifeBalance": {"type": "int", "min": 1, "max": 4, "default": 3},
    "YearsAtCompany": {"type": "numeric", "min": 0, "max": 45, "default": 4},
    "YearsInCurrentRole": {"type": "numeric", "min": 0, "max": 25, "default": 2},
    "YearsSinceLastPromotion": {"type": "numeric", "min": 0, "max": 20, "default": 1},
    "YearsWithCurrManager": {"type": "numeric", "min": 0, "max": 20, "default": 2},
}


def validate_single_record(record: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate a single employee input dictionary against schema rules.
    Returns:
        (is_valid, error_list, sanitized_record)
    """
    errors = []
    sanitized = {}

    for field, rule in SCHEMA_CONTRACT.items():
        val = record.get(field)

        # Missing check
        if val is None or (isinstance(val, float) and np.isnan(val)):
            errors.append(f"Missing mandatory field '{field}', defaulted to {rule['default']}")
            sanitized[field] = rule["default"]
            continue

        # Categorical check
        if rule["type"] == "category":
            str_val = str(val).strip()
            if str_val not in rule["allowed"]:
                errors.append(f"Invalid category '{val}' for field '{field}'. Allowed: {rule['allowed']}")
                sanitized[field] = rule["default"]
            else:
                sanitized[field] = str_val

        # Numeric bounds check
        elif rule["type"] in ("int", "numeric"):
            try:
                num_val = int(val) if rule["type"] == "int" else float(val)
                if num_val < rule["min"] or num_val > rule["max"]:
                    errors.append(
                        f"Field '{field}' value {num_val} out of expected range [{rule['min']}, {rule['max']}]. Clipped."
                    )
                    num_val = max(rule["min"], min(rule["max"], num_val))
                sanitized[field] = num_val
            except (ValueError, TypeError):
                errors.append(f"Non-numeric value '{val}' for field '{field}'. Replaced with default.")
                sanitized[field] = rule["default"]

    is_valid = len([e for e in errors if "out of expected range" in e or "Invalid category" in e]) == 0
    return is_valid, errors, sanitized


def validate_batch_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and sanitize an entire uploaded CSV DataFrame.
    Returns:
        (cleaned_df, audit_report_dict)
    """
    df_clean = df.copy()
    total_records = len(df_clean)
    missing_fields = []
    out_of_bounds_counts = {}
    invalid_categories = {}
    null_counts = {}

    # Check for missing required columns
    for field, rule in SCHEMA_CONTRACT.items():
        if field not in df_clean.columns:
            missing_fields.append(field)
            df_clean[field] = rule["default"]
        else:
            # Check for nulls
            null_count = int(df_clean[field].isnull().sum())
            if null_count > 0:
                null_counts[field] = null_count
                df_clean[field] = df_clean[field].fillna(rule["default"])

            # Check categorical validity
            if rule["type"] == "category":
                invalid_mask = ~df_clean[field].astype(str).str.strip().isin(rule["allowed"])
                num_invalid = int(invalid_mask.sum())
                if num_invalid > 0:
                    invalid_categories[field] = num_invalid
                    df_clean.loc[invalid_mask, field] = rule["default"]

            # Check numeric boundaries
            elif rule["type"] in ("int", "numeric"):
                df_clean[field] = pd.to_numeric(df_clean[field], errors="coerce").fillna(rule["default"])
                oob_mask = (df_clean[field] < rule["min"]) | (df_clean[field] > rule["max"])
                num_oob = int(oob_mask.sum())
                if num_oob > 0:
                    out_of_bounds_counts[field] = num_oob
                    df_clean[field] = df_clean[field].clip(lower=rule["min"], upper=rule["max"])

    # Compute overall quality score (0 to 100%)
    total_checks = total_records * len(SCHEMA_CONTRACT)
    total_clippings = sum(out_of_bounds_counts.values())
    total_violations = sum(null_counts.values()) + sum(invalid_categories.values()) + total_clippings + (len(missing_fields) * total_records)
    quality_score = max(0.0, round(100.0 * (1.0 - (total_violations / max(1, total_checks))), 2))
    flagged_records_count = sum(null_counts.values()) + sum(invalid_categories.values()) + total_clippings

    audit_report = {
        "total_records": total_records,
        "quality_score_pct": quality_score,
        "data_quality_score": quality_score,
        "missing_columns_synthesized": missing_fields,
        "null_imputations": null_counts,
        "invalid_categories_normalized": invalid_categories,
        "out_of_bounds_clipped": out_of_bounds_counts,
        "total_boundary_clippings": total_clippings,
        "flagged_records": flagged_records_count,
        "is_valid": quality_score >= 80.0,
        "is_production_ready": quality_score >= 80.0
    }

    return df_clean, audit_report
