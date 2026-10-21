import pytest
import pandas as pd
import numpy as np
from src.data_validation import validate_single_record, validate_batch_dataframe, SCHEMA_CONTRACT


def test_schema_contract_completeness():
    assert "MonthlyIncome" in SCHEMA_CONTRACT
    assert "Age" in SCHEMA_CONTRACT
    assert "OverTime" in SCHEMA_CONTRACT
    assert len(SCHEMA_CONTRACT) >= 30


def test_validate_single_record_valid():
    sample = {
        "Age": 35,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 800.0,
        "Department": "Sales",
        "DistanceFromHome": 10.0,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 3,
        "Gender": "Female",
        "HourlyRate": 70.0,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Sales Executive",
        "JobSatisfaction": 4,
        "MaritalStatus": "Married",
        "MonthlyIncome": 6000.0,
        "MonthlyRate": 15000.0,
        "NumCompaniesWorked": 2,
        "OverTime": "No",
        "PercentSalaryHike": 14.0,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 10.0,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5.0,
        "YearsInCurrentRole": 3.0,
        "YearsSinceLastPromotion": 1.0,
        "YearsWithCurrManager": 3.0
    }
    is_valid, errors, sanitized = validate_single_record(sample)
    assert is_valid is True
    assert len(errors) == 0
    assert sanitized["Age"] == 35
    assert sanitized["MonthlyIncome"] == 6000.0


def test_validate_single_record_invalid_and_clipped():
    invalid_sample = {
        "Age": 12,  # Below min 18 -> clipped
        "OverTime": "UnknownValue",  # Invalid category -> defaulted
        "MonthlyIncome": -500.0  # Below min 1000 -> clipped
    }
    is_valid, errors, sanitized = validate_single_record(invalid_sample)
    assert is_valid is False
    assert len(errors) > 0
    assert sanitized["Age"] == 18
    assert sanitized["OverTime"] == "No"
    assert sanitized["MonthlyIncome"] == 1000.0


def test_validate_batch_dataframe():
    df_sample = pd.DataFrame([
        {"Age": 30, "MonthlyIncome": 4000, "OverTime": "Yes"},
        {"Age": 75, "MonthlyIncome": None, "OverTime": "InvalidCategory"}  # 75 > 65
    ])
    clean_df, report = validate_batch_dataframe(df_sample)
    assert len(clean_df) == 2
    assert report["total_records"] == 2
    assert "MonthlyIncome" in report["null_imputations"]
    assert clean_df["Age"].max() <= 65
