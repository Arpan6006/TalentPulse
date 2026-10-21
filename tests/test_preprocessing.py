import pytest
import pandas as pd
from src.data_preprocessing import engineer_features, COLS_TO_DROP, CATEGORICAL_COLS, NUMERICAL_COLS


def test_engineer_features_creates_expected_columns():
    """Verify that domain feature engineering adds all 5 derived features."""
    sample_df = pd.DataFrame({
        "Age": [35, 45],
        "TotalWorkingYears": [10, 20],
        "YearsAtCompany": [6, 10],
        "YearsInCurrentRole": [3, 5],
        "YearsSinceLastPromotion": [1, 4],
        "MonthlyIncome": [6000, 10000]
    })

    result_df = engineer_features(sample_df)

    expected_new_cols = [
        "YearsAtOtherCompanies",
        "IncomePerYearExp",
        "RoleStability",
        "PromotionGap",
        "AgeWhenStarted"
    ]
    for col in expected_new_cols:
        assert col in result_df.columns, f"Engineered column {col} was not created"

    # Math validations
    # YearsAtOtherCompanies = TotalWorkingYears - YearsAtCompany (10 - 6 = 4)
    assert result_df["YearsAtOtherCompanies"].iloc[0] == 4
    # IncomePerYearExp = MonthlyIncome / (TotalWorkingYears + 1) (6000 / 11 = 545.45...)
    assert round(result_df["IncomePerYearExp"].iloc[0], 2) == round(6000 / 11, 2)
    # RoleStability = YearsInCurrentRole / (YearsAtCompany + 1) (3 / 7)
    assert round(result_df["RoleStability"].iloc[0], 2) == round(3 / 7, 2)
    # PromotionGap = YearsAtCompany - YearsSinceLastPromotion (6 - 1 = 5)
    assert result_df["PromotionGap"].iloc[0] == 5
    # AgeWhenStarted = Age - TotalWorkingYears (35 - 10 = 25)
    assert result_df["AgeWhenStarted"].iloc[0] == 25


def test_constants_definitions():
    """Verify that constant lists are properly maintained."""
    assert "EmployeeCount" in COLS_TO_DROP
    assert "StandardHours" in COLS_TO_DROP
    assert "OverTime" in CATEGORICAL_COLS
    assert "MonthlyIncome" in NUMERICAL_COLS
