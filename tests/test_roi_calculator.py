import pytest
import numpy as np
import pandas as pd
from src.roi_calculator import (
    calculate_replacement_cost,
    calculate_flight_risk_exposure,
    calculate_retention_roi,
    compute_cohort_financial_impact,
    find_cost_optimal_threshold
)


def test_replacement_cost():
    # $5,000 monthly income * 12 * 1.5 = $90,000
    cost = calculate_replacement_cost(5000.0, multiplier=1.5)
    assert cost == 90000.0


def test_flight_risk_exposure():
    # $5,000/mo * 12 * 1.5 = $90,000 replacement cost. At 50% risk, loss is $45,000
    exposure = calculate_flight_risk_exposure(0.50, 5000.0, multiplier=1.5)
    assert exposure["replacement_cost"] == 90000.0
    assert exposure["expected_loss"] == 45000.0


def test_retention_roi():
    # $90k replacement cost, 80% flight risk -> $72k expected loss
    # 70% retention success -> $50.4k gross loss avoided
    # $3k intervention cost -> $47.4k net savings
    roi = calculate_retention_roi(
        probability=0.80,
        monthly_income=5000.0,
        intervention_cost=3000.0,
        success_rate=0.70
    )
    assert roi["is_profitable"] is True
    assert roi["net_savings"] > 40000.0
    assert roi["roi_percentage"] > 1000.0


def test_compute_cohort_financial_impact():
    df = pd.DataFrame([
        {"MonthlyIncome": 5000, "Attrition_Probability": 0.80},
        {"MonthlyIncome": 10000, "Attrition_Probability": 0.10},
        {"MonthlyIncome": 6000, "Attrition_Probability": 0.70}
    ])
    impact = compute_cohort_financial_impact(df)
    assert impact["cohort_size"] == 3
    assert impact["flagged_at_risk_count"] == 2
    assert impact["total_financial_exposure"] > 0


def test_find_cost_optimal_threshold():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.05, 0.02])

    optimal_tau, min_cost, summary = find_cost_optimal_threshold(
        y_true, y_prob,
        cost_false_negative=75000.0,
        cost_false_positive=2000.0,
        cost_true_positive=3000.0
    )
    assert 0.10 <= optimal_tau <= 0.90
    assert min_cost > 0
    assert "cost_savings_vs_default" in summary
