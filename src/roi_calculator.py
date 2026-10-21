# ============================================================
# Employee Attrition Prediction - Financial ROI & Cost Modeling
# ============================================================
# Translates probabilistic ML inferences into executive financial metrics:
#   - Employee replacement cost estimation (SHRM benchmark: 1.5x annual salary)
#   - Expected financial flight risk exposure
#   - Retention intervention cost-benefit & net ROI calculation
#   - Cost-sensitive decision threshold optimization
# ============================================================

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


# Industry benchmarks (SHRM / Gallup retention cost research)
DEFAULT_SALARY_MULTIPLIER = 1.5      # Cost to replace a departed employee
DEFAULT_INTERVENTION_COST = 3000.0   # Cost of retention action (stay bonus, equity bump, perk)
DEFAULT_RETENTION_SUCCESS_RATE = 0.70  # Success rate of timely intervention


def calculate_replacement_cost(
    monthly_income: float,
    multiplier: float = DEFAULT_SALARY_MULTIPLIER
) -> float:
    """
    Compute estimated replacement cost for an employee.
    Formula: MonthlyIncome * 12 * Multiplier
    Accounts for recruiting fees, onboarding, ramp-up lag, and lost departmental velocity.
    """
    annual_salary = float(monthly_income) * 12.0
    return round(annual_salary * multiplier, 2)


def calculate_flight_risk_exposure(
    probability: float,
    monthly_income: float,
    multiplier: float = DEFAULT_SALARY_MULTIPLIER
) -> Dict[str, float]:
    """
    Calculate expected financial capital exposure for a single employee.
    
    Returns:
        Dict with annual_salary, replacement_cost, expected_financial_loss
    """
    annual_salary = float(monthly_income) * 12.0
    replacement_cost = calculate_replacement_cost(monthly_income, multiplier)
    expected_loss = round(float(probability) * replacement_cost, 2)

    return {
        "annual_salary": round(annual_salary, 2),
        "replacement_cost": replacement_cost,
        "expected_loss": expected_loss
    }


def calculate_retention_roi(
    probability: float,
    monthly_income: float,
    intervention_cost: float = DEFAULT_INTERVENTION_COST,
    success_rate: float = DEFAULT_RETENTION_SUCCESS_RATE,
    multiplier: float = DEFAULT_SALARY_MULTIPLIER
) -> Dict[str, Any]:
    """
    Calculate the net financial return and cost avoidance of intervening
    to retain an employee flagged as a flight risk.

    Returns:
        Dict containing expected_loss, intervention_cost, gross_loss_avoided,
        net_savings, and roi_percentage.
    """
    exposure = calculate_flight_risk_exposure(probability, monthly_income, multiplier)
    replacement_cost = exposure["replacement_cost"]
    expected_loss = exposure["expected_loss"]

    # Gross savings if intervention succeeds in saving the expected loss
    gross_loss_avoided = round(expected_loss * float(success_rate), 2)
    net_savings = round(gross_loss_avoided - float(intervention_cost), 2)
    
    roi_pct = round((net_savings / float(intervention_cost)) * 100.0, 1) if intervention_cost > 0 else 0.0
    roi_multiple = round(gross_loss_avoided / float(intervention_cost), 2) if intervention_cost > 0 else 0.0
    break_even_prob = round(float(intervention_cost) / (replacement_cost * float(success_rate)), 4) if (replacement_cost * success_rate) > 0 else 0.0

    return {
        "annual_salary": exposure["annual_salary"],
        "replacement_cost": replacement_cost,
        "expected_loss": expected_loss,
        "intervention_cost": round(float(intervention_cost), 2),
        "gross_loss_avoided": gross_loss_avoided,
        "gross_cost_avoidance": gross_loss_avoided,
        "net_savings": net_savings,
        "net_cost_avoidance": net_savings,
        "roi_percentage": roi_pct,
        "roi_multiple": roi_multiple,
        "break_even_probability": break_even_prob,
        "is_profitable": net_savings > 0
    }


def compute_cohort_financial_impact(
    df: pd.DataFrame,
    probability_col: str = "Attrition_Probability",
    income_col: str = "MonthlyIncome",
    intervention_cost: float = DEFAULT_INTERVENTION_COST,
    success_rate: float = DEFAULT_RETENTION_SUCCESS_RATE,
    multiplier: float = DEFAULT_SALARY_MULTIPLIER
) -> Dict[str, Any]:
    """
    Aggregate workforce-wide financial exposure and potential retention ROI.
    """
    if df.empty or probability_col not in df.columns or income_col not in df.columns:
        return {
            "total_financial_exposure": 0.0,
            "total_loss_exposure": 0.0,
            "total_replacement_cost_at_risk": 0.0,
            "estimated_cohort_roi_multiple": 0.0,
            "cohort_size": 0
        }

    annual_salaries = df[income_col] * 12.0
    replacement_costs = annual_salaries * multiplier
    expected_losses = df[probability_col] * replacement_costs

    total_exposure = round(float(expected_losses.sum()), 2)
    total_payroll = round(float(annual_salaries.sum()), 2)
    avg_replacement_cost = round(float(replacement_costs.mean()), 2)

    # Filter employees recommended for intervention (e.g. probability >= 0.40)
    flagged_mask = df[probability_col] >= 0.40
    flagged_count = int(flagged_mask.sum())
    
    flagged_exposure = float(expected_losses[flagged_mask].sum())
    flagged_replacement = float(replacement_costs[flagged_mask].sum()) if flagged_count > 0 else 0.0
    total_intervention_budget = flagged_count * float(intervention_cost)
    projected_loss_avoided = flagged_exposure * float(success_rate)
    net_projected_savings = round(projected_loss_avoided - total_intervention_budget, 2)
    
    net_roi_pct = round((net_projected_savings / total_intervention_budget) * 100.0, 1) if total_intervention_budget > 0 else 0.0
    roi_multiple = round(projected_loss_avoided / total_intervention_budget, 2) if total_intervention_budget > 0 else 0.0

    return {
        "cohort_size": len(df),
        "total_annual_payroll": total_payroll,
        "avg_replacement_cost": avg_replacement_cost,
        "total_financial_exposure": total_exposure,
        "total_loss_exposure": total_exposure,
        "total_replacement_cost_at_risk": round(flagged_replacement, 2),
        "flagged_at_risk_count": flagged_count,
        "total_intervention_budget": round(total_intervention_budget, 2),
        "projected_loss_avoided": round(projected_loss_avoided, 2),
        "net_projected_savings": net_projected_savings,
        "net_roi_percentage": net_roi_pct,
        "estimated_cohort_roi_multiple": roi_multiple
    }


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_false_negative: float = 75000.0,   # Cost of missing an actual leaver (unplanned vacancy)
    cost_false_positive: float = 2000.0,    # Cost of unnecessary retention perks
    cost_true_positive: float = 3000.0      # Cost of planned retention intervention
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Calculate the business cost-optimal decision threshold tau* that minimizes total expected cost:
        Total Cost = FN * C_FN + FP * C_FP + TP * C_TP
    
    Returns:
        (optimal_threshold, min_cost, threshold_evaluation_summary)
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    costs = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = np.sum((y_true == 1) & (preds == 1))
        fp = np.sum((y_true == 0) & (preds == 1))
        fn = np.sum((y_true == 1) & (preds == 0))
        tn = np.sum((y_true == 0) & (preds == 0))

        cost = (fn * cost_false_negative) + (fp * cost_false_positive) + (tp * cost_true_positive)
        costs.append(cost)

    best_idx = int(np.argmin(costs))
    optimal_tau = round(float(thresholds[best_idx]), 3)
    min_cost = round(float(costs[best_idx]), 2)
    
    # Baseline cost if using naive 0.50 cutoff
    default_preds = (y_prob >= 0.50).astype(int)
    default_cost = float(
        np.sum((y_true == 1) & (default_preds == 0)) * cost_false_negative +
        np.sum((y_true == 0) & (default_preds == 1)) * cost_false_positive +
        np.sum((y_true == 1) & (default_preds == 1)) * cost_true_positive
    )
    cost_savings = round(default_cost - min_cost, 2)

    return optimal_tau, min_cost, {
        "optimal_threshold": optimal_tau,
        "minimized_total_cost": min_cost,
        "default_threshold_cost": default_cost,
        "cost_savings_vs_default": cost_savings
    }
