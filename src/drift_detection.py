# ============================================================
# Employee Attrition Prediction - Population Drift & Covariate Shift
# ============================================================
# Production MLOps monitoring module:
#   - Population Stability Index (PSI) calculation for continuous & discrete features
#   - Two-sample Kolmogorov-Smirnov (KS) statistical test for distribution shift
#   - Categorical distribution shift analysis
#   - Automated drift alert level classification (Stable / Moderate / Critical)
# ============================================================

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10
) -> float:
    """
    Calculate Population Stability Index (PSI) between baseline (expected)
    and newly observed (actual) distributions.
    
    Industry Interpretation:
        PSI < 0.10: No significant distribution change (Stable)
        0.10 <= PSI < 0.20: Moderate distribution shift (Monitor)
        PSI >= 0.20: Significant distribution drift (Recalibration Required)
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Quantile bin edges based on expected baseline
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) <= 2:
        return 0.0

    # Count frequencies
    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to fractions with small epsilon to prevent log(0) or div by zero
    eps = 1e-4
    exp_pct = (exp_counts + eps) / (np.sum(exp_counts) + eps * len(exp_counts))
    act_pct = (act_counts + eps) / (np.sum(act_counts) + eps * len(act_counts))

    # PSI Formula
    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(round(max(0.0, psi_val), 4))


def compute_feature_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluate feature-level and dataset-level drift between baseline training cohort
    and a newly ingested production batch.
    
    Returns:
        Dict with feature_drift_scores, dataset_psi, drift_level, alert_badge
    """
    if features is None:
        features = [
            "MonthlyIncome", "Age", "TotalWorkingYears", "YearsAtCompany",
            "YearsSinceLastPromotion", "DistanceFromHome", "OverTime",
            "JobSatisfaction", "WorkLifeBalance", "PercentSalaryHike"
        ]

    feature_scores = {}
    psi_values = []

    for feat in features:
        if feat not in baseline_df.columns or feat not in current_df.columns:
            continue

        base_col = baseline_df[feat]
        curr_col = current_df[feat]

        # Numeric features
        if pd.api.types.is_numeric_dtype(base_col):
            b_arr = base_col.dropna().values.astype(float)
            c_arr = curr_col.dropna().values.astype(float)

            psi = calculate_psi(b_arr, c_arr)
            ks_stat, ks_pval = stats.ks_2samp(b_arr, c_arr)

            status = "Stable"
            if psi >= 0.20 or ks_pval < 0.01:
                status = "Drift Detected"
            elif psi >= 0.10 or ks_pval < 0.05:
                status = "Moderate Shift"

            feature_scores[feat] = {
                "type": "numeric",
                "psi": psi,
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pval), 4),
                "status": status
            }
            psi_values.append(psi)

        # Categorical features
        else:
            base_counts = base_col.value_counts(normalize=True)
            curr_counts = curr_col.value_counts(normalize=True)
            
            all_cats = list(set(base_counts.index).union(set(curr_counts.index)))
            eps = 1e-4
            b_probs = np.array([base_counts.get(c, eps) for c in all_cats])
            c_probs = np.array([curr_counts.get(c, eps) for c in all_cats])
            
            b_probs = b_probs / np.sum(b_probs)
            c_probs = c_probs / np.sum(c_probs)
            
            cat_psi = float(round(np.sum((c_probs - b_probs) * np.log(c_probs / b_probs)), 4))
            
            status = "Stable"
            if cat_psi >= 0.20:
                status = "Drift Detected"
            elif cat_psi >= 0.10:
                status = "Moderate Shift"

            feature_scores[feat] = {
                "type": "categorical",
                "psi": cat_psi,
                "status": status
            }
            psi_values.append(cat_psi)

    dataset_psi = float(round(np.mean(psi_values) if psi_values else 0.0, 4))

    if dataset_psi >= 0.20:
        drift_level = "Significant Drift"
        alert_badge = "CRITICAL COVARIATE SHIFT"
        recommendation = "Data distribution differs significantly from baseline. Model recalibration is recommended."
    elif dataset_psi >= 0.10:
        drift_level = "Moderate Drift"
        alert_badge = "MODERATE DRIFT WARNING"
        recommendation = "Minor distribution shifts detected in key features. Continue monitoring inference outputs."
    else:
        drift_level = "Stable"
        alert_badge = "DISTRIBUTION STABLE"
        recommendation = "Production distribution aligns with baseline training distribution. Predictions are highly reliable."

    return {
        "dataset_psi": dataset_psi,
        "drift_level": drift_level,
        "alert_badge": alert_badge,
        "recommendation": recommendation,
        "feature_metrics": feature_scores
    }
