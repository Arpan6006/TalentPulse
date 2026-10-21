import pytest
import numpy as np
import pandas as pd
from src.drift_detection import calculate_psi, compute_feature_drift


def test_psi_identical_distributions():
    """Identical distributions should have a PSI near 0.0."""
    np.random.seed(42)
    expected = np.random.normal(loc=50, scale=10, size=1000)
    actual = np.random.normal(loc=50, scale=10, size=1000)
    psi = calculate_psi(expected, actual, num_bins=10)
    assert psi < 0.05, f"Expected near zero PSI for identical distributions, got {psi}"


def test_psi_shifted_distribution():
    """Substantially shifted distributions should produce PSI >= 0.20."""
    np.random.seed(42)
    expected = np.random.normal(loc=30, scale=5, size=1000)
    actual = np.random.normal(loc=65, scale=5, size=1000)
    psi = calculate_psi(expected, actual, num_bins=10)
    assert psi >= 0.20, f"Expected high PSI for major distribution shift, got {psi}"


def test_compute_feature_drift_numeric_and_categorical():
    """Validate full feature drift detection on sample DataFrames."""
    np.random.seed(42)
    baseline = pd.DataFrame({
        "MonthlyIncome": np.random.normal(loc=6500, scale=1000, size=500),
        "Age": np.random.normal(loc=37, scale=6, size=500),
        "OverTime": ["No"] * 360 + ["Yes"] * 140
    })

    # Similar distribution with minor natural sampling variation
    current_stable = pd.DataFrame({
        "MonthlyIncome": np.random.normal(loc=6520, scale=990, size=500),
        "Age": np.random.normal(loc=37.1, scale=5.9, size=500),
        "OverTime": ["No"] * 355 + ["Yes"] * 145
    })

    report = compute_feature_drift(baseline, current_stable, features=["MonthlyIncome", "Age", "OverTime"])
    assert "dataset_psi" in report
    assert "drift_level" in report
    assert report["drift_level"] in ["Stable", "Moderate Drift"]
    assert "MonthlyIncome" in report["feature_metrics"]
    assert "OverTime" in report["feature_metrics"]


def test_compute_feature_drift_critical_shift():
    """Severely shifted distributions should trigger Drift Detected."""
    baseline = pd.DataFrame({
        "MonthlyIncome": np.random.normal(loc=3000, scale=300, size=200),
        "Age": np.random.normal(loc=25, scale=2, size=200),
        "OverTime": ["No"] * 180 + ["Yes"] * 20
    })

    current_drifted = pd.DataFrame({
        "MonthlyIncome": np.random.normal(loc=15000, scale=1000, size=200),
        "Age": np.random.normal(loc=55, scale=3, size=200),
        "OverTime": ["Yes"] * 180 + ["No"] * 20
    })

    report = compute_feature_drift(baseline, current_drifted, features=["MonthlyIncome", "Age", "OverTime"])
    assert report["drift_level"] in ["Moderate Drift", "Significant Drift"]
    assert report["feature_metrics"]["MonthlyIncome"]["status"] == "Drift Detected"
