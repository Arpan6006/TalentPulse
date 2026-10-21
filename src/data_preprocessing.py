# ============================================================
# Employee Attrition Prediction - Data Preprocessing Module
# ============================================================
# Extracted from: notebooks/02_Preprocessing.ipynb
# This module is the reusable, production version of the
# preprocessing pipeline used in the Jupyter notebooks.
# ============================================================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE


# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

# Columns to drop (constant values or IDs — no predictive power)
COLS_TO_DROP = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

# Categorical features to one-hot encode
CATEGORICAL_COLS = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime"
]

# Numerical features to scale
NUMERICAL_COLS = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
    # Engineered features
    "YearsAtOtherCompanies", "IncomePerYearExp", "RoleStability",
    "PromotionGap", "AgeWhenStarted"
]


# ──────────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that improve model performance.

    Features created:
    - YearsAtOtherCompanies: Experience outside this company
    - IncomePerYearExp:       Salary efficiency ratio
    - RoleStability:          How long in current role vs. tenure
    - PromotionGap:           Years without a promotion
    - AgeWhenStarted:         Age when career began
    """
    df = df.copy()
    df["YearsAtOtherCompanies"] = df["TotalWorkingYears"] - df["YearsAtCompany"]
    df["IncomePerYearExp"]       = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
    df["RoleStability"]          = df["YearsInCurrentRole"] / (df["YearsAtCompany"] + 1)
    df["PromotionGap"]           = df["YearsAtCompany"] - df["YearsSinceLastPromotion"]
    df["AgeWhenStarted"]         = df["Age"] - df["TotalWorkingYears"]
    return df


# ──────────────────────────────────────────────────────────
# Full Preprocessing Pipeline
# ──────────────────────────────────────────────────────────

def preprocess_data(
    data_path: str,
    output_dir: str = "data/processed",
    model_dir: str  = "models",
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True
):
    """
    Full preprocessing pipeline:
      1. Load raw CSV
      2. Drop useless columns
      3. Encode target (Attrition Yes/No → 1/0)
      4. Engineer new features
      5. Train/test split (stratified)
      6. Fit ColumnTransformer (StandardScaler + OneHotEncoder)
      7. Apply SMOTE to balance training set
      8. Save processed arrays and artifacts

    Args:
        data_path:    Path to the raw IBM HR CSV file.
        output_dir:   Where to save processed .npy and .pkl files.
        model_dir:    Where to save the fitted preprocessor.pkl.
        test_size:    Fraction of data to hold out for testing.
        random_state: RNG seed for reproducibility.
        use_smote:    Whether to oversample the minority class with SMOTE.

    Returns:
        X_train_processed, X_test_processed, y_train, y_test, feature_names
    """
    print("=" * 60)
    print("EMPLOYEE ATTRITION — DATA PREPROCESSING")
    print("=" * 60)

    # ── 1. Load ───────────────────────────────────────────
    df = pd.read_csv(data_path)
    print(f"\n[1/6] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols")

    # ── 2. Drop useless columns ───────────────────────────
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    print(f"[2/6] After dropping constant/ID columns: {df.shape}")

    # ── 3. Encode target ──────────────────────────────────
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    counts = df["Attrition"].value_counts()
    pct    = df["Attrition"].value_counts(normalize=True).mul(100).round(1)
    print(f"[3/6] Target distribution — No: {counts[0]} ({pct[0]}%)  |  Yes: {counts[1]} ({pct[1]}%)")

    # ── 4. Feature engineering ────────────────────────────
    df = engineer_features(df)
    print(f"[4/6] Engineered features added. Total columns: {df.shape[1]}")

    # ── 5. Train/test split ───────────────────────────────
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print(f"[5/6] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows (stratified)")

    # ── 6. Fit preprocessor ───────────────────────────────
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)

    # Build feature name list (for SHAP and interpretability)
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS)
    feature_names = NUMERICAL_COLS + list(cat_feature_names)
    print(f"[6/6] Preprocessor fitted. Feature count after encoding: {len(feature_names)}")

    # ── 7. SMOTE oversampling ─────────────────────────────
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
        print(f"      SMOTE applied → Training set balanced to {X_train_processed.shape[0]} rows (50/50)")

    # ── 8. Save artifacts ─────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir,  exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train_processed)
    np.save(os.path.join(output_dir, "X_test.npy"),  X_test_processed)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"),  y_test.to_numpy())
    joblib.dump(feature_names,  os.path.join(output_dir, "feature_names.pkl"))
    joblib.dump(preprocessor,   os.path.join(model_dir,  "preprocessor.pkl"))

    print("\n✅ Saved to disk:")
    print(f"   {output_dir}/X_train.npy  |  {output_dir}/X_test.npy")
    print(f"   {output_dir}/y_train.npy  |  {output_dir}/y_test.npy")
    print(f"   {output_dir}/feature_names.pkl")
    print(f"   {model_dir}/preprocessor.pkl")
    print("\n" + "=" * 60)

    return X_train_processed, X_test_processed, y_train, y_test.to_numpy(), feature_names


# ──────────────────────────────────────────────────────────
# Inference helper (single employee → processed row)
# ──────────────────────────────────────────────────────────

def prepare_single_employee(input_dict: dict, preprocessor) -> np.ndarray:
    """
    Transform a single employee's raw feature dict into a
    preprocessed numpy row ready for model.predict_proba().

    This mirrors exactly what the training pipeline does,
    including the engineered features.

    Args:
        input_dict:   Dict of raw HR feature values.
        preprocessor: Fitted ColumnTransformer from training.

    Returns:
        Processed 2D numpy array (1 × n_features).
    """
    df = pd.DataFrame([input_dict])
    df = engineer_features(df)
    return preprocessor.transform(df)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_preprocessing.py <path_to_raw_csv>")
        sys.exit(1)
    preprocess_data(data_path=sys.argv[1])
