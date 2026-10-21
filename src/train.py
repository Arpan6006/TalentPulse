# ============================================================
# Employee Attrition Prediction - Model Training Module
# ============================================================
# Extracted from: notebooks/03_Modeling.ipynb
# Run with: python src/train.py
# ============================================================

import os
import json
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap


# ──────────────────────────────────────────────────────────
# Model Definitions
# ──────────────────────────────────────────────────────────

def get_models():
    """Return a dict of model name → fitted estimator."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            class_weight="balanced",
            verbose=-1
        ),
    }


# ──────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model") -> dict:
    """
    Evaluate a trained model and print a full classification report.

    Returns a dict with Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model":     model_name,
        "Accuracy":  round(accuracy_score(y_test, y_pred),             4),
        "Precision": round(precision_score(y_test, y_pred),            4),
        "Recall":    round(recall_score(y_test, y_pred),               4),
        "F1-Score":  round(f1_score(y_test, y_pred),                   4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba),             4),
        "PR-AUC":    round(average_precision_score(y_test, y_proba),   4),
    }

    print(f"\n{'=' * 50}")
    print(f"  {model_name}")
    print(f"{'=' * 50}")
    print(classification_report(y_test, y_pred, target_names=["No Attrition", "Attrition"]))
    print(f"  ROC-AUC : {metrics['ROC-AUC']}")
    print(f"  PR-AUC  : {metrics['PR-AUC']}")

    return metrics


# ──────────────────────────────────────────────────────────
# SHAP Explainer
# ──────────────────────────────────────────────────────────

def build_shap_explainer(model, X_train: np.ndarray, model_dir: str):
    """
    Fit and save a SHAP TreeExplainer for the best model.
    Falls back to KernelExplainer for non-tree models.
    """
    print("\n[SHAP] Building explainer …")
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        # Fallback for logistic regression / non-tree models
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))

    path = os.path.join(model_dir, "shap_explainer.pkl")
    joblib.dump(explainer, path)
    print(f"[SHAP] Explainer saved → {path}")
    return explainer


# ──────────────────────────────────────────────────────────
# Main Training Pipeline
# ──────────────────────────────────────────────────────────

def train(
    data_dir:  str = "data/processed",
    model_dir: str = "models",
):
    """
    Load preprocessed data, train all models, select the best
    by ROC-AUC, save individual model files plus the SHAP explainer.

    Results summary is written to models/results.json.
    """
    print("=" * 60)
    print("EMPLOYEE ATTRITION — MODEL TRAINING")
    print("=" * 60)

    # ── Load processed data ───────────────────────────────
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))

    print(f"\nData loaded  →  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"Class balance (train) — No: {(y_train==0).sum()} | Yes: {(y_train==1).sum()}")

    os.makedirs(model_dir, exist_ok=True)

    # ── Train & evaluate each model ───────────────────────
    models  = get_models()
    results = []
    trained = {}

    for name, model in models.items():
        print(f"\nTraining {name} …")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        results.append(metrics)
        trained[name] = model

        path = os.path.join(model_dir, f"{name}.pkl")
        joblib.dump(model, path)
        print(f"  Saved → {path}")

    # ── Select best model by ROC-AUC ─────────────────────
    best_result = max(results, key=lambda m: m["ROC-AUC"])
    best_name   = best_result["Model"]
    best_model  = trained[best_name]

    best_path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(best_model, best_path)
    print(f"\n🏆 Best model: {best_name}  (ROC-AUC = {best_result['ROC-AUC']})")
    print(f"   Saved → {best_path}")

    # ── SHAP explainer ────────────────────────────────────
    build_shap_explainer(best_model, X_train, model_dir)

    # ── Save results JSON ─────────────────────────────────
    results_path = os.path.join(model_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Results saved → {results_path}")

    # ── Print comparison table ────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    header = f"{'Model':<25} {'Accuracy':>9} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}"
    print(header)
    print("-" * 60)
    for r in sorted(results, key=lambda m: m["ROC-AUC"], reverse=True):
        mark = " ⭐" if r["Model"] == best_name else ""
        print(
            f"{r['Model']:<25} {r['Accuracy']:>9.4f} "
            f"{r['Recall']:>8.4f} {r['F1-Score']:>8.4f} {r['ROC-AUC']:>9.4f}{mark}"
        )
    print("=" * 60)

    return best_model, results


if __name__ == "__main__":
    train()
