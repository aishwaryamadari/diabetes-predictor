"""
training.py — Full ML training pipeline with multi-model comparison,
hyperparameter tuning, cross-validation, and model persistence.

Run directly:  python ml/training.py
"""

import os
import sys
import io
import json
import logging

# Force UTF-8 output so unicode symbols display correctly on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, GridSearchCV, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Add project root to path when running standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.preprocessing import build_preprocessor, get_engineered_feature_names, FEATURE_NAMES

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_model")
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_dataset():
    """Return the built-in Pima-like dataset as a DataFrame."""
    from ml.dataset import load_pima_dataset
    return load_pima_dataset()


def evaluate_model(name, y_true, y_pred, y_prob):
    return {
        "model":     name,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ─── Model Definitions + Grids ────────────────────────────────────────────────

def get_model_configs():
    return [
        {
            "name": "Logistic Regression",
            "model": LogisticRegression(max_iter=2000, random_state=42,
                                         class_weight="balanced"),
            "grid": {
                "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "classifier__solver": ["lbfgs", "liblinear"],
                "classifier__penalty": ["l2"],
            },
        },
        {
            "name": "Random Forest",
            "model": RandomForestClassifier(random_state=42,
                                             class_weight="balanced"),
            "grid": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [None, 5, 10, 15],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
        },
        {
            "name": "Gradient Boosting",
            "model": GradientBoostingClassifier(random_state=42),
            "grid": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "classifier__subsample": [0.8, 1.0],
            },
        },
        {
            "name": "SVM",
            "model": SVC(probability=True, random_state=42,
                         class_weight="balanced"),
            "grid": {
                "classifier__C": [0.1, 1, 10, 50],
                "classifier__kernel": ["rbf", "linear"],
                "classifier__gamma": ["scale", "auto"],
            },
        },
    ]


# ─── Training ─────────────────────────────────────────────────────────────────

def train_all():
    logger.info("Loading dataset ...")
    df = load_dataset()
    X = df[FEATURE_NAMES].values
    y = df["Outcome"].values

    logger.info(f"Dataset: {len(df)} samples, {y.sum()} positive ({y.mean()*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    configs = get_model_configs()

    all_metrics = []
    best_auc    = -1
    best_pipeline = None
    best_name   = None

    for cfg in configs:
        logger.info(f"Training {cfg['name']} ...")
        preprocessor = build_preprocessor()
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   cfg["model"]),
        ])

        gs = GridSearchCV(pipe, cfg["grid"], cv=cv, scoring="roc_auc",
                          n_jobs=-1, refit=True, verbose=0)
        gs.fit(X_train, y_train)

        best_pipe = gs.best_estimator_
        y_pred = best_pipe.predict(X_test)
        y_prob = best_pipe.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(cfg["name"], y_test, y_pred, y_prob)
        metrics["best_params"] = {k.replace("classifier__", ""): v
                                   for k, v in gs.best_params_.items()}
        all_metrics.append(metrics)
        logger.info(f"  AUC={metrics['roc_auc']}  F1={metrics['f1']}  Acc={metrics['accuracy']}")

        if metrics["roc_auc"] > best_auc:
            best_auc      = metrics["roc_auc"]
            best_pipeline = best_pipe
            best_name     = cfg["name"]

    logger.info(f"Best model: {best_name} (AUC={best_auc})")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    model_path   = os.path.join(SAVED_MODEL_DIR, "best_model.joblib")
    metrics_path = os.path.join(SAVED_MODEL_DIR, "metrics.json")

    joblib.dump(best_pipeline, model_path)

    summary = {
        "best_model": best_name,
        "best_auc":   best_auc,
        "feature_names": FEATURE_NAMES,
        "models": all_metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\u2713  Model saved \u2192 {os.path.relpath(model_path)}")
    logger.info(f"\u2713  Metrics saved \u2192 {os.path.relpath(metrics_path)}")
    return best_pipeline, summary


# ─── Feature importance (RF) / coef (LR) / generic ───────────────────────────

def get_feature_importance(pipeline):
    """
    Extract feature importance / coefficients from the final estimator.
    Returns list of {feature, importance} sorted descending.
    """
    clf = pipeline.named_steps["classifier"]
    eng_names = get_engineered_feature_names()

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return []

    # Pad / trim to match engineered feature count
    if len(importances) < len(eng_names):
        importances = np.pad(importances, (0, len(eng_names) - len(importances)))
    else:
        importances = importances[: len(eng_names)]

    result = sorted(
        [{"feature": n, "importance": round(float(v), 4)}
         for n, v in zip(eng_names, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )
    return result


if __name__ == "__main__":
    train_all()
