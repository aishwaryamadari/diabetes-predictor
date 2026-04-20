"""
app/services/prediction_service.py — Loads the saved model and runs predictions.
Also computes feature-importance explanations per prediction.
"""
import os
import json
import logging
import numpy as np
import joblib

logger = logging.getLogger(__name__)

_pipeline = None
_metrics  = None


def _load():
    global _pipeline, _metrics
    if _pipeline is not None:
        return

    from flask import current_app
    model_path   = current_app.config["MODEL_PATH"]
    metrics_path = current_app.config["METRICS_PATH"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run  python ml/training.py  first."
        )

    _pipeline = joblib.load(model_path)
    logger.info("Model loaded from %s", model_path)

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _metrics = json.load(f)


def _risk_level(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    if prob < 0.60:
        return "Moderate"
    return "High"


def predict_single(input_dict: dict) -> dict:
    """
    input_dict keys: Pregnancies, Glucose, BloodPressure,
                     SkinThickness, Insulin, BMI,
                     DiabetesPedigreeFunction, Age
    """
    _load()
    from ml.preprocessing import FEATURE_NAMES, get_engineered_feature_names
    from ml.training import get_feature_importance

    row = np.array([[input_dict[k] for k in FEATURE_NAMES]], dtype=float)

    prediction  = int(_pipeline.predict(row)[0])
    probability = float(_pipeline.predict_proba(row)[0][1])
    risk        = _risk_level(probability)

    # Feature importance explanation
    top_factors = get_feature_importance(_pipeline)[:5]

    # Generate doctor-style summary
    factors = [f["feature"].replace("_", " ") if isinstance(f, dict) else f[0].replace("_", " ") for f in top_factors[:2]]
    prob_pct = round(probability * 100)
    if prediction == 1:
        doc_summary = f"Patient shows a {prob_pct}% probability of diabetes. Elevated risk is primarily driven by their {factors[0]} and {factors[1]} levels. Lifestyle or medical intervention is recommended."
    else:
        doc_summary = f"Patient shows a low probability ({prob_pct}%) of diabetes. {factors[0]} and {factors[1]} are currently the most influential factors. Continue monitoring as part of routine care."

    return {
        "prediction":  prediction,
        "probability": round(probability, 4),
        "risk_level":  risk,
        "label":       "Diabetic" if prediction == 1 else "Non-Diabetic",
        "top_factors": top_factors,
        "doctor_summary": doc_summary,
    }


def get_metrics() -> dict:
    _load()
    return _metrics or {}
