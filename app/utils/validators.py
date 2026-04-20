"""
app/utils/validators.py — Strict input schema validation.
"""
from typing import Tuple, Dict, Any

SCHEMA = {
    "Pregnancies":             {"type": float, "min": 0,   "max": 20},
    "Glucose":                 {"type": float, "min": 0,   "max": 300},
    "BloodPressure":           {"type": float, "min": 0,   "max": 200},
    "SkinThickness":           {"type": float, "min": 0,   "max": 100},
    "Insulin":                 {"type": float, "min": 0,   "max": 900},
    "BMI":                     {"type": float, "min": 0,   "max": 70},
    "DiabetesPedigreeFunction":{"type": float, "min": 0,   "max": 2.5},
    "Age":                     {"type": float, "min": 1,   "max": 120},
}


def validate_prediction_input(data: Dict[str, Any]) -> Tuple[Dict, list]:
    """
    Returns (cleaned_dict, errors).
    errors is empty list on success.
    """
    cleaned = {}
    errors  = []

    for field, rules in SCHEMA.items():
        if field not in data:
            errors.append(f"Missing required field: {field}")
            continue
        try:
            val = rules["type"](data[field])
        except (ValueError, TypeError):
            errors.append(f"Invalid type for {field}: expected number")
            continue
        if val < rules["min"] or val > rules["max"]:
            errors.append(
                f"{field} must be between {rules['min']} and {rules['max']}, got {val}"
            )
            continue
        cleaned[field] = val

    return cleaned, errors
