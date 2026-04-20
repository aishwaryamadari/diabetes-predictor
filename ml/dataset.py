"""
dataset.py — Loads diabetes dataset from CSV or falls back to synthetic data.

To use your own CSV:
  1. Place your file at:  data/diabetes.csv  (inside the project folder)
  2. Make sure it has these columns:
     Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
     BMI, DiabetesPedigreeFunction, Age, Outcome
  3. Run:  python ml/training.py
"""
import os
import numpy as np
import pandas as pd

_SEED = 42

# Path to your CSV file (relative to project root)
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "my_data_3000.csv")

REQUIRED_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]


def load_pima_dataset() -> pd.DataFrame:
    """
    Load dataset from CSV if available, otherwise use synthetic data.
    """
    csv_path = os.path.abspath(CSV_PATH)

    if os.path.exists(csv_path):
        print(f"[Dataset] Loading from CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        # Validate required columns
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Your CSV is missing these columns: {missing}\n"
                f"Required: {REQUIRED_COLUMNS}"
            )

        print(f"[Dataset] Loaded {len(df)} rows, {df['Outcome'].sum()} positive cases.")
        return df[REQUIRED_COLUMNS]

    else:
        print(f"[Dataset] No CSV found at: {csv_path}")
        print("[Dataset] Using built-in synthetic dataset (768 rows).")
        return _generate_synthetic()


def _generate_synthetic() -> pd.DataFrame:
    """Generate a reproducible 768-row Pima-like dataset as fallback."""
    rng = np.random.default_rng(_SEED)
    n = 768

    preg = rng.choice(range(0, 15), n,
                      p=[0.15, 0.14, 0.12, 0.11, 0.10, 0.09, 0.08,
                         0.07, 0.05, 0.04, 0.02, 0.01, 0.01, 0.005, 0.005])
    gluc = np.clip(rng.normal(121, 32, n), 44, 280).astype(int)
    bp   = np.clip(rng.normal(69, 19, n), 0, 122).astype(int)
    skin = np.clip(rng.normal(20, 16, n), 0, 99).astype(int)
    ins  = np.clip(rng.exponential(80, n), 0, 846).astype(int)
    bmi  = np.round(np.clip(rng.normal(32, 7, n), 0, 67.1), 1)
    dpf  = np.round(np.clip(rng.exponential(0.47, n), 0.078, 2.42), 3)
    age  = np.clip(rng.normal(33, 12, n), 21, 81).astype(int)

    logit = (-8 + 0.045 * gluc + 0.035 * bmi + 0.018 * age
             + 0.8 * dpf - 0.012 * bp + 0.003 * ins + 0.06 * preg)
    prob    = 1 / (1 + np.exp(-logit))
    outcome = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "Pregnancies":              preg,
        "Glucose":                  gluc,
        "BloodPressure":            bp,
        "SkinThickness":            skin,
        "Insulin":                  ins,
        "BMI":                      bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age":                      age,
        "Outcome":                  outcome,
    })
