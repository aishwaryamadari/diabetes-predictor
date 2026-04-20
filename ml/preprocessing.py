"""
preprocessing.py — Robust data preprocessing pipeline for Pima Indians Diabetes Dataset.
Handles missing values, outliers, feature engineering, and scaling.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

# Features that cannot biologically be zero — treat as missing
ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]


class ZeroToNanTransformer(BaseEstimator, TransformerMixin):
    """Replace biologically impossible zeros with NaN before imputation."""

    def __init__(self, columns=None):
        self.columns = columns or ZERO_AS_NAN_COLS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=FEATURE_NAMES)
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)
        return X.values if isinstance(X, pd.DataFrame) else X


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip outliers using IQR method. Fitted on training data only."""

    def __init__(self, factor=3.0):
        self.factor = factor
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        Q1 = np.nanpercentile(X, 25, axis=0)
        Q3 = np.nanpercentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_ = Q1 - self.factor * IQR
        self.upper_ = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(float)
        X = np.clip(X, self.lower_, self.upper_)
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Add engineered features:
      - BMI category (underweight/normal/overweight/obese) as ordinal
      - Age group ordinal
      - Glucose-BMI interaction
      - Insulin-to-Glucose ratio (insulin resistance proxy)
      - High-risk pregnancy flag
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=FEATURE_NAMES)
        else:
            df = X.copy()

        # BMI category (ordinal: 0=underweight, 1=normal, 2=overweight, 3=obese)
        bmi_bins   = [0, 18.5, 25, 30, np.inf]
        bmi_labels = [0, 1, 2, 3]
        df["BMI_Category"] = pd.cut(df["BMI"], bins=bmi_bins, labels=bmi_labels).astype(float)

        # Age group (ordinal)
        age_bins   = [0, 30, 45, 60, np.inf]
        age_labels = [0, 1, 2, 3]
        df["Age_Group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels).astype(float)

        # Glucose x BMI interaction (captures compound risk)
        df["Glucose_BMI"] = df["Glucose"] * df["BMI"]

        # Insulin-to-Glucose ratio (insulin resistance proxy)
        glucose_safe = df["Glucose"].replace(0, np.nan).fillna(1)
        df["Insulin_Glucose_Ratio"] = df["Insulin"] / glucose_safe

        # High-risk pregnancy flag (age > 35 & pregnancies >= 4)
        df["HighRisk_Pregnancy"] = ((df["Age"] > 35) & (df["Pregnancies"] >= 4)).astype(float)

        # Glucose categories (low / normal / pre-diabetic / diabetic)
        gluc_bins   = [0, 70, 100, 125, np.inf]
        gluc_labels = [0, 1, 2, 3]
        df["Glucose_Category"] = pd.cut(df["Glucose"], bins=gluc_bins, labels=gluc_labels).astype(float)

        return df.values


def get_engineered_feature_names():
    return FEATURE_NAMES + [
        "BMI_Category", "Age_Group", "Glucose_BMI",
        "Insulin_Glucose_Ratio", "HighRisk_Pregnancy", "Glucose_Category"
    ]


def build_preprocessor():
    """Return an unfitted sklearn-compatible preprocessing pipeline."""
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("zero_to_nan",   ZeroToNanTransformer()),
        ("outlier_clip",  OutlierClipper(factor=3.0)),
        ("imputer",       SimpleImputer(strategy="median")),
        ("feature_eng",   FeatureEngineer()),
        ("scaler",        StandardScaler()),
    ])
    return pipe
