"""
tests/test_app.py — Comprehensive test suite for the Diabetes Risk Predictor.

Run with:  python -m pytest tests/ -v
"""
import os
import sys
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_INPUT = {
    "Pregnancies": 2,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
}

HIGH_RISK_INPUT = {
    "Pregnancies": 8,
    "Glucose": 196,
    "BloodPressure": 80,
    "SkinThickness": 0,
    "Insulin": 0,
    "BMI": 39.8,
    "DiabetesPedigreeFunction": 0.451,
    "Age": 51,
}

LOW_RISK_INPUT = {
    "Pregnancies": 0,
    "Glucose": 85,
    "BloodPressure": 66,
    "SkinThickness": 29,
    "Insulin": 0,
    "BMI": 26.6,
    "DiabetesPedigreeFunction": 0.351,
    "Age": 31,
}


@pytest.fixture(scope="session")
def app():
    """Create a test Flask app with an in-memory database."""
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    from app import create_app
    application = create_app("development")
    application.config["TESTING"] = True
    application.config["DATABASE_URL"] = "sqlite:///:memory:"
    return application


@pytest.fixture(scope="session")
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def auth_header(client):
    """Register a test user and return JWT auth header."""
    client.post("/auth/register", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123",
    })
    res  = client.post("/auth/login", json={
        "username": "testuser",
        "password": "testpass123",
    })
    data  = json.loads(res.data)
    token = data.get("token", "")
    return {"Authorization": f"Bearer {token}"}


# ── ML Pipeline Tests ──────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_zero_to_nan_transformer(self):
        from ml.preprocessing import ZeroToNanTransformer, FEATURE_NAMES
        import pandas as pd

        raw = {k: [0] for k in FEATURE_NAMES}
        df  = pd.DataFrame(raw)
        t   = ZeroToNanTransformer()
        out = t.fit_transform(df)
        # BloodPressure zero → nan
        df2 = pd.DataFrame(out, columns=FEATURE_NAMES)
        assert np.isnan(df2["Glucose"].iloc[0])
        assert np.isnan(df2["BMI"].iloc[0])

    def test_outlier_clipper(self):
        from ml.preprocessing import OutlierClipper
        X = np.array([[1, 2], [3, 4], [1000, -1000]])
        oc = OutlierClipper(factor=1.5)
        oc.fit(X)
        Xt = oc.transform(X)
        assert Xt[2, 0] < 1000   # clipped
        assert Xt[2, 1] > -1000  # clipped

    def test_full_pipeline_shape(self):
        from ml.preprocessing import build_preprocessor, FEATURE_NAMES

        X = np.array([[VALID_INPUT[k] for k in FEATURE_NAMES]])
        pipe = build_preprocessor()
        # Need to fit first
        from ml.dataset import load_pima_dataset
        df   = load_pima_dataset()
        Xtrain = df[FEATURE_NAMES].values
        pipe.fit(Xtrain)
        Xt = pipe.transform(X)
        # 8 original + 3 engineered = 11 features
        assert Xt.shape == (1, 11)

    def test_feature_engineering_names(self):
        from ml.preprocessing import get_engineered_feature_names, FEATURE_NAMES
        eng = get_engineered_feature_names()
        assert len(eng) == len(FEATURE_NAMES) + 3
        assert "BMI_Category" in eng
        assert "Age_Group" in eng
        assert "Glucose_BMI" in eng


class TestDataset:
    def test_dataset_shape(self):
        from ml.dataset import load_pima_dataset
        df = load_pima_dataset()
        assert df.shape == (300, 9)

    def test_dataset_reproducible(self):
        from ml.dataset import load_pima_dataset
        df1 = load_pima_dataset()
        df2 = load_pima_dataset()
        assert df1.equals(df2)

    def test_outcome_binary(self):
        from ml.dataset import load_pima_dataset
        df = load_pima_dataset()
        assert set(df["Outcome"].unique()) == {0, 1}

    def test_no_negative_values(self):
        from ml.dataset import load_pima_dataset
        df = load_pima_dataset()
        numeric_cols = ["Glucose", "BloodPressure", "BMI", "Age"]
        for col in numeric_cols:
            assert df[col].min() >= 0, f"{col} has negative values"


class TestTraining:
    def test_model_exists(self):
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "saved_model", "best_model.joblib"
        )
        assert os.path.exists(model_path), "Model file not found. Run ml/training.py first."

    def test_metrics_exist(self):
        metrics_path = os.path.join(
            os.path.dirname(__file__), "..", "saved_model", "metrics.json"
        )
        assert os.path.exists(metrics_path)
        with open(metrics_path) as f:
            data = json.load(f)
        assert "best_model" in data
        assert "models" in data
        assert len(data["models"]) >= 2

    def test_model_predicts(self):
        import joblib
        from ml.preprocessing import FEATURE_NAMES

        model_path = os.path.join(
            os.path.dirname(__file__), "..", "saved_model", "best_model.joblib"
        )
        pipeline = joblib.load(model_path)
        X = np.array([[VALID_INPUT[k] for k in FEATURE_NAMES]])
        pred = pipeline.predict(X)
        prob = pipeline.predict_proba(X)
        assert pred[0] in [0, 1]
        assert 0.0 <= prob[0][1] <= 1.0

    def test_feature_importance(self):
        import joblib
        from ml.training import get_feature_importance

        model_path = os.path.join(
            os.path.dirname(__file__), "..", "saved_model", "best_model.joblib"
        )
        pipeline   = joblib.load(model_path)
        importance = get_feature_importance(pipeline)
        assert len(importance) > 0
        assert all("feature" in i and "importance" in i for i in importance)
        # sorted descending
        imps = [i["importance"] for i in importance]
        assert imps == sorted(imps, reverse=True)


# ── Validator Tests ────────────────────────────────────────────────────────────

class TestValidator:
    def test_valid_input(self):
        from app.utils.validators import validate_prediction_input
        cleaned, errors = validate_prediction_input(VALID_INPUT)
        assert errors == []
        assert len(cleaned) == 8

    def test_missing_field(self):
        from app.utils.validators import validate_prediction_input
        bad = {k: v for k, v in VALID_INPUT.items() if k != "Glucose"}
        _, errors = validate_prediction_input(bad)
        assert any("Glucose" in e for e in errors)

    def test_out_of_range(self):
        from app.utils.validators import validate_prediction_input
        bad = {**VALID_INPUT, "Glucose": 999}
        _, errors = validate_prediction_input(bad)
        assert any("Glucose" in e for e in errors)

    def test_negative_value(self):
        from app.utils.validators import validate_prediction_input
        bad = {**VALID_INPUT, "BMI": -5}
        _, errors = validate_prediction_input(bad)
        assert any("BMI" in e for e in errors)

    def test_type_coercion(self):
        from app.utils.validators import validate_prediction_input
        str_input = {k: str(v) for k, v in VALID_INPUT.items()}
        cleaned, errors = validate_prediction_input(str_input)
        assert errors == []
        assert isinstance(cleaned["Glucose"], float)


# ── API Endpoint Tests ─────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_valid(self, client):
        res  = client.post("/api/predict",
                           json=VALID_INPUT,
                           content_type="application/json")
        data = json.loads(res.data)
        assert res.status_code == 200
        assert "prediction" in data
        assert "probability" in data
        assert "risk_level" in data
        assert "label" in data
        assert "top_factors" in data
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0
        assert data["risk_level"] in ["Low", "Moderate", "High"]

    def test_predict_missing_field(self, client):
        bad = {k: v for k, v in VALID_INPUT.items() if k != "Glucose"}
        res = client.post("/api/predict", json=bad)
        assert res.status_code == 422

    def test_predict_no_body(self, client):
        res = client.post("/api/predict")
        assert res.status_code == 400

    def test_predict_out_of_range(self, client):
        bad = {**VALID_INPUT, "Glucose": 9999}
        res = client.post("/api/predict", json=bad)
        assert res.status_code == 422

    def test_predict_high_risk(self, client):
        res  = client.post("/api/predict", json=HIGH_RISK_INPUT)
        data = json.loads(res.data)
        assert res.status_code == 200
        assert data["probability"] > 0.3  # should have meaningful probability

    def test_predict_low_risk(self, client):
        res  = client.post("/api/predict", json=LOW_RISK_INPUT)
        data = json.loads(res.data)
        assert res.status_code == 200
        assert isinstance(data["probability"], float)


class TestMetricsEndpoint:
    def test_metrics_returns_ok(self, client):
        res  = client.get("/api/metrics")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert "best_model" in data
        assert "models" in data

    def test_metrics_has_required_fields(self, client):
        res  = client.get("/api/metrics")
        data = json.loads(res.data)
        for model in data["models"]:
            for field in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                assert field in model, f"Missing {field} in model metrics"


class TestAuthEndpoints:
    def test_register(self, client):
        res  = client.post("/auth/register", json={
            "username": "newuser_test",
            "email":    "newuser@test.com",
            "password": "password123",
        })
        data = json.loads(res.data)
        assert res.status_code == 201
        assert "token" in data
        assert "user" in data

    def test_register_duplicate(self, client):
        payload = {"username": "dupuser", "email": "dup@test.com", "password": "pass1234"}
        client.post("/auth/register", json=payload)
        res = client.post("/auth/register", json=payload)
        assert res.status_code == 409

    def test_login_success(self, client):
        client.post("/auth/register", json={
            "username": "logintest", "email": "login@test.com", "password": "mypassword"
        })
        res  = client.post("/auth/login", json={"username": "logintest", "password": "mypassword"})
        data = json.loads(res.data)
        assert res.status_code == 200
        assert "token" in data

    def test_login_wrong_password(self, client):
        res = client.post("/auth/login", json={"username": "testuser", "password": "wrongpass"})
        assert res.status_code == 401

    def test_login_nonexistent_user(self, client):
        res = client.post("/auth/login", json={"username": "ghost", "password": "anything"})
        assert res.status_code == 401

    def test_protected_history_no_token(self, client):
        res = client.get("/api/history")
        assert res.status_code in [401, 422]

    def test_protected_history_with_token(self, client, auth_header):
        res = client.get("/api/history", headers=auth_header)
        assert res.status_code == 200
        data = json.loads(res.data)
        assert isinstance(data, list)


class TestPageRoutes:
    def test_index_returns_html(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert b"Diabetes" in res.data

    def test_dashboard_returns_html(self, client):
        res = client.get("/dashboard")
        assert res.status_code == 200
