<<<<<<< HEAD
# 🩺 Diabetes Risk Predictor

> A production-grade, full-stack diabetes risk assessment system powered by Machine Learning, Flask, and a premium healthcare-grade UI.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ Overview

Diabetes Risk Predictor is a clinical-grade web application that uses trained machine learning models to assess a user's diabetes risk from key health metrics. Built to portfolio-level standards — it features explainable AI, JWT authentication, prediction history, batch CSV predictions, and PDF report downloads.

---

## 🎯 Features

| Feature | Description |
|---|---|
| 🤖 **ML Prediction** | Random Forest + Logistic Regression + SVM with GridSearchCV tuning |
| 📊 **Explainability** | Feature importance rankings per prediction |
| 🔐 **Authentication** | JWT-based register/login with BCrypt password hashing |
| 📜 **History** | Per-user prediction history stored in SQLite |
| 📦 **Batch Predict** | Upload CSV files for bulk predictions |
| 📄 **PDF Reports** | Downloadable clinical-style PDF for each prediction |
| 📈 **Live Metrics** | Real-time model performance dashboard (AUC, F1, Accuracy) |
| 📱 **Responsive** | Mobile-first, accessible UI |

---

## 🛠 Tech Stack

**Backend**
- Python 3.10+, Flask 3.0
- SQLAlchemy + SQLite (PostgreSQL-ready)
- Flask-JWT-Extended (authentication)
- scikit-learn (ML pipeline)
- joblib (model serialization)

**Frontend**
- Vanilla JS + Chart.js (no build step required)
- DM Serif Display + DM Sans fonts
- CSS custom properties for theming

**ML Pipeline**
- ZeroToNaN imputation for biological zeros
- IQR-based outlier clipping
- Median imputation for missing values
- Feature engineering (BMI category, age group, Glucose×BMI)
- StandardScaler normalization
- GridSearchCV with StratifiedKFold cross-validation

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/diabetes-risk-predictor
cd diabetes-risk-predictor
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env .env.local
# Edit .env.local and set your SECRET_KEY and JWT_SECRET_KEY
```

### 3. Train the Model

```bash
python ml/training.py
# Trains 3 models, selects best by AUC, saves to saved_model/
```

### 4. Run the App

```bash
python app.py
# → http://localhost:5000
```

That's it. The SQLite database is created automatically on first run.

---

## 📁 Project Structure

```
diabetes_predictor/
├── app/
│   ├── __init__.py              # Flask application factory
│   ├── models/
│   │   └── db_models.py         # SQLAlchemy: User, PredictionHistory
│   ├── routes/
│   │   ├── auth.py              # /auth/register, /auth/login, /auth/me
│   │   ├── predict.py           # /api/predict, /api/metrics, /api/history
│   │   └── pages.py             # HTML page serving
│   ├── services/
│   │   ├── prediction_service.py # Model loading + inference
│   │   └── report_service.py    # PDF report generation
│   └── utils/
│       └── validators.py        # Input schema validation
├── ml/
│   ├── dataset.py               # Reproducible dataset generator
│   ├── preprocessing.py         # sklearn-compatible Pipeline transformers
│   └── training.py              # Multi-model training + GridSearch
├── saved_model/
│   ├── best_model.joblib        # Serialized best pipeline
│   └── metrics.json             # Model comparison metrics
├── static/
│   ├── css/print.css
│   └── sample_batch.csv         # Example batch upload file
├── templates/
│   └── index.html               # Single-page application
├── tests/
│   └── test_app.py              # pytest test suite (30+ tests)
├── app.py                       # Entry point
├── config.py                    # Environment-based config
├── retrain.py                   # Standalone retraining script
├── requirements.txt
├── Procfile                     # Render/Railway deployment
└── .env                         # Environment variables
```

---

## 🔌 API Reference

### POST `/api/predict`

Predict diabetes risk for a single patient.

**Request Body:**
```json
{
  "Pregnancies": 2,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.742,
  "risk_level": "High",
  "label": "Diabetic",
  "top_factors": [
    { "feature": "Glucose", "importance": 0.2841 },
    { "feature": "BMI", "importance": 0.1923 }
  ],
  "history_id": 42
}
```

### GET `/api/metrics`

Returns model performance metrics for all trained models.

### POST `/api/batch_predict`

Upload a CSV file (field: `file`) with the same columns as the predict endpoint. Returns predictions for all rows.

### GET `/api/history`

Returns the authenticated user's prediction history (JWT required).

### GET `/api/report/<id>`

Download a PDF report for prediction record `<id>`.

### POST `/auth/register`

```json
{ "username": "alice", "email": "alice@example.com", "password": "secure123" }
```

### POST `/auth/login`

```json
{ "username": "alice", "password": "secure123" }
```

---

## 🧪 Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Expected: **30+ tests** covering ML pipeline, validators, all API endpoints, and auth flows.

---

## 🚢 Deployment

### Render / Railway

1. Push to GitHub
2. Connect repo in Render → New Web Service
3. Set environment variables: `SECRET_KEY`, `JWT_SECRET_KEY`, `FLASK_ENV=production`
4. Build command: `pip install -r requirements.txt && python ml/training.py`
5. Start command: `gunicorn app:app`

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python ml/training.py
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 71.7% | 55.6% | 0.541 | 0.755 |
| Random Forest ⭐ | 68.3% | 50.0% | 0.487 | **0.759** |
| SVM | 71.7% | 55.6% | 0.541 | 0.759 |

*Best model selected by ROC-AUC. Dataset: 300-record Pima Indians-like dataset.*

---

## ⚠️ Medical Disclaimer

This application is built for **educational and portfolio purposes only**. It does not constitute medical advice and should never replace a qualified healthcare professional's assessment.

---

## 📄 License

MIT © 2025
=======
---
title: Diabetes Predictor
emoji: 🐠
colorFrom: pink
colorTo: yellow
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 0caffa0c62eaa9b923aee94733459f225ce7a673
