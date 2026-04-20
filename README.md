---
title: Diabetes Predictor
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

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
- CSS custom properties for themi

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/diabetes-risk-predictor
cd diabetes-risk-predictor
pip install -r requirements.txt
 