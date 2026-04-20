"""
app/models/db_models.py — SQLAlchemy ORM models.
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(80), unique=True, nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin   = db.Column(db.Boolean, default=False)

    predictions = db.relationship("PredictionHistory", backref="user", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            "id":       self.id,
            "username": self.username,
            "email":    self.email,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat(),
        }


class PredictionHistory(db.Model):
    __tablename__ = "prediction_history"

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    timestamp  = db.Column(db.DateTime, default=datetime.utcnow)

    # Inputs
    pregnancies  = db.Column(db.Float)
    glucose      = db.Column(db.Float)
    blood_pressure = db.Column(db.Float)
    skin_thickness = db.Column(db.Float)
    insulin      = db.Column(db.Float)
    bmi          = db.Column(db.Float)
    dpf          = db.Column(db.Float)
    age          = db.Column(db.Float)

    # Outputs
    prediction   = db.Column(db.Integer)   # 0 or 1
    probability  = db.Column(db.Float)
    risk_level   = db.Column(db.String(20))  # Low / Moderate / High

    def to_dict(self):
        return {
            "id":        self.id,
            "timestamp": self.timestamp.isoformat(),
            "inputs": {
                "Pregnancies":             self.pregnancies,
                "Glucose":                 self.glucose,
                "BloodPressure":           self.blood_pressure,
                "SkinThickness":           self.skin_thickness,
                "Insulin":                 self.insulin,
                "BMI":                     self.bmi,
                "DiabetesPedigreeFunction": self.dpf,
                "Age":                     self.age,
            },
            "prediction":  self.prediction,
            "probability": self.probability,
            "risk_level":  self.risk_level,
        }
