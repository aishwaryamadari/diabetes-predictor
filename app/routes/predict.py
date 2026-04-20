"""
app/routes/predict.py — Prediction API endpoints.
"""
import io
import csv
import logging
from flask import Blueprint, request, jsonify, Response
from flask_jwt_extended import jwt_required, get_jwt_identity, verify_jwt_in_request

from app.models.db_models import db, PredictionHistory
from app.services.prediction_service import predict_single, get_metrics
from app.utils.validators import validate_prediction_input

logger = logging.getLogger(__name__)
predict_bp = Blueprint("predict", __name__, url_prefix="/api")


def _try_get_user_id():
    try:
        verify_jwt_in_request(optional=True)
        return get_jwt_identity()
    except Exception:
        return None


@predict_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    cleaned, errors = validate_prediction_input(data)
    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 422

    try:
        result = predict_single(cleaned)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

    # Persist to history
    user_id = _try_get_user_id()
    record  = PredictionHistory(
        user_id        = user_id,
        pregnancies    = cleaned["Pregnancies"],
        glucose        = cleaned["Glucose"],
        blood_pressure = cleaned["BloodPressure"],
        skin_thickness = cleaned["SkinThickness"],
        insulin        = cleaned["Insulin"],
        bmi            = cleaned["BMI"],
        dpf            = cleaned["DiabetesPedigreeFunction"],
        age            = cleaned["Age"],
        prediction     = result["prediction"],
        probability    = result["probability"],
        risk_level     = result["risk_level"],
    )
    db.session.add(record)
    db.session.commit()
    result["history_id"] = record.id

    return jsonify(result), 200


@predict_bp.route("/metrics", methods=["GET"])
def metrics():
    try:
        data = get_metrics()
        return jsonify(data), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503


@predict_bp.route("/upload_predict", methods=["POST"])
def upload_predict():
    if "file" not in request.files:
        return jsonify({"error": "File required (field: file)"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400

    from app.services.file_parser import parse_file
    
    try:
        parsed_records = parse_file(f)
    except Exception as e:
        logger.exception("File parsing failed")
        return jsonify({"error": f"Failed to parse file: {str(e)}"}), 400
        
    if not parsed_records:
        return jsonify({"error": "No valid patient data found in the file."}), 400

    results = []
    errors  = []

    for i, record in enumerate(parsed_records, start=1):
        cleaned, errs = validate_prediction_input(record)
        if errs:
            errors.append({"row": i, "errors": errs, "extracted_data": record})
            continue
        try:
            res = predict_single(cleaned)
            results.append({**cleaned, **res})
        except Exception as e:
            errors.append({"row": i, "errors": [str(e)]})

    return jsonify({
        "results": results,
        "errors": errors,
        "processed": len(results),
        "total_found": len(parsed_records)
    }), 200


@predict_bp.route("/history", methods=["GET"])
@jwt_required()
def history():
    user_id = get_jwt_identity()
    records = (PredictionHistory.query
               .filter_by(user_id=user_id)
               .order_by(PredictionHistory.timestamp.desc())
               .limit(50).all())
    return jsonify([r.to_dict() for r in records]), 200


@predict_bp.route("/report/<int:record_id>", methods=["GET"])
def download_report(record_id):
    """Download a PDF report for a specific prediction."""
    record = PredictionHistory.query.get(record_id)
    if not record:
        return jsonify({"error": "Record not found"}), 404

    from app.services.report_service import generate_pdf_report

    inputs = {
        "Pregnancies":              record.pregnancies,
        "Glucose":                  record.glucose,
        "BloodPressure":            record.blood_pressure,
        "SkinThickness":            record.skin_thickness,
        "Insulin":                  record.insulin,
        "BMI":                      record.bmi,
        "DiabetesPedigreeFunction": record.dpf,
        "Age":                      record.age,
    }
    prediction_data = {
        "prediction":  record.prediction,
        "probability": record.probability,
        "risk_level":  record.risk_level,
        "label":       "Diabetic" if record.prediction == 1 else "Non-Diabetic",
        "top_factors": [],
    }

    try:
        # Re-run prediction to get feature importance
        from app.services.prediction_service import predict_single
        full = predict_single(inputs)
        prediction_data["top_factors"] = full.get("top_factors", [])
    except Exception:
        pass

    try:
        pdf_bytes = generate_pdf_report(prediction_data, inputs)
    except ImportError:
        return jsonify({"error": "reportlab not installed"}), 503

    from flask import send_file
    buf = __import__("io").BytesIO(pdf_bytes)
    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"diabetes_report_{record_id}.pdf",
    )


@predict_bp.route("/history/all", methods=["GET"])
@jwt_required()
def all_history():
    """Admin: view all predictions."""
    from app.models.db_models import User
    user_id = get_jwt_identity()
    user    = User.query.get(user_id)
    if not user or not user.is_admin:
        return jsonify({"error": "Admin access required"}), 403

    records = (PredictionHistory.query
               .order_by(PredictionHistory.timestamp.desc())
               .limit(200).all())

    stats = {
        "total":    PredictionHistory.query.count(),
        "diabetic": PredictionHistory.query.filter_by(prediction=1).count(),
        "users":    db.session.query(PredictionHistory.user_id).distinct().count(),
    }
    return jsonify({"stats": stats, "records": [r.to_dict() for r in records]}), 200
