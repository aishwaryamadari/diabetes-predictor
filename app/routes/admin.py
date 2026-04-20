"""
app/routes/admin.py — Admin panel endpoints (admin users only).
"""
from flask import Blueprint, jsonify, render_template
from flask_jwt_extended import jwt_required, get_jwt_identity

from app.models.db_models import db, User, PredictionHistory
from sqlalchemy import func

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def _require_admin():
    uid  = get_jwt_identity()
    user = User.query.get(uid)
    if not user or not user.is_admin:
        return None, (jsonify({"error": "Admin access required"}), 403)
    return user, None


@admin_bp.route("/stats", methods=["GET"])
@jwt_required()
def stats():
    user, err = _require_admin()
    if err:
        return err

    total_predictions = PredictionHistory.query.count()
    total_users       = User.query.count()
    diabetic_count    = PredictionHistory.query.filter_by(prediction=1).count()
    avg_prob          = db.session.query(
        func.avg(PredictionHistory.probability)
    ).scalar() or 0.0

    risk_breakdown = {
        "Low":      PredictionHistory.query.filter_by(risk_level="Low").count(),
        "Moderate": PredictionHistory.query.filter_by(risk_level="Moderate").count(),
        "High":     PredictionHistory.query.filter_by(risk_level="High").count(),
    }

    # Daily prediction counts for last 7 days
    from datetime import datetime, timedelta
    daily = []
    for i in range(6, -1, -1):
        day   = datetime.utcnow().date() - timedelta(days=i)
        count = PredictionHistory.query.filter(
            func.date(PredictionHistory.timestamp) == day
        ).count()
        daily.append({"date": str(day), "count": count})

    return jsonify({
        "total_predictions": total_predictions,
        "total_users":       total_users,
        "diabetic_count":    diabetic_count,
        "diabetic_rate":     round(diabetic_count / max(total_predictions, 1), 4),
        "avg_probability":   round(float(avg_prob), 4),
        "risk_breakdown":    risk_breakdown,
        "daily_activity":    daily,
    }), 200


@admin_bp.route("/users", methods=["GET"])
@jwt_required()
def list_users():
    user, err = _require_admin()
    if err:
        return err

    users = User.query.order_by(User.created_at.desc()).all()
    return jsonify([{
        **u.to_dict(),
        "prediction_count": PredictionHistory.query.filter_by(user_id=u.id).count()
    } for u in users]), 200


@admin_bp.route("/promote/<int:user_id>", methods=["POST"])
@jwt_required()
def promote_user(user_id):
    admin, err = _require_admin()
    if err:
        return err

    target = User.query.get(user_id)
    if not target:
        return jsonify({"error": "User not found"}), 404

    target.is_admin = True
    db.session.commit()
    return jsonify({"message": f"{target.username} is now an admin"}), 200
