"""
app/__init__.py — Flask application factory.
"""
import logging
from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager

from app.models.db_models import db
from config import config_map


def create_app(env="default"):
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    cfg = config_map.get(env, config_map["default"])
    app.config.from_object(cfg)

    # Extensions
    db.init_app(app)
    CORS(app, resources={r"/api/*": {"origins": "*"}, r"/auth/*": {"origins": "*"}})
    JWTManager(app)

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if app.config["DEBUG"] else logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    # Blueprints
    from app.routes.auth    import auth_bp
    from app.routes.predict import predict_bp
    from app.routes.pages   import pages_bp
    from app.routes.admin   import admin_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(pages_bp)
    app.register_blueprint(admin_bp)

    with app.app_context():
        db.create_all()

    return app
