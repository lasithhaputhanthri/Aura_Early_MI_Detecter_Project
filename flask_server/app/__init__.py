from flask import Flask
from app.routes import bp as routes_bp
import os

def create_app():
    app = Flask(__name__)

    # Configuration
    upload_dir = os.path.join(os.getcwd(), 'uploads')
    app.config['UPLOAD_FOLDER'] = upload_dir
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

    # Ensure upload directory exists
    os.makedirs(upload_dir, exist_ok=True)

    # Register blueprints (routes)
    app.register_blueprint(routes_bp)

    return app
