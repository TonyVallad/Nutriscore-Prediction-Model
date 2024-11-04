from flask import Flask
from config import Config

def create_app():
    """
    Initializes and configures the Flask application.

    :return: The configured Flask application instance.
    """
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize the loading status
    app.config['loading_dataframe_status'] = {"complete": False}

    # Import and register blueprints
    from app.modules.routes import main
    app.register_blueprint(main)

    from app.modules.api_routes import api_bp
    app.register_blueprint(api_bp)

    return app
