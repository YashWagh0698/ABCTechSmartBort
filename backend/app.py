"""Flask application entry point.

This file ONLY creates the app, registers routes and starts the server.
No parsing, embedding, ChromaDB or LLM logic belongs here.
"""
from flask import Flask, jsonify, request

from backend.config import Config
from backend.routes.health import health_bp
from backend.routes.ingestion import ingestion_bp
from backend.routes.sources import sources_bp
from backend.services import Services
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def create_app(services: Services = None) -> Flask:
    """Application factory: builds and returns a configured Flask app.

    `services` lets tests pass in fake services. Normally it is created empty and
    the embedding model / vector database are loaded only when first needed.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

    app.extensions["services"] = services or Services()

    # All API routes live under /api
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(ingestion_bp, url_prefix="/api")
    app.register_blueprint(sources_bp, url_prefix="/api")

    @app.before_request
    def reject_oversized_requests():
        # Flask's MAX_CONTENT_LENGTH only fires when a route READS the body. Checking the
        # declared size here makes the limit apply to every route, including ones that ignore it.
        if request.content_length is not None and request.content_length > Config.MAX_CONTENT_LENGTH:
            return jsonify({"error": "Request too large"}), 413

    # Return JSON (not HTML) for errors so the frontend can read them
    @app.errorhandler(404)
    def not_found(_e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(_e):
        return jsonify({"error": "Method not allowed for this URL"}), 405

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({"error": "Request too large"}), 413

    @app.errorhandler(500)
    def server_error(_e):
        logger.exception("Unhandled server error")
        return jsonify({"error": "Internal server error"}), 500

    logger.info("ABC Tech Smart Bot started (env=%s)", Config.APPLICATION_ENV)
    return app


if __name__ == "__main__":
    create_app().run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
