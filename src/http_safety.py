"""Minimal HTTP hardening for local FraudGuard Z endpoints.

This is *not* a substitute for production TLS, a reverse proxy, authentication
policy, network isolation, rate limiting, or encrypted database storage.
"""
from flask import jsonify, request

MAX_TRANSACTION_REQUEST_BYTES = 16 * 1024


def configure_http_safety(app):
    # Applies to all requests; protects against accidentally enormous JSON
    # transaction bodies and avoids unchecked consumption by Flask parsers.
    app.config["MAX_CONTENT_LENGTH"] = MAX_TRANSACTION_REQUEST_BYTES

    @app.before_request
    def require_transaction_json():
        if request.path == "/transactions/score" and request.method == "POST":
            if request.mimetype != "application/json":
                return jsonify({
                    "error": "Transaction scoring requires application/json"
                }), 415
        return None

    @app.errorhandler(413)
    def transaction_too_large(error):
        return jsonify({"error": "Request payload exceeds permitted size"}), 413

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        # Basic auth and transaction responses must not be cached by intermediaries.
        response.headers["Cache-Control"] = "no-store"
        return response
