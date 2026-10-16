"""Optional HTTP Basic authentication for the local FraudGuard dashboard.

A password must be configured before access through a TLS reverse proxy.
The built-in development server binds only to 127.0.0.1.
"""
import os
from secrets import compare_digest

from flask import jsonify, request


def configure_auth(app, password=None):
    if password is None:
        password = os.environ.get("FGZ_AUTH_PASSWORD", "")
    if password and len(password) < 16:
        raise RuntimeError("FGZ_AUTH_PASSWORD must contain at least 16 characters")

    app.extensions["fraudguard_auth_configured"] = bool(password)

    @app.before_request
    def require_password():
        if not password:
            return None  # Local development only. Never expose unprotected.
        credentials = request.authorization
        authenticated = (
            credentials is not None
            and credentials.type == "basic"
            and compare_digest(credentials.username or "", "fraudguard")
            and compare_digest(credentials.password or "", password)
        )
        if authenticated:
            return None
        response = jsonify({"error": "Authentication required"})
        response.status_code = 401
        response.headers["WWW-Authenticate"] = 'Basic realm="FraudGuard Z", charset="UTF-8"'
        return response

    @app.after_request
    def disable_caching(response):
        response.headers["Cache-Control"] = "no-store"
        return response
