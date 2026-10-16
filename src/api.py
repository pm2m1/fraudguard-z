
import sqlite3

from .auth import configure_auth
from .explanations import explain_record, ExplanationUnavailable

from flask import Flask, jsonify, render_template, request

def create_app(engine):
    app = Flask(__name__)
    app.json.sort_keys = False
    app.extensions["decision_engine"] = engine
    configure_auth(app)

    @app.get("/")
    def dashboard():
        return render_template("dashboard.html")

    @app.get("/health")
    def health():
        try:
            status = engine.status()
        except sqlite3.Error:
            return jsonify({
                "status": "unavailable"
            }), 503

        return jsonify({
            "status": "ok" if status["healthy"] else "unavailable",
            **status,
        }), 200 if status["healthy"] else 503

    @app.post("/transactions/score")
    def score_transaction():
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Expected a JSON transaction object"
            }), 400

        try:
            result = engine.score(data)

        except (ValueError, KeyError, TypeError, OverflowError) as error:
            message = str(error)

            conflict = (
                "Event ID already exists" in message
                or "New transactions must be in replay order" in message
            )

            return jsonify({
                "error": message
            }), 409 if conflict else 400

        except (RuntimeError, sqlite3.Error):
            return jsonify({
                "error": "Scoring is unavailable. Check engine health."
            }), 503

        return jsonify(result), 200 if result["duplicate"] else 201

    @app.get("/transactions/recent")
    def recent_transactions():
        try:
            limit = int(request.args.get("limit", 20))
        except (TypeError, ValueError):
            return jsonify({
                "error": "Invalid limit"
            }), 400

        if not 1 <= limit <= 100:
            return jsonify({
                "error": "Limit must be between 1 and 100"
            }), 400

        try:
            with engine.lock:
                rows = engine.db.execute("""
                    SELECT event_id, step, source_row_index,
                           txn_type, amount, n3, risk, decision
                    FROM decisions
                    ORDER BY step DESC, source_row_index DESC
                    LIMIT ?
                """, (limit,)).fetchall()

            return jsonify({
                "transactions": [dict(row) for row in rows]
            }), 200

        except sqlite3.Error:
            return jsonify({
                "error": "Could not read recent transactions"
            }), 503

    @app.get("/stats")
    def stats():
        try:
            with engine.lock:
                row = engine.db.execute("""
                    SELECT
                        COUNT(*) AS total,
                        SUM(CASE WHEN decision = 'ALLOW'
                            THEN 1 ELSE 0 END) AS allowed,
                        SUM(CASE WHEN decision = 'REVIEW'
                            THEN 1 ELSE 0 END) AS reviewed,
                        AVG(risk) AS average_risk
                    FROM decisions
                """).fetchone()

            return jsonify({
                "total": row["total"],
                "allowed": row["allowed"],
                "reviewed": row["reviewed"],
                "average_risk": row["average_risk"] or 0.0,
                "automatic_blocking": False,
            }), 200

        except sqlite3.Error:
            return jsonify({
                "error": "Could not calculate statistics"
            }), 503

    @app.get("/transactions/<path:event_id>/explanation")
    def explain_transaction(event_id):
        if not app.extensions.get("fraudguard_auth_configured"):
            return jsonify({"error": "Explanation access requires configured authentication"}), 403
        try:
            with engine.lock:
                record = engine.db.execute(
                    "SELECT event_id, features_json, risk, decision, schema_version "
                    "FROM decisions WHERE event_id = ?", (event_id,)
                ).fetchone()
                if record is None:
                    return jsonify({"error": "Transaction not found"}), 404
                explanation = explain_record(engine, record)
            return jsonify(explanation), 200
        except ExplanationUnavailable:
            return jsonify({"error": "Explanation unavailable for this model"}), 422
        except sqlite3.Error:
            return jsonify({"error": "Could not load transaction"}), 503

    return app
