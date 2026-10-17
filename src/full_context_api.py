"""Separate authenticated localhost API for full-context journal v2.

This app factory does NOT select, load or promote a fraud model or policy.
Its only input is an initialized experimental FullContextJournal.
Never expose through the Flask development server on a public interface.
"""
import hashlib
import json
import sqlite3

import numpy as np

from flask import Flask, jsonify, render_template, request

from .auth import configure_auth
from .full_context_evidence import canonical_evidence_json
from .full_context_journal import (
    FEATURE_SCHEMA_SHA256, FullContextJournal, SCHEMA_ID,
)


def _verified_evidence(row):
    """Check stored canonical evidence, its digest, and feature derivation."""
    saved = row["evidence_json"]
    if hashlib.sha256(saved.encode("utf-8")).hexdigest() != row["evidence_sha256"]:
        raise RuntimeError("Stored evidence checksum mismatch")
    features = json.loads(row["features_json"])
    expected = canonical_evidence_json(features)
    if saved != expected or row["feature_schema_sha256"] != FEATURE_SCHEMA_SHA256:
        raise RuntimeError("Stored evidence/feature mismatch")
    return json.loads(saved)


def create_full_context_app(
    journal: FullContextJournal, *, auth_password: str,
    scoring_source: str = "UNAPPROVED_EXPERIMENTAL_SIMULATION",
    attribution_provider=None,
):
    if not isinstance(journal, FullContextJournal):
        raise TypeError("This API requires an isolated FullContextJournal")
    if not isinstance(auth_password, str) or len(auth_password) < 16:
        raise RuntimeError("Full-context API requires an auth password >=16 characters")
    if not isinstance(scoring_source, str) or not scoring_source.strip():
        raise ValueError("Scoring source must be explicitly labeled")
    if attribution_provider is not None and not callable(attribution_provider):
        raise TypeError("Attribution provider must be callable")
    app = Flask(__name__)
    app.json.sort_keys = False
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024
    # Local simulator only: never accept arbitrary Host authority values.
    app.config["TRUSTED_HOSTS"] = ["localhost", "127.0.0.1"]

    @app.before_request
    def same_origin_write_guard():
        # A browser can retain Basic credentials; deny cross-origin mutations.
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin:
                from urllib.parse import urlsplit
                try:
                    parsed = urlsplit(origin)
                    allowed = (
                        parsed.scheme in ("http", "https")
                        and parsed.netloc == request.host
                    )
                except ValueError:
                    allowed = False
                if not allowed:
                    return jsonify({"error": "Cross-origin writes are disabled"}), 403

    @app.after_request
    def browser_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self'; form-action 'self'; "
            "base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    app.extensions["full_context_journal"] = journal
    configure_auth(app, password=auth_password)

    def unavailable():
        return jsonify({"error": "Full-context journal unavailable; no decision made"}), 503

    @app.get("/")
    def dashboard():
        return render_template("full_context_dashboard.html")

    @app.get("/health")
    def health():
        try:
            status = journal.status()
        except (sqlite3.Error, RuntimeError):
            return unavailable()
        return jsonify({
            "status": "ok" if status["healthy"] else "unavailable",
            "simulation_only": True,
            "scoring_source": scoring_source,
            "journal_schema_id": SCHEMA_ID,
            **status,
        }), 200 if status["healthy"] else 503

    @app.post("/transactions/score")
    def score():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected a JSON transaction object"}), 400
        try:
            result = journal.process(payload)
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            message = str(error)
            conflict = (
                "Event ID already exists" in message or
                "New transactions must be in replay order" in message
            )
            return jsonify({"error": message}), 409 if conflict else 400
        except (RuntimeError, sqlite3.Error):
            return unavailable()
        return jsonify({
            **result,
            "simulation_only": True,
            "scoring_source": scoring_source,
            "model_version": journal.model_version,
            "policy_version": journal.policy_version,
            "block_enabled": False,
            "risk_is_real_world_probability": False,
            "model_attribution_provided": False,
        }), 200 if result["duplicate"] else 201

    @app.get("/transactions/recent")
    def recent():
        try:
            limit = int(request.args.get("limit", 20))
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid limit"}), 400
        if not 1 <= limit <= 100:
            return jsonify({"error": "Limit must be between 1 and 100"}), 400
        try:
            with journal.lock:
                if not journal.healthy:
                    return unavailable()
                rows = journal.db.execute("""
                    SELECT event_id, step, source_row_index, txn_type, amount,
                           calibrated_risk, decision, model_version, policy_version,
                           feature_schema_sha256, features_json,
                           evidence_json, evidence_sha256
                    FROM decisions
                    ORDER BY step DESC, source_row_index DESC LIMIT ?
                """, (limit,)).fetchall()
                records = []
                for row in rows:
                    evidence = _verified_evidence(row)
                    lookup = {e["code"]: e["observed"] for e in evidence["items"]}
                    records.append({
                        "event_id": row["event_id"],
                        "step": row["step"],
                        "source_row_index": row["source_row_index"],
                        "type": row["txn_type"],
                        "amount": row["amount"],
                        "risk": row["calibrated_risk"],
                        "decision": row["decision"],
                        "model_version": row["model_version"],
                        "policy_version": row["policy_version"],
                        "recipient_prior_originators":
                            lookup["RECIPIENT_PRIOR_ORIGINATORS"],
                        "pair_first_observed":
                            lookup.get("PAIR_FIRST_OBSERVED") is True,
                    })
            return jsonify({
                "transactions": records, "simulation_only": True,
                "scoring_source": scoring_source,
            }), 200
        except (sqlite3.Error, RuntimeError, ValueError, KeyError, TypeError):
            return unavailable()

    @app.get("/transactions/<path:event_id>/evidence")
    def evidence(event_id):
        try:
            with journal.lock:
                if not journal.healthy:
                    return unavailable()
                row = journal.db.execute("""
                    SELECT event_id, step, source_row_index,
                           calibrated_risk, decision, model_version, policy_version,
                           feature_schema_sha256, features_json,
                           evidence_json, evidence_sha256
                    FROM decisions WHERE event_id=?
                """, (event_id,)).fetchone()
                if row is None:
                    return jsonify({"error": "Transaction not found"}), 404
                stored = _verified_evidence(row)
            return jsonify({
                "event_id": row["event_id"],
                "decision": row["decision"],
                "risk": row["calibrated_risk"],
                "evidence": stored,
                "model_version": row["model_version"],
                "policy_version": row["policy_version"],
                "model_attribution_provided": False,
                "simulation_only": True,
                "scoring_source": scoring_source,
                "block_enabled": False,
            }), 200
        except (sqlite3.Error, RuntimeError, KeyError, ValueError, TypeError):
            return unavailable()

    @app.get("/transactions/<path:event_id>/attribution")
    def attribution(event_id):
        # Unlike observational evidence, attribution needs a verified model.
        if attribution_provider is None:
            return jsonify({"error": "Model attribution unavailable"}), 501
        try:
            with journal.lock:
                if not journal.healthy:
                    return unavailable()
                row = journal.db.execute("""
                    SELECT event_id, calibrated_risk, model_version, policy_version,
                           feature_schema_sha256, features_json,
                           evidence_json, evidence_sha256
                    FROM decisions WHERE event_id=?
                """, (event_id,)).fetchone()
                if row is None:
                    return jsonify({"error": "Transaction not found"}), 404
                _verified_evidence(row)
                if (row["model_version"] != journal.model_version or
                        row["policy_version"] != journal.policy_version):
                    raise RuntimeError("Stored model/policy identity mismatch")
                features = np.asarray(json.loads(row["features_json"]),
                                      dtype=np.float32)
                if features.shape != (21,) or not np.isfinite(features).all():
                    raise RuntimeError("Invalid stored feature vector")
                margin_explanation = attribution_provider(features)
                recomputed = float(journal.score_calibrated(
                    features.reshape(1, 21)))
                if (not np.isfinite(recomputed) or
                        abs(recomputed - row["calibrated_risk"]) > 1e-8):
                    raise RuntimeError("Stored score/model mismatch")
                return jsonify({
                    "event_id": row["event_id"],
                    "decision_margin_attribution": margin_explanation,
                    "calibrated_simulation_score": row["calibrated_risk"],
                    "model_version": row["model_version"],
                    "policy_version": row["policy_version"],
                    "attribution_kind": "precalibration_logistic_margin_only",
                    "not_fraud_causation": True,
                    "not_probability_contributions": True,
                    "simulation_only": True,
                    "block_enabled": False,
                }), 200
        except (sqlite3.Error, RuntimeError, ValueError, TypeError, KeyError,
                OverflowError, IndexError):
            return unavailable()

    @app.get("/stats")
    def stats():
        try:
            with journal.lock:
                if not journal.healthy:
                    return unavailable()
                row = journal.db.execute("""
                    SELECT COUNT(*) AS total,
                       COALESCE(SUM(CASE WHEN decision='ALLOW' THEN 1 ELSE 0 END),0) AS allowed,
                       COALESCE(SUM(CASE WHEN decision='REVIEW' THEN 1 ELSE 0 END),0) AS reviewed,
                       COALESCE(AVG(calibrated_risk),0) AS average_risk
                    FROM decisions
                """).fetchone()
            return jsonify({
                **dict(row),
                "blocked": 0,
                "block_enabled": False,
                "simulation_only": True,
                "scoring_source": scoring_source,
                "model_version": journal.model_version,
                "policy_version": journal.policy_version,
            }), 200
        except (sqlite3.Error, RuntimeError):
            return unavailable()

    return app
