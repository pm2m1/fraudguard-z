import base64
import sqlite3

import pytest

from src.full_context_api import create_full_context_app
from src.full_context_journal import FullContextJournal


PASSWORD = "local-demo-password-very-long"
KEY = "separate-full-context-demo-key-with-32-bytes"


def item(i):
    return {
        "event_id": f"test-{i}", "source_row_index": i-1, "step": i,
        "type": "TRANSFER", "amount": 20, "oldbalanceOrg": 100,
        "nameOrig": "secret-originator", "nameDest": "secret-recipient",
    }


@pytest.fixture
def session(tmp_path):
    journal = FullContextJournal(
        tmp_path/"api-v2.sqlite3", lambda x: .03, lambda r: "ALLOW",
        model_version="explicit-injected-test-model",
        policy_version="explicit-injected-test-policy", privacy_key=KEY,
    )
    app = create_full_context_app(journal, auth_password=PASSWORD,
                                  scoring_source="SYNTHETIC_TEST_FIXTURE")
    app.config["TESTING"] = True
    token = base64.b64encode(("fraudguard:"+PASSWORD).encode()).decode()
    with app.test_client() as client:
        yield client, journal, {"Authorization": "Basic "+token}
    journal.close()


def test_requires_auth_for_every_endpoint(session):
    client, _, auth = session
    for endpoint in ("/", "/health", "/stats", "/transactions/recent",
                     "/transactions/test-1/evidence"):
        assert client.get(endpoint).status_code == 401
    assert client.post("/transactions/score", json=item(1)).status_code == 401
    assert client.get("/health", headers=auth).status_code == 200


def test_dashboard_and_scoring_and_stored_evidence(session):
    client, journal, auth = session
    page = client.get("/", headers=auth)
    assert page.status_code == 200
    assert b"BLOCK DISABLED" in page.data
    assert b"Local simulation only" in page.data
    first = client.post("/transactions/score", json=item(1), headers=auth)
    assert first.status_code == 201
    assert first.json["simulation_only"] is True
    assert first.json["model_attribution_provided"] is False
    assert first.json["block_enabled"] is False
    assert first.json["evidence"]["items"][0]["observed"] == 0
    assert first.headers["Cache-Control"] == "no-store"
    again = client.post("/transactions/score", json=item(1), headers=auth)
    assert again.status_code == 200
    assert again.json["duplicate"] is True
    assert again.json["evidence"] == first.json["evidence"]
    second = client.post("/transactions/score", json=item(2), headers=auth)
    assert second.status_code == 201
    assert second.json["evidence"]["items"][0]["observed"] == 1
    row = client.get("/transactions/test-2/evidence", headers=auth)
    assert row.status_code == 200
    assert row.json["evidence"] == second.json["evidence"]
    recent = client.get("/transactions/recent", headers=auth)
    assert recent.json["transactions"][0]["event_id"] == "test-2"
    assert "originator" not in recent.json["transactions"][0]
    assert "secret-originator" not in recent.get_data(as_text=True)
    assert "secret-recipient" not in recent.get_data(as_text=True)
    stats = client.get("/stats", headers=auth).json
    assert stats["total"] == 2
    assert stats["allowed"] == 2
    assert stats["reviewed"] == 0
    assert stats["blocked"] == 0
    assert journal.status()["committed_events"] == 2


def test_validation_conflicts_and_unavailable_state(session):
    client, journal, auth = session
    assert client.post("/transactions/score", json=[], headers=auth).status_code == 400
    assert client.get("/transactions/recent?limit=101", headers=auth).status_code == 400
    assert client.get("/transactions/recent?limit=banana", headers=auth).status_code == 400
    assert client.get("/transactions/no-such/evidence", headers=auth).status_code == 404
    assert client.post("/transactions/score", json=item(2), headers=auth).status_code == 201
    assert client.post("/transactions/score", json=item(1), headers=auth).status_code == 409
    bad = dict(item(2))
    bad["amount"] = 99
    assert client.post("/transactions/score", json=bad, headers=auth).status_code == 409
    journal.healthy = False
    for endpoint in ("/health", "/stats", "/transactions/recent",
                     "/transactions/test-2/evidence"):
        assert client.get(endpoint, headers=auth).status_code == 503
    assert client.post("/transactions/score", json=item(3), headers=auth).status_code == 503


def test_evidence_tamper_fails_closed_on_read(session):
    client, journal, auth = session
    assert client.post("/transactions/score", json=item(1), headers=auth).status_code == 201
    journal.db.execute("UPDATE decisions SET evidence_json='{}' WHERE event_id='test-1'")
    assert client.get("/transactions/test-1/evidence", headers=auth).status_code == 503
    assert client.get("/transactions/recent", headers=auth).status_code == 503


def test_refuse_unauthenticated_factory(tmp_path):
    journal = FullContextJournal(
        tmp_path/"journal.sqlite3", lambda x: .01, lambda p: "ALLOW",
        model_version="m", policy_version="p", privacy_key=KEY
    )
    try:
        with pytest.raises(RuntimeError, match="auth password"):
            create_full_context_app(journal, auth_password="")
    finally:
        journal.close()


def test_dashboard_discloses_policy_limitations_and_external_assets(session):
    client, _, auth = session
    page = client.get("/", headers=auth)
    assert page.status_code == 200
    assert b"0 of 206 fraudulent CASH_OUT" in page.data
    assert b"NOT validated for real payments" in page.data
    assert b"full_context_dashboard.js" in page.data
    assert b"full_context_dashboard.css" in page.data
    assert b"<script>" not in page.data
    assert b"<style>" not in page.data
    assert "script-src 'self'" in page.headers["Content-Security-Policy"]
    assert page.headers["X-Frame-Options"] == "DENY"
    assert page.headers["Referrer-Policy"] == "no-referrer"
    script = client.get("/static/full_context_dashboard.js", headers=auth)
    style = client.get("/static/full_context_dashboard.css", headers=auth)
    assert script.status_code == style.status_code == 200
    assert b"renderEvidence" in script.data
    assert b".evidence-item" in style.data
    assert client.get("/static/full_context_dashboard.js").status_code == 401


def test_browser_cross_origin_post_denied_without_state_change(session):
    client, journal, auth = session
    forbidden = client.post(
        "/transactions/score", json=item(1),
        headers={**auth, "Origin": "https://attacker.example"}
    )
    assert forbidden.status_code == 403
    assert journal.status()["committed_events"] == 0
    allowed = client.post(
        "/transactions/score", json=item(1),
        headers={**auth, "Origin": "http://localhost"}
    )
    assert allowed.status_code == 201
    assert journal.status()["committed_events"] == 1


def test_untrusted_host_rejected(session):
    client, _, auth = session
    result = client.get("/health", headers={**auth, "Host": "attacker.example"})
    assert result.status_code == 400
    good = client.get("/health", headers={**auth, "Host": "127.0.0.1:5051"})
    assert good.status_code == 200


def test_margin_attribution_unavailable_without_verified_model(session):
    client, journal, auth = session
    assert client.post("/transactions/score", json=item(1), headers=auth).status_code == 201
    unavailable = client.get("/transactions/test-1/attribution", headers=auth)
    assert unavailable.status_code == 501
    assert client.get("/transactions/test-1/attribution").status_code == 401
    assert journal.status()["committed_events"] == 1


def test_verified_margin_attribution_read_only_and_fail_closed(tmp_path):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.full_context_calibration import INDICES
    from src.full_context_margin_attribution import explain_margin

    rng = np.random.default_rng(12345)
    features = rng.normal(size=(40, len(INDICES))).astype(np.float32)
    labels = np.array([0, 1] * 20)
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(features, labels)

    def score(matrix):
        return float(1 / (1 + np.exp(-model.decision_function(matrix[:, INDICES])[0])))

    j = FullContextJournal(
        tmp_path / "attribution.sqlite3", score, lambda _: "ALLOW",
        model_version="local-synthetic-logistic-fixture",
        policy_version="allow-fixture-v1", privacy_key=KEY,
    )
    app = create_full_context_app(
        j, auth_password=PASSWORD, scoring_source="SYNTHETIC_TEST_FIXTURE",
        attribution_provider=lambda v: explain_margin(model, v)
    )
    token = base64.b64encode(("fraudguard:" + PASSWORD).encode()).decode()
    auth = {"Authorization": "Basic " + token}
    try:
        with app.test_client() as client:
            assert client.post("/transactions/score", json=item(1),
                               headers=auth).status_code == 201
            before = j.status()["committed_events"]
            response = client.get("/transactions/test-1/attribution", headers=auth)
            assert response.status_code == 200
            info = response.json
            assert info["not_fraud_causation"] is True
            assert info["not_probability_contributions"] is True
            assert len(info["decision_margin_attribution"]["items"]) == len(INDICES)
            assert j.status()["committed_events"] == before
            assert client.get("/transactions/nonexistent/attribution",
                              headers=auth).status_code == 404
            j.db.execute(
                "UPDATE decisions SET calibrated_risk=.99 WHERE event_id='test-1'"
            )
            assert client.get("/transactions/test-1/attribution",
                              headers=auth).status_code == 503
    finally:
        j.close()
