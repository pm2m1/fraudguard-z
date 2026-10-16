"""End-to-end WSGI factory tests using fake ML models and real private SQLite."""
import base64

import numpy as np

import src.serve as serve


class FakeModel:
    n_features_in_ = 8

    def decision_function(self, X):
        return np.asarray(X)[:, -1]


class FakeCalibrator:
    n_features_in_ = 1

    def predict_proba(self, margins):
        p = np.full(len(margins), 0.1)
        return np.column_stack((1-p, p))


def _auth():
    token = base64.b64encode(b"fraudguard:some-strong-test-password").decode()
    return {"Authorization": f"Basic {token}"}


def _txn():
    return {
        "event_id": "wsgi-test-e1", "step": 1, "source_row_index": 0,
        "type": "TRANSFER", "amount": 75.0,
        "oldbalanceOrg": 200.0,
        "nameOrig": "private-source-test",
        "nameDest": "private-target-test",
    }


def test_wsgi_factory_auth_privacy_and_restart(tmp_path, monkeypatch):
    db = tmp_path / "wsgi_private.sqlite3"
    monkeypatch.setenv("FGZ_DEPLOYMENT_MODE", "production")
    monkeypatch.setenv("FGZ_AUTH_PASSWORD", "some-strong-test-password")
    monkeypatch.setenv("FGZ_PRIVACY_KEY", "some-private-test-key-which-is-long-enough")
    monkeypatch.setenv("FGZ_DB_PATH", str(db))
    monkeypatch.setattr(
        serve, "load_model",
        lambda name: FakeModel() if name == "model_b.joblib" else FakeCalibrator(),
    )

    app = serve.build_app(production=True)
    app.config["TESTING"] = True
    try:
        with app.test_client() as client:
            assert client.get("/health").status_code == 401
            health = client.get("/health", headers=_auth())
            assert health.status_code == 200
            assert health.json["committed_events"] == 0
            assert health.headers["Cache-Control"] == "no-store"
            first = client.post("/transactions/score", headers=_auth(), json=_txn())
            assert first.status_code == 201
            assert first.json["duplicate"] is False
        persisted = app.extensions["decision_engine"].db.execute(
            "SELECT originator, recipient FROM decisions"
        ).fetchone()
        assert persisted["originator"].startswith("acct_v1_")
        assert persisted["recipient"].startswith("acct_v1_")
        assert "private-source-test" not in str(tuple(persisted))
    finally:
        app.extensions["decision_engine"].close()

    restarted = serve.build_app(production=True)
    restarted.config["TESTING"] = True
    try:
        with restarted.test_client() as client:
            health = client.get("/health", headers=_auth())
            assert health.status_code == 200
            assert health.json["committed_events"] == 1
            same_event = client.post(
                "/transactions/score", headers=_auth(), json=_txn()
            )
            assert same_event.status_code == 200
            assert same_event.json["duplicate"] is True
    finally:
        restarted.extensions["decision_engine"].close()
