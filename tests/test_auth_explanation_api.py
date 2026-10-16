import base64
import json

import numpy as np
import pytest

from src.api import create_app
from src.decision_engine import DecisionEngine


class FakeModel:
    def decision_function(self, X):
        return X[:, -1]


class FakeCalibrator:
    def predict_proba(self, margins):
        p = np.where(margins[:, 0] > 0, 0.8, 0.1)
        return np.column_stack((1-p, p))


@pytest.fixture
def guarded(tmp_path, monkeypatch):
    monkeypatch.setenv("FGZ_AUTH_PASSWORD", "this-is-a-demo-password-of-sufficient-length")
    engine = DecisionEngine(tmp_path / "secured.sqlite3", FakeModel(), FakeCalibrator(), 0.5)
    app = create_app(engine)
    app.testing = True
    yield app.test_client(), engine
    engine.close()


def basic_auth_header(password):
    token = base64.b64encode(f"fraudguard:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_password_required_on_all_routes(guarded):
    client, _ = guarded
    for path in ("/", "/health", "/stats", "/transactions/recent",
                 "/transactions/E1/explanation"):
        response = client.get(path)
        assert response.status_code == 401
        assert response.headers["Cache-Control"] == "no-store"


def test_invalid_password_rejected(guarded):
    client, _ = guarded
    assert client.get("/health", headers=basic_auth_header("wrong")).status_code == 401


def test_authorized_access_and_missing_explanation(guarded):
    client, _ = guarded
    headers = basic_auth_header("this-is-a-demo-password-of-sufficient-length")
    assert client.get("/health", headers=headers).status_code == 200
    assert client.get("/transactions/E1/explanation", headers=headers).status_code == 404


def test_fake_model_explanation_fails_closed(guarded):
    client, _ = guarded
    headers = basic_auth_header("this-is-a-demo-password-of-sufficient-length")
    txn = {"event_id":"E1", "source_row_index":0, "step":1, "type":"TRANSFER",
           "amount":100, "nameOrig":"A", "nameDest":"B", "oldbalanceOrg":200}
    response = client.post("/transactions/score", json=txn, headers=headers)
    assert response.status_code == 201
    explanation = client.get("/transactions/E1/explanation", headers=headers)
    assert explanation.status_code == 422


def test_short_password_fails_at_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("FGZ_AUTH_PASSWORD", "short")
    engine = DecisionEngine(tmp_path / "weak.sqlite3", FakeModel(), FakeCalibrator(), 0.5)
    try:
        with pytest.raises(RuntimeError, match="at least 16"):
            create_app(engine)
    finally:
        engine.close()
