import numpy as np
import pytest

from src.api import create_app
from src.decision_engine import DecisionEngine


class Model:
    def decision_function(self, X):
        return X[:, -1]


class Calibrator:
    def predict_proba(self, margins):
        p = np.full(len(margins), .1)
        return np.column_stack((1-p, p))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("FGZ_AUTH_PASSWORD", raising=False)
    engine = DecisionEngine(tmp_path / "safety.sqlite3", Model(), Calibrator(), .5)
    app = create_app(engine)
    app.config["TESTING"] = True
    with app.test_client() as session:
        yield session
    engine.close()


def test_security_headers_on_dashboard_and_health(client):
    for path in ("/", "/health"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Referrer-Policy"] == "no-referrer"
        assert response.headers["Cache-Control"] == "no-store"


def test_requires_json_media_type(client):
    response = client.post("/transactions/score", data="a=b")
    assert response.status_code == 415


def test_rejects_oversized_json(client):
    response = client.post("/transactions/score",
                           data='{"junk":"' + "x" * 20000 + '"}',
                           content_type="application/json")
    assert response.status_code == 413


def test_valid_json_still_scores(client):
    payload = {
        "event_id": "safety-1", "source_row_index": 0, "step": 1,
        "type": "TRANSFER", "amount": 20,
        "nameOrig": "A", "nameDest": "B", "oldbalanceOrg": 100
    }
    response = client.post("/transactions/score", json=payload)
    assert response.status_code == 201
    assert response.json["decision"] == "ALLOW"
