import numpy as np
import pytest

from src.api import create_app
from src.decision_engine import DecisionEngine


class FakeModel:
    def decision_function(self, X):
        return X[:, -1]


class FakeCalibrator:
    def predict_proba(self, margins):
        risk = np.where(margins[:, 0] > 0, 0.8, 0.1)
        return np.column_stack((1 - risk, risk))


def transaction(event_id, step, index):
    return {
        "event_id": event_id,
        "source_row_index": index,
        "step": step,
        "type": "TRANSFER",
        "amount": 100.0,
        "nameOrig": f"C{index + 1}",
        "nameDest": "C500",
        "oldbalanceOrg": 200.0,
    }


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    # Legacy API behavior tests run without auth, even if the developer has
    # FGZ_AUTH_PASSWORD set in their shell for the protected demo server.
    monkeypatch.delenv("FGZ_AUTH_PASSWORD", raising=False)
    engine = DecisionEngine(
        tmp_path / "api.sqlite3",
        FakeModel(),
        FakeCalibrator(),
        0.5,
    )

    app = create_app(engine)
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client, engine

    engine.close()


def test_health(api_client):
    client, _ = api_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json["healthy"] is True


def test_score_and_duplicate(api_client):
    client, _ = api_client
    event = transaction("E1", 1, 0)

    first = client.post("/transactions/score", json=event)
    second = client.post("/transactions/score", json=event)

    assert first.status_code == 201
    assert first.json["decision"] == "ALLOW"
    assert first.json["duplicate"] is False

    assert second.status_code == 200
    assert second.json["duplicate"] is True
    assert second.json["risk"] == first.json["risk"]


def test_conflicting_duplicate(api_client):
    client, _ = api_client
    event = transaction("E1", 1, 0)

    client.post("/transactions/score", json=event)

    changed = dict(event)
    changed["amount"] = 500.0

    response = client.post(
        "/transactions/score", json=changed
    )

    assert response.status_code == 409


def test_out_of_order_transaction(api_client):
    client, _ = api_client

    client.post(
        "/transactions/score",
        json=transaction("E1", 3, 0),
    )

    response = client.post(
        "/transactions/score",
        json=transaction("E2", 2, 1),
    )

    assert response.status_code == 409


def test_invalid_request(api_client):
    client, _ = api_client

    response = client.post(
        "/transactions/score",
        json={"type": "TRANSFER"},
    )

    assert response.status_code == 400


def test_recent_transactions_and_stats(api_client):
    client, _ = api_client

    client.post(
        "/transactions/score",
        json=transaction("E1", 1, 0),
    )
    client.post(
        "/transactions/score",
        json=transaction("E2", 2, 1),
    )

    recent = client.get("/transactions/recent?limit=10")
    stats = client.get("/stats")

    assert recent.status_code == 200
    assert len(recent.json["transactions"]) == 2
    assert recent.json["transactions"][0]["event_id"] == "E2"

    assert stats.status_code == 200
    assert stats.json["total"] == 2
    assert stats.json["allowed"] == 1
    assert stats.json["reviewed"] == 1
    assert stats.json["automatic_blocking"] is False


def test_unhealthy_engine_returns_503(api_client):
    client, engine = api_client

    engine.healthy = False

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json["healthy"] is False


def test_dashboard(api_client):
    client, _ = api_client

    response = client.get("/")

    assert response.status_code == 200
    assert b"FraudGuard Z" in response.data
    assert b"Score transaction" in response.data