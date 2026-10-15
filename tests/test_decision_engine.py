import numpy as np
import pytest

from src.decision_engine import DecisionEngine


class FakeModel:
    def decision_function(self, features):
        return features[:, -1]


class FakeCalibrator:
    def predict_proba(self, margins):
        risk = np.where(margins[:, 0] > 0, 0.8, 0.1)
        return np.column_stack((1 - risk, risk))


def transaction(event_id, step, index, recipient="X"):
    return {
        "event_id": event_id,
        "step": step,
        "source_row_index": index,
        "type": "TRANSFER",
        "amount": 100,
        "nameOrig": "A",
        "nameDest": recipient,
        "oldbalanceOrg": 500,
    }


def create_engine(path):
    return DecisionEngine(
        path, FakeModel(), FakeCalibrator(), 0.5
    )


@pytest.fixture
def engine(tmp_path):
    instance = create_engine(tmp_path / "events.sqlite3")
    yield instance
    instance.close()


def test_first_transaction(engine):
    result = engine.score(transaction("E1", 1, 0))

    assert result["decision"] == "ALLOW"
    assert result["n3"] == 0
    assert engine.status()["committed_events"] == 1


def test_recipient_history(engine):
    engine.score(transaction("E1", 1, 0))
    result = engine.score(transaction("E2", 2, 1))

    assert result["n3"] == 1
    assert result["decision"] == "REVIEW"


def test_duplicate_event(engine):
    first = transaction("E1", 1, 0)

    engine.score(first)
    duplicate = engine.score(first)

    assert duplicate["duplicate"] is True
    assert engine.status()["committed_events"] == 1

    next_result = engine.score(transaction("E2", 2, 1))
    assert next_result["n3"] == 1


def test_conflicting_duplicate(engine):
    first = transaction("E1", 1, 0)
    engine.score(first)

    changed = dict(first)
    changed["amount"] = 200

    with pytest.raises(ValueError):
        engine.score(changed)


def test_out_of_order_transaction(engine):
    engine.score(transaction("E1", 3, 0))

    with pytest.raises(ValueError):
        engine.score(transaction("E2", 2, 1))


def test_restart_rebuilds_history(tmp_path):
    path = tmp_path / "events.sqlite3"

    first = create_engine(path)
    first.score(transaction("E1", 1, 0))
    first.close()

    restarted = create_engine(path)

    result = restarted.score(transaction("E2", 2, 1))

    assert result["n3"] == 1
    assert result["decision"] == "REVIEW"
    assert restarted.status()["committed_events"] == 2

    restarted.close()


def test_same_step_uses_input_order(engine):
    engine.score(transaction("E1", 5, 0))
    result = engine.score(transaction("E2", 5, 1))

    assert result["n3"] == 1
    assert result["decision"] == "REVIEW"