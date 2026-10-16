import sqlite3

import numpy as np
import pytest

from src.decision_engine import DecisionEngine
from src.privacy import AccountPseudonymizer

KEY = "demonstration-privacy-key-with-at-least-32-bytes"
OTHER_KEY = "different-demonstration-key-with-32bytes"


class Model:
    def decision_function(self, X):
        return X[:, -1]


class Calibrator:
    def predict_proba(self, margins):
        risk = np.where(margins[:, 0] > 0, 0.8, 0.1)
        return np.column_stack((1-risk, risk))


def event(event_id, step):
    return {
        "event_id": event_id, "step": step, "source_row_index": step-1,
        "type": "TRANSFER", "amount": 100.0, "oldbalanceOrg": 300.0,
        "nameOrig": "sender-personal-123", "nameDest": "recipient-personal-987",
    }


def engine(path, key):
    return DecisionEngine(path, Model(), Calibrator(), 0.5, privacy_key=key)


def test_account_pseudonyms_are_deterministic_and_domain_separated():
    p = AccountPseudonymizer(KEY)
    assert p.encode("A") == p.encode("A")
    assert p.encode("A") != p.encode("B")
    assert p.encode("A") != AccountPseudonymizer(OTHER_KEY).encode("A")
    assert "A" not in p.encode("A")
    assert len(p.encode("A")) == len(p.encode("B"))


def test_raw_identifiers_absent_from_pseudonymous_database(tmp_path):
    path = tmp_path / "private.sqlite3"
    e = engine(path, KEY)
    try:
        first = e.score(event("E1", 1))
        assert first["n3"] == 0
        record = e.db.execute("SELECT originator, recipient FROM decisions").fetchone()
        assert record["originator"].startswith("acct_v1_")
        assert record["recipient"].startswith("acct_v1_")
        assert "personal" not in str(tuple(record))
        assert e.score(event("E1", 1))["duplicate"] is True
        assert e.score(event("E2", 2))["n3"] == 1
    finally:
        e.close()
    assert b"sender-personal-123" not in path.read_bytes()
    assert b"recipient-personal-987" not in path.read_bytes()


def test_rebuild_preserves_history_and_rejects_wrong_key(tmp_path):
    path = tmp_path / "rebuild.sqlite3"
    a = engine(path, KEY)
    a.score(event("E1", 1))
    a.close()
    with pytest.raises(RuntimeError, match="privacy mode or key mismatch"):
        engine(path, OTHER_KEY)
    b = engine(path, KEY)
    try:
        assert b.score(event("E2", 2))["n3"] == 1
    finally:
        b.close()


def test_cannot_enable_privacy_over_legacy_database(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    e = engine(path, None)
    e.score(event("E1", 1))
    e.close()
    with pytest.raises(RuntimeError, match="privacy mode or key mismatch"):
        engine(path, KEY)


def test_plain_mode_cannot_open_private_database(tmp_path):
    path = tmp_path / "private.sqlite3"
    e = engine(path, KEY)
    e.close()
    with pytest.raises(RuntimeError, match="privacy mode or key mismatch"):
        engine(path, None)


def test_short_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="32"):
        engine(tmp_path / "invalid.sqlite3", "short")
