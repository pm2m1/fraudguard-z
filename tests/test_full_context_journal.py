import json
import sqlite3

import pytest

from src.full_context_journal import FullContextJournal, FULL_FEATURE_NAMES


KEY = "separate-full-context-demo-key-with-32-bytes"


def event(i, *, sender="sender-raw", dest="dest-raw", step=None, amount=20):
    return {
        "event_id": f"E{i}", "source_row_index": i - 1,
        "step": step if step is not None else i, "type": "TRANSFER",
        "amount": amount, "oldbalanceOrg": 100,
        "nameOrig": sender, "nameDest": dest,
    }


def scorer(x):
    assert x.shape == (1, len(FULL_FEATURE_NAMES))
    return 0.75 if x[0, 16] > 0 else 0.1


def policy(p):
    return "REVIEW" if p >= 0.5 else "ALLOW"


def journal(path, *, key=KEY, version="model-test-v1", policy_version="policy-test-v1"):
    return FullContextJournal(
        path, scorer, policy, privacy_key=key, model_version=version,
        policy_version=policy_version,
    )


def test_new_event_persists_all_features_and_is_idempotent(tmp_path):
    j = journal(tmp_path / "new.sqlite3")
    try:
        first = j.process(event(1))
        assert first["decision"] == "ALLOW"
        assert len(first["features"]) == 21
        assert first["features"]["first_time_recipient"] == 1
        assert j.process(event(1))["duplicate"]
        later = j.process(event(2))
        assert later["features"]["recipient_in_count_6h"] == 1
        assert j.status()["committed_events"] == 2
        stored = j.db.execute("SELECT originator,recipient FROM decisions LIMIT 1").fetchone()
        assert stored["originator"].startswith("acct_v1_")
        assert stored["recipient"].startswith("acct_v1_")
        assert "sender-raw" not in str(tuple(stored))
    finally:
        j.close()


def test_restart_rebuild_and_idempotency(tmp_path):
    path = tmp_path / "restart.sqlite3"
    first = journal(path)
    first.process(event(1))
    first.close()
    second = journal(path)
    try:
        assert second.process(event(1))["duplicate"]
        assert second.process(event(2))["features"]["sender_out_count_6h"] == 1
        assert second.status()["committed_events"] == 2
    finally:
        second.close()


def test_reject_out_of_order_and_conflicting_duplicate(tmp_path):
    j = journal(tmp_path / "order.sqlite3")
    try:
        j.process(event(2))
        with pytest.raises(ValueError, match="replay order"):
            j.process(event(1))
        with pytest.raises(ValueError, match="different data"):
            j.process(event(2, amount=50))
        assert j.status()["committed_events"] == 1
    finally:
        j.close()


def test_model_policy_and_key_mismatch(tmp_path):
    path = tmp_path / "versioned.sqlite3"
    j = journal(path)
    j.process(event(1))
    j.close()
    with pytest.raises(RuntimeError, match="schema/model/policy mismatch"):
        journal(path, version="different-model")
    with pytest.raises(RuntimeError, match="privacy mode or key mismatch"):
        journal(path, key="another-different-and-long-enough-secret-key")


def test_refuse_legacy_model_b_database(tmp_path):
    path = tmp_path / "old.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE decisions (event_id TEXT PRIMARY KEY)")
    db.commit()
    db.close()
    with pytest.raises(RuntimeError, match="legacy"):
        journal(path)
    con = sqlite3.connect(path)
    try:
        assert [x[1] for x in con.execute("PRAGMA table_info(decisions)")] == ["event_id"]
    finally:
        con.close()


def test_invalid_risk_and_unvalidated_block_fail_closed(tmp_path):
    path = tmp_path / "bad.sqlite3"
    j = FullContextJournal(
        path, lambda x: float("nan"), policy,
        model_version="v1", policy_version="p1", privacy_key=KEY,
    )
    try:
        with pytest.raises(ValueError, match="invalid risk"):
            j.process(event(1))
        assert not j.status()["healthy"]
        with pytest.raises(RuntimeError, match="unavailable"):
            j.process(event(1))
        j.rebuild()
        assert j.status()["healthy"]
    finally:
        j.close()
    j = FullContextJournal(
        tmp_path / "block.sqlite3", lambda x: .999,
        lambda p: "BLOCK", model_version="v1", policy_version="p1",
        privacy_key=KEY,
    )
    try:
        with pytest.raises(RuntimeError, match="BLOCK"):
            j.process(event(1))
        assert j.status()["committed_events"] == 0
    finally:
        j.close()


def test_postcommit_observation_failure_rebuilds(tmp_path, monkeypatch):
    j = journal(tmp_path / "crash.sqlite3")
    try:
        monkeypatch.setattr(j.history, "observe", lambda row: (_ for _ in ()).throw(RuntimeError("oops")))
        with pytest.raises(RuntimeError, match="Decision committed"):
            j.process(event(1))
        assert not j.status()["healthy"]
        assert j.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
        j.rebuild()
        assert j.process(event(1))["duplicate"]
        assert j.process(event(2))["features"]["recipient_in_count_6h"] == 1
    finally:
        j.close()


def test_tampered_saved_features_detected_on_restart(tmp_path):
    path = tmp_path / "tamper.sqlite3"
    j = journal(path)
    j.process(event(1))
    j.close()
    db = sqlite3.connect(path)
    db.execute("UPDATE decisions SET features_json='[]'")
    db.commit()
    db.close()
    with pytest.raises(RuntimeError, match="feature history mismatch"):
        journal(path)


def test_block_flag_is_disallowed(tmp_path):
    with pytest.raises(RuntimeError, match="BLOCK"):
        FullContextJournal(
            tmp_path / "never.sqlite3", scorer, policy,
            model_version="v", policy_version="p", privacy_key=KEY,
            block_enabled=True,
        )


@pytest.mark.parametrize("column, value", [
    ("evidence_json", "{}"),
    ("evidence_sha256", "0" * 64),
    ("feature_schema_sha256", "0" * 64),
    ("model_version", "unauthorized-model"),
    ("policy_version", "unauthorized-policy"),
    ("features_json", "[]"),
    ("calibrated_risk", 0.99),
])
def test_duplicate_tampering_fails_closed_on_disposable_db(tmp_path, column, value):
    path = tmp_path / "duplicate-tamper.sqlite3"
    j = journal(path)
    try:
        j.process(event(1))
        original_count = j.status()["committed_events"]
        j.db.execute(f"UPDATE decisions SET {column}=? WHERE event_id='E1'", (value,))
        if column == "calibrated_risk":
            # The durable journal does not authenticate the scored risk
            # independently; this limitation must be tested honestly.
            assert j.process(event(1))["duplicate"] is True
            assert j.status()["committed_events"] == original_count
        else:
            with pytest.raises(RuntimeError, match="integrity mismatch"):
                j.process(event(1))
            assert j.status()["healthy"] is False
            assert j.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == original_count
            with pytest.raises(RuntimeError, match="unavailable"):
                j.process(event(2))
    finally:
        j.close()


def test_valid_duplicate_remains_idempotent_after_integrity_guard(tmp_path):
    j = journal(tmp_path / "duplicate-clean.sqlite3")
    try:
        first = j.process(event(1))
        second = j.process(event(1))
        assert second["duplicate"] is True
        assert second["decision"] == first["decision"]
        assert second["evidence"] == first["evidence"]
        assert j.status()["healthy"] is True
        assert j.status()["committed_events"] == 1
    finally:
        j.close()
