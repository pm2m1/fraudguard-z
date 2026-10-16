import hashlib
import json
import sqlite3

import pytest

from src.full_context_journal import FullContextJournal
from src.full_context_evidence import canonical_evidence_json

KEY = "separate-full-context-demo-key-with-32-bytes"


def event(i):
    return {
        "event_id": f"e{i}", "source_row_index": i-1, "step": i,
        "type": "TRANSFER", "amount": 20, "oldbalanceOrg": 100,
        "nameOrig": "raw-sender", "nameDest": "raw-destination",
    }


def journal(path):
    return FullContextJournal(
        path, lambda x: .02, lambda risk: "ALLOW",
        model_version="model-v-test", policy_version="policy-v-test",
        privacy_key=KEY,
    )


def test_evidence_persists_atomically_duplicate_and_restart(tmp_path):
    path = tmp_path / "new-evidence.sqlite3"
    j = journal(path)
    try:
        first = j.process(event(1))
        saved = j.db.execute(
            "SELECT evidence_json,evidence_sha256 FROM decisions WHERE event_id='e1'"
        ).fetchone()
        assert hashlib.sha256(saved["evidence_json"].encode()).hexdigest() == saved["evidence_sha256"]
        assert json.loads(saved["evidence_json"]) == first["evidence"]
        duplicate = j.process(event(1))
        assert duplicate["duplicate"] is True
        assert duplicate["evidence"] == first["evidence"]
        assert j.status()["committed_events"] == 1
    finally:
        j.close()
    again = journal(path)
    try:
        assert again.process(event(1))["evidence"] == first["evidence"]
        later = again.process(event(2))
        assert later["evidence"]["items"][0]["observed"] == 1
    finally:
        again.close()


def test_tampered_evidence_causes_fail_closed_restart(tmp_path):
    path = tmp_path / "tamper.sqlite3"
    j = journal(path)
    j.process(event(1))
    j.close()
    db = sqlite3.connect(path)
    db.execute("UPDATE decisions SET evidence_json='{}'")
    db.commit()
    db.close()
    with pytest.raises(RuntimeError, match="evidence integrity"):
        journal(path)


def test_legacy_full_context_schema_not_silently_migrated(tmp_path):
    path = tmp_path / "v1.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE decisions (event_id TEXT PRIMARY KEY)")
    db.execute("CREATE TABLE full_context_meta (singleton INTEGER PRIMARY KEY, schema_id TEXT)")
    db.commit()
    db.close()
    with pytest.raises(RuntimeError, match="decision schema"):
        journal(path)
    db = sqlite3.connect(path)
    try:
        assert [row[1] for row in db.execute("PRAGMA table_info(decisions)")] == ["event_id"]
    finally:
        db.close()
