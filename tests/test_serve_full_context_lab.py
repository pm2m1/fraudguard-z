import base64
import json

import numpy as np
import pytest

from src import serve_full_context_lab as lab


KEY = "separate-full-context-demo-key-with-32-bytes"
PASSWORD = "long-local-lab-demo-password"


def transaction(i, *, amount=10):
    return {
        "event_id": f"lab-{i}", "source_row_index": i-1, "step": i,
        "type": "TRANSFER", "amount": amount, "oldbalanceOrg": 100,
        "nameOrig": "sender", "nameDest": "recipient",
    }


def test_local_lab_needs_secrets_before_artifact_use(tmp_path):
    with pytest.raises(RuntimeError, match="password"):
        lab.open_local_lab(tmp_path/"a.sqlite3", KEY, "")
    with pytest.raises(RuntimeError, match="pseudonymization"):
        lab.open_local_lab(tmp_path/"b.sqlite3", "", PASSWORD)


def test_restart_duplicate_and_durable_review_quota(tmp_path, monkeypatch):
    # Only the integration fixture is synthetic. The launcher normally
    # requires checksum-verified on-disk fitted model/calibrator/policy.
    monkeypatch.setattr(
        lab, "load_frozen_lab_artifacts",
        lambda *a, **k: (lambda x: .9, .5, "policy-v2-test", "model-test")
    )
    path = tmp_path/"new-lab.sqlite3"
    app, journal = lab.open_local_lab(path, KEY, PASSWORD)
    token = base64.b64encode(f"fraudguard:{PASSWORD}".encode()).decode()
    headers = {"Authorization": "Basic "+token}
    try:
        with app.test_client() as client:
            for i in range(1, 101):
                response = client.post("/transactions/score", headers=headers,
                                       json=transaction(i))
                assert response.status_code == 201
                if i < 100:
                    assert response.json["decision"] == "ALLOW"
                else:
                    assert response.json["decision"] == "REVIEW"
            duplicate = client.post("/transactions/score", headers=headers,
                                    json=transaction(100))
            assert duplicate.status_code == 200
            assert duplicate.json["duplicate"] is True
            assert journal.status()["committed_events"] == 100
            assert journal.status()["reviewed_events"] == 1
            assert client.get("/stats", headers=headers).json["reviewed"] == 1
    finally:
        journal.close()
    app2, journal2 = lab.open_local_lab(path, KEY, PASSWORD)
    try:
        with app2.test_client() as client:
            # Review quota is based on COMMITTED transactions, not event IDs
            # or source_row_index. Replay the actual 100 subsequent events.
            for i in range(101, 201):
                response = client.post(
                    "/transactions/score", headers=headers,
                    json=transaction(i),
                )
                assert response.status_code == 201
                assert response.json["decision"] == (
                    "REVIEW" if i == 200 else "ALLOW"
                )
            assert journal2.status()["committed_events"] == 200
            assert journal2.status()["reviewed_events"] == 2
            assert client.get("/stats", headers=headers).json["reviewed"] == 2
            assert client.get("/stats", headers=headers).json["blocked"] == 0
    finally:
        journal2.close()


def test_fail_closed_on_count_divergence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lab, "load_frozen_lab_artifacts",
        lambda *a, **k: (lambda x: .99, .5, "p", "m")
    )
    app, journal = lab.open_local_lab(tmp_path/"mismatch.sqlite3", KEY, PASSWORD)
    try:
        # A corrupted counter projection must fail closed.
        journal.reviewed_events = 1
        with pytest.raises(RuntimeError, match="decision count mismatch"):
            journal.process(transaction(1))
        assert journal.status()["healthy"] is False
    finally:
        journal.close()


def test_main_rejects_unacknowledged_and_production(monkeypatch):
    monkeypatch.delenv("FGZ_FULL_CONTEXT_LAB_ACK", raising=False)
    with pytest.raises(RuntimeError, match="SIMULATION_ONLY"):
        lab.main()
    monkeypatch.setenv("FGZ_FULL_CONTEXT_LAB_ACK", "SIMULATION_ONLY")
    monkeypatch.setenv("FGZ_DEPLOYMENT_MODE", "production")
    with pytest.raises(RuntimeError, match="prohibited"):
        lab.main()


def test_sparse_event_ids_cannot_inflate_committed_review_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lab, "load_frozen_lab_artifacts",
        lambda *a, **k: (lambda x: .9, .5, "policy-v2-test", "model-test")
    )
    app, journal = lab.open_local_lab(
        tmp_path / "sparse-ordinal.sqlite3", KEY, PASSWORD
    )
    token = base64.b64encode(f"fraudguard:{PASSWORD}".encode()).decode()
    headers = {"Authorization": "Basic " + token}
    try:
        with app.test_client() as client:
            for ordinal in (10_000, 20_000):
                response = client.post(
                    "/transactions/score", headers=headers,
                    json=transaction(ordinal),
                )
                assert response.status_code == 201
                assert response.json["decision"] == "ALLOW"
            assert journal.status()["committed_events"] == 2
            stats = client.get("/stats", headers=headers).json
            assert stats["reviewed"] == 0
            assert stats["blocked"] == 0
    finally:
        journal.close()
