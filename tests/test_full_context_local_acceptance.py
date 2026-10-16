import csv
import json

import pytest

from src import full_context_local_acceptance as acceptance
from src.full_context_api import create_full_context_app
from src.full_context_journal import FullContextJournal


PASSWORD = "local-acceptance-password-123"
KEY = "local-acceptance-privacy-key-123"


def make_csv(path, n=8, step=1):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "step", "type", "amount", "oldbalanceOrg", "nameOrig", "nameDest",
            "isFraud",
        ])
        writer.writeheader()
        for i in range(n):
            writer.writerow({
                "step": step, "type": "TRANSFER", "amount": i+1,
                "oldbalanceOrg": 100, "nameOrig": f"sender-{i%3}",
                "nameDest": f"dest-{i%2}", "isFraud": 1,
            })


def test_bounded_acceptance_with_restart_and_no_overwrite(tmp_path, monkeypatch):
    source = tmp_path/"small.csv"
    make_csv(source)
    def open_lab(db_path, privacy_key, password):
        journal = FullContextJournal(
            db_path, lambda x: .001, lambda p: "ALLOW",
            model_version="SYNTHETIC_TEST_ONLY",
            policy_version="TEST_ONLY", privacy_key=privacy_key
        )
        return create_full_context_app(
            journal, auth_password=password, scoring_source="TEST_ONLY"
        ), journal
    monkeypatch.setattr(acceptance, "open_local_lab", open_lab)
    dest = tmp_path/"report.json"
    report = acceptance.run(source, dest, 8, PASSWORD, KEY)
    assert report["transaction_count"] == 8
    assert report["restart_count"] == 1
    assert report["reviewed"] == 0
    assert report["blocked"] == 0
    assert report["post_latency_ms"]["p99"] >= 0
    assert json.loads(dest.read_text())["status"] == "LOCAL_ENGINEERING_ACCEPTANCE"
    with pytest.raises(FileExistsError):
        acceptance.run(source, dest, 8, PASSWORD, KEY)


def test_acceptance_rejects_boundary_and_unsafe_limits(tmp_path, monkeypatch):
    source = tmp_path/"late.csv"
    make_csv(source, step=239)
    with pytest.raises(RuntimeError, match="boundary crossed"):
        list(acceptance.source_events(source, 2))
    with pytest.raises(ValueError, match="2..2000"):
        list(acceptance.source_events(source, 2001))
    with pytest.raises(RuntimeError, match="password"):
        acceptance.run(source, tmp_path/"missing.json", 2, "", KEY)


def test_prefix_parses_no_label_as_feature(tmp_path):
    source = tmp_path/"small.csv"
    make_csv(source, n=3)
    rows = list(acceptance.source_events(source, 3))
    assert [x["source_row_index"] for x in rows] == [0, 1, 2]
    assert all("isFraud" not in x for x in rows)
