import csv

import pytest

from src.replay import iter_transactions, percentile, run_replay


def make_csv(path):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg"
        ])
        writer.writeheader()
        writer.writerows([
            {"step": 1, "type": "TRANSFER", "amount": 10,
             "nameOrig": "A", "nameDest": "X", "oldbalanceOrg": 20},
            {"step": 1, "type": "CASH_OUT", "amount": 15,
             "nameOrig": "B", "nameDest": "X", "oldbalanceOrg": 30},
            {"step": 2, "type": "TRANSFER", "amount": 5,
             "nameOrig": "C", "nameDest": "Y", "oldbalanceOrg": 10},
        ])


def test_paysim_events_have_stable_ids_and_row_order(tmp_path):
    csv_path = tmp_path / "sample.csv"
    make_csv(csv_path)
    rows = list(iter_transactions(csv_path, 3))
    assert [row["event_id"] for row in rows] == ["paysim-0", "paysim-1", "paysim-2"]
    assert [row["source_row_index"] for row in rows] == [0, 1, 2]
    assert [row["step"] for row in rows] == [1, 1, 2]
    assert "isFraud" not in rows[0]


def test_refuses_empty_or_short_input(tmp_path):
    csv_path = tmp_path / "sample.csv"
    make_csv(csv_path)
    with pytest.raises(ValueError):
        list(iter_transactions(csv_path, 0))
    with pytest.raises(ValueError, match="only 3 rows"):
        list(iter_transactions(csv_path, 4))


def test_stream_tracks_real_responses(tmp_path):
    csv_path = tmp_path / "sample.csv"
    make_csv(csv_path)
    submitted = []

    def fake_api(event):
        submitted.append(event)
        return {
            "event_id": event["event_id"],
            "decision": "REVIEW" if len(submitted) == 2 else "ALLOW",
            "duplicate": False,
            "n3": 1 if len(submitted) == 2 else 0,
            "risk": 0.2,
        }

    report = run_replay(iter_transactions(csv_path, 3), fake_api, progress_every=10)
    assert report["records"] == 3
    assert report["decisions"] == {"ALLOW": 2, "REVIEW": 1}
    assert report["transactions_with_n3_positive"] == 1
    assert len(submitted) == 3
    assert report["http_latency_ms"]["p95"] >= 0


def test_replay_rejects_duplicate(tmp_path):
    csv_path = tmp_path / "sample.csv"
    make_csv(csv_path)
    with pytest.raises(RuntimeError, match="duplicate"):
        run_replay(iter_transactions(csv_path, 1), lambda row: {
            "event_id": row["event_id"],
            "duplicate": True,
            "decision": "ALLOW",
            "n3": 0,
        })


def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], 0.5) == 25
    assert percentile([10, 20, 30, 40], 0.95) == pytest.approx(38.5)
