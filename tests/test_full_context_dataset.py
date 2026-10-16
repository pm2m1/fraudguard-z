import csv
import math

import numpy as np
import pytest

from src.full_context_dataset import (
    build_feature_batch, iter_feature_records, iter_labelled_development_csv,
)
from src.full_context_journal import FullContextJournal, FULL_FEATURE_NAMES

KEY = "full-context-parity-key-of-adequate-length"


def event(i, step, sender="A", dest="B", kind="TRANSFER", amount=10):
    return {
        "event_id": f"E{i}", "source_row_index": i,
        "step": step, "nameOrig": sender, "nameDest": dest,
        "type": kind, "amount": amount, "oldbalanceOrg": 100,
    }


def sequence():
    return [
        event(0, 1),
        event(1, 1, sender="C"),
        event(2, 2, sender="B", dest="D", kind="CASH_OUT"),
        event(3, 6, sender="A", amount=30),
        event(4, 7, sender="A", dest="D"),
        event(5, 25, sender="C", dest="B"),
    ]


def test_offline_serving_feature_parity_and_restart(tmp_path):
    events = sequence()
    X, y, cursors = build_feature_batch((e, i % 2) for i, e in enumerate(events))
    assert X.shape == (len(events), 21)
    assert y.tolist() == [i % 2 for i in range(len(events))]
    db = tmp_path / "new_full_context.sqlite3"

    def open_journal():
        return FullContextJournal(
            db, lambda _: .1, lambda _: "ALLOW",
            model_version="test-model", policy_version="test-policy",
            privacy_key=KEY,
        )

    journal = open_journal()
    try:
        for i, row in enumerate(events[:3]):
            result = journal.process(row)
            assert result["duplicate"] is False
            actual = np.asarray(list(result["features"].values()), dtype=np.float32)
            np.testing.assert_array_equal(actual, X[i])
    finally:
        journal.close()
    journal = open_journal()
    try:
        for i, row in enumerate(events[3:], start=3):
            result = journal.process(row)
            actual = np.asarray(list(result["features"].values()), dtype=np.float32)
            np.testing.assert_array_equal(actual, X[i])
        assert journal.status()["committed_events"] == 6
    finally:
        journal.close()
    assert cursors == [(e["step"], e["source_row_index"]) for e in events]


def test_no_target_or_future_fields_affect_features():
    rows = sequence()
    regular, *_ = build_feature_batch((e, 0) for e in rows)
    tampered = [
        {**e, "isFraud": 1, "isFlaggedFraud": 1,
         "newbalanceOrig": -99999, "newbalanceDest": -99999}
        for e in rows
    ]
    altered, *_ = build_feature_batch((e, 1) for e in tampered)
    np.testing.assert_array_equal(regular, altered)


def test_invalid_order_and_label():
    rows = sequence()
    with pytest.raises(ValueError, match="strictly ordered"):
        list(iter_feature_records([(rows[1], 0), (rows[0], 1)]))
    with pytest.raises(ValueError, match="Fraud label"):
        list(iter_feature_records([(rows[0], 2)]))


def test_dev_csv_stop_before_holdout_without_reading_its_label(tmp_path):
    path = tmp_path / "paysim.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "step", "type", "amount", "nameOrig", "nameDest",
            "oldbalanceOrg", "isFraud",
        ])
        writer.writeheader()
        writer.writerow({**{k: event(0, 372)[k] for k in writer.fieldnames if k != "isFraud"}, "isFraud": 1})
        writer.writerow({**{k: event(1, 373)[k] for k in writer.fieldnames if k != "isFraud"}, "isFraud": "SECRET_TEST_LABEL"})
    yielded = list(iter_labelled_development_csv(path))
    assert len(yielded) == 1
    assert yielded[0][1] == "1"


def test_strict_dev_boundary_and_empty_batch():
    assert build_feature_batch(iter([]))[0].shape == (0, len(FULL_FEATURE_NAMES))
    with pytest.raises(ValueError, match="Unsafe development boundary"):
        build_feature_batch([], max_step=373)
