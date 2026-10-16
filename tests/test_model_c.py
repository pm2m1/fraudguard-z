import math
import csv

import numpy as np
import pytest

from src.model_c import MODEL_C_FEATURE_NAMES, ModelCHistory
from src.train_model_c import review_metrics, explain_ranking, load_train_validation
from src.features import RecipientHistory, make_feature_row, MODEL_B_FEATURE_NAMES


def event(step, sender="A", recipient="Z", amount=100, balance=100, kind="TRANSFER"):
    return {"step": step, "nameOrig": sender, "nameDest": recipient,
            "amount": amount, "oldbalanceOrg": balance, "type": kind}


def test_schema_and_current_event_are_not_counted():
    history = ModelCHistory()
    row = event(1)
    first = history.features(row)
    assert len(first) == len(MODEL_C_FEATURE_NAMES) == 17
    assert first[7] == first[11] == first[14] == 0
    assert first[9] == 1.0  # Sender balance nearly drained.
    assert history.features(row) == first
    history.observe(row)
    second = history.features(event(1, "B", "Z", 50, 100))
    assert second[7] == math.log1p(1)
    assert second[14] == math.log1p(1)


def test_sender_and_recipient_history_expire():
    history = ModelCHistory()
    history.observe(event(1, "A", "Z", 100))
    history.observe(event(2, "A", "Z", 20))
    at_6 = history.features(event(6, "A", "Z"))
    assert at_6[7] == math.log1p(2)
    assert at_6[11] == math.log1p(2)
    assert at_6[13] == math.log1p(120)
    assert at_6[14] == math.log1p(1)
    at_7 = history.features(event(7, "A", "Z"))
    assert at_7[7] == math.log1p(1)
    assert at_7[13] == math.log1p(20)
    at_8 = history.features(event(8, "A", "Z"))
    assert at_8[7] == 0
    assert at_8[11] == 0
    assert at_8[15] == math.log1p(7)


def test_unique_senders_count_with_repeated_pair():
    history = ModelCHistory()
    history.observe(event(1, "A", "Z"))
    history.observe(event(1, "A", "Z"))
    history.observe(event(1, "B", "Z"))
    assert history.features(event(2, "C", "Z"))[14] == math.log1p(2)
    history.features(event(7))
    assert not history.recipient_senders


def test_never_reads_post_balance_or_label():
    original = event(2)
    edited = {**original, "isFraud": 1, "newbalanceOrig": 0, "newbalanceDest": 123}
    assert ModelCHistory().features(original) == ModelCHistory().features(edited)


def test_bad_order_and_input():
    history = ModelCHistory()
    history.features(event(4))
    with pytest.raises(ValueError):
        history.features(event(3))
    with pytest.raises(ValueError):
        ModelCHistory().features(event(1, amount=-1))


def test_review_metrics_tie_break_and_types():
    labels = np.array([1, 0, 1, 0], dtype=np.uint8)
    scores = np.array([0.2, 0.2, 0.1, 0.0])
    selection, summary = review_metrics(labels, scores, fraction=0.5)
    assert selection.tolist() == [True, True, False, False]
    assert summary["true_positives"] == 1
    assert summary["false_positives"] == 1
    types = np.array([4, 4, 1, 1], dtype=np.uint8)
    report = explain_ranking(labels, types, scores)
    assert report["overall"]["reviews"] == 1


def test_small_csv_uses_only_training_and_validation(tmp_path):
    path = tmp_path / "toy.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "type", "amount",
            "nameOrig", "nameDest", "oldbalanceOrg", "isFraud"])
        writer.writeheader()
        for step in range(1, 5):
            writer.writerow({**event(step), "isFraud": str(int(step % 2 == 0))})
    manifest = {"split_end_steps": [2, 3, 4],
                "partitions": {"train": {"rows": 2, "frauds": 1},
                               "validation": {"rows": 1, "frauds": 0}}}
    X, labels, types, cutoff = load_train_validation(manifest, path)
    assert X.shape == (3, 17)
    assert labels.tolist() == [0, 1, 0]
    assert cutoff == 2


def test_first_eight_columns_reproduce_model_b():
    assert MODEL_C_FEATURE_NAMES[:8] == MODEL_B_FEATURE_NAMES
    base_history = RecipientHistory(window_steps=6)
    candidate_history = ModelCHistory(window_steps=6)
    sequence = [
        event(1, "A", "Z", 100, 250),
        event(1, "B", "Z", 10, 100, "CASH_OUT"),
        event(2, "A", "Y", 50, 150),
        event(7, "C", "Z", 70, 100),
        event(8, "A", "Z", 5, 100),
    ]
    for row in sequence:
        expected = np.asarray(make_feature_row(row, base_history), dtype=np.float32)
        actual = np.asarray(candidate_history.features(row)[:8], dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)
        base_history.observe(row["nameDest"], row["step"])
        candidate_history.observe(row)
