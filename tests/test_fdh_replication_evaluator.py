import numpy as np
import pytest

from src.fdh_seeded_generator import Event
from src.fdh_replication_evaluator import (
    STATIC_NAMES, HISTORY_NAMES, TRAIN_END, TUNE_END,
    cutoff_from_unlabeled_scores, delayed_label_report, evaluate_events,
    online_selection, past_only_arrays,
)


def ev(index, day, customer=1, terminal=2, amount=10.0, scenario=0):
    return Event(index, day * 86400 + index % 100, day, customer, terminal,
                 amount, int(scenario > 0), scenario)


def test_features_use_prior_rows_only():
    records = [ev(0, 0), ev(1, 0, amount=20), ev(2, 7)]
    X, days, labels, scenarios = past_only_arrays(records)
    offset = len(STATIC_NAMES)
    assert np.all(X[0, offset:] == 0)
    assert X[1, offset] == pytest.approx(np.log1p(1))
    assert X[1, offset+1] == pytest.approx(np.log1p(10))
    assert np.all(X[2, offset:] == 0)


def test_change_labels_not_features():
    records = [ev(0, 0), ev(1, 0, amount=30, scenario=3)]
    changed = [records[0]._replace(fraud=1, fraud_scenario=2), records[1]]
    assert np.array_equal(past_only_arrays(records)[0],
                          past_only_arrays(changed)[0])


def test_hard_streaming_prefix_caps_and_causality():
    scores = np.ones(125)
    choice = online_selection(scores, 0.5, budget=.1, block=50)
    assert choice.sum() == 12
    for i in range(len(scores)):
        start = (i // 50) * 50
        assert choice[start:i+1].sum() <= ((i-start+1) // 10)
    longer = online_selection(np.ones(175), .5, budget=.1, block=50)
    assert np.array_equal(choice, longer[:125])


def test_cutoff_with_ties_and_errors():
    assert cutoff_from_unlabeled_scores([1, 2, 3, 4], .25) == 4
    with pytest.raises(ValueError):
        cutoff_from_unlabeled_scores([np.inf, 2], .5)


def test_delayed_accounting():
    d = delayed_label_report(
        np.array([72, 73, 80, 89]),
        np.array([1, 0, 1, 1]),
        np.array([1, 1, 0, 1], dtype=bool),
    )
    assert d["total_eventual_labels"] == 4
    assert d["detected_after_reporting_delay"] == 2
    assert d["all_labels_released"] is True


def test_known_feature_dimensions():
    assert len(STATIC_NAMES) == 5
    assert len(HISTORY_NAMES) == 5


def test_eval_rejects_incomplete_stream():
    with pytest.raises(ValueError, match="Incomplete"):
        evaluate_events([ev(0, 0), ev(1, 1)])


def test_reject_out_of_order_events():
    with pytest.raises(ValueError, match="Unsorted"):
        past_only_arrays([ev(1, 1), ev(0, 0)])
