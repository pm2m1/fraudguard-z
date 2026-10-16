import math
import numpy as np
import pytest

from src.banksim_evaluate import (
    make_features, streaming_reviews, tuning_cutoff, delayed_accounting,
    candidate_models, describe,
)


def event(step, idx, customer, merchant, category="food", amount=10, label=0):
    return (step, idx, customer, merchant, category, amount, label)


def test_history_past_only_and_expiration():
    rows = [
        event(0, 0, "A", "M", amount=10),
        event(0, 1, "A", "M", amount=20),
        event(7, 2, "A", "M", amount=5),
    ]
    x, steps, labels, vocab = make_features(rows, tune_start=1)
    history_offset = 1 + len(vocab)
    assert np.all(x[0, history_offset:] == 0)
    assert x[1, history_offset] == pytest.approx(math.log1p(1))
    assert x[1, history_offset + 1] == pytest.approx(math.log1p(10))
    assert np.all(x[2, history_offset:] == 0)


def test_labels_never_affect_features():
    rows = [event(0, 0, "A", "M"), event(1, 1, "A", "M")]
    baseline = make_features(rows)[0]
    altered = [(*r[:-1], 1 - r[-1]) for r in rows]
    changed = make_features(altered)[0]
    np.testing.assert_array_equal(baseline, changed)


def test_unseen_category_not_added_to_training_vocabulary():
    rows = [
        event(0, 0, "A", "M", category="food"),
        event(108, 1, "A", "M", category="new_category"),
    ]
    x, _, _, vocab = make_features(rows)
    assert vocab == ["food"]
    assert x[1, 1] == 0


def test_review_cap_each_prefix_without_future_lookahead():
    scores = np.ones(230)
    decision = streaming_reviews(scores, 0.5, budget=.1, block=100)
    for i in range(230):
        start = (i // 100) * 100
        assert decision[start:i+1].sum() <= math.floor(
            ((i - start) + 1) * .1 + 1e-12
        )
    assert decision.sum() == 23
    longer = streaming_reviews(np.r_[scores, np.ones(50)], .5, budget=.1, block=100)
    np.testing.assert_array_equal(decision, longer[:230])


def test_tuning_is_unlabeled_and_ties_are_bounded():
    assert tuning_cutoff(np.arange(100.0), .1) == 90
    with pytest.raises(ValueError):
        tuning_cutoff(np.array([float("nan"), 1]))


def test_delay_aggregation_not_used_to_change_decisions():
    steps = np.array([0, 1, 7, 8], dtype=int)
    labels = np.array([1, 1, 0, 1], dtype=int)
    decisions = np.array([1, 0, 1, 1], dtype=bool)
    outcome = delayed_accounting(steps, labels, decisions)
    assert outcome["true_positives_after_reporting_delay"] == 2
    assert outcome["all_labels_released"] is True
    assert outcome["labels_released_after_final_watermark"] > 0


def test_models_predefined():
    specs = candidate_models(3)
    assert sorted(specs) == sorted(
        ["lr_static", "hgb_static", "hgb_history", "hgb_amount_only"]
    )
    assert [len(specs[key][1]) for key in
            ["lr_static", "hgb_static", "hgb_history", "hgb_amount_only"]] == [
                4, 4, 9, 1
            ]


def test_metrics_not_tuning_model():
    y = np.array([0, 1, 0, 1])
    scores = np.array([.1, .9, .2, .8])
    selected = np.array([0, 1, 0, 0], dtype=bool)
    steps = np.array([1, 1, 2, 2])
    summary = describe(y, scores, selected, steps)
    assert summary["true_positives"] == 1
    assert summary["false_positives"] == 0
