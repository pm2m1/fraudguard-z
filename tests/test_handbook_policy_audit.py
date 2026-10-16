import numpy as np
import pytest

from src.handbook_policy_audit import (
    daily_fixed_counts, fixed_threshold_metrics,
    moving_block_confidence_intervals, validation_cutoff,
)


def test_validation_cutoff_uses_only_supplied_scores():
    assert validation_cutoff([.1, .2, .3, .4], .25) == pytest.approx(.4)
    assert validation_cutoff([.1, .2, .3, .4], .50) == pytest.approx(.3)


def test_validation_cutoff_ties_cannot_guarantee_exact_budget():
    cutoff = validation_cutoff([.1, .5, .5, .5], .25)
    assert cutoff == .5
    result = fixed_threshold_metrics([0, 1, 0, 0], [.1, .5, .5, .5], cutoff)
    assert result["reviews"] == 3
    assert result["review_rate"] == .75


def test_fixed_threshold_metrics_correct_workload():
    res = fixed_threshold_metrics([1, 0, 1, 0],
                                  [.9, .8, .1, .0], .5)
    assert res["reviews"] == 2
    assert res["true_positives"] == 1
    assert res["false_positives"] == 1
    assert res["false_negatives"] == 1
    assert res["precision"] == .5
    assert res["recall"] == .5


def test_daily_counts_and_alignment():
    days = np.array([4, 4, 5, 5])
    y = np.array([1, 0, 1, 0])
    s = np.array([.9, .3, .2, .8])
    unique, counts = daily_fixed_counts(days, y, s, .5)
    assert unique.tolist() == [4, 5]
    assert counts.tolist() == [[2, 1, 1, 1], [2, 1, 1, 0]]


def test_paired_block_bootstrap_fixed_improvement():
    baseline = np.array([[100, 20, 10, 3]] * 8)
    history = np.array([[100, 20, 10, 5]] * 8)
    a = moving_block_confidence_intervals(baseline, history,
                                          iterations=100, block_days=3)
    b = moving_block_confidence_intervals(baseline, history,
                                          iterations=100, block_days=3)
    assert a == b
    delta = a["history_minus_baseline_recall"]
    assert delta["lower_95pct"] == pytest.approx(.1)
    assert delta["upper_95pct"] == pytest.approx(.1)
    assert a["history_minus_baseline_review_rate"]["median"] == 0


def test_audit_input_guards():
    with pytest.raises(ValueError):
        validation_cutoff([.1, float("nan")])
    with pytest.raises(ValueError):
        validation_cutoff([.1], 0)
    with pytest.raises(ValueError):
        fixed_threshold_metrics([0], [.2], float("nan"))
    with pytest.raises(ValueError):
        daily_fixed_counts([1], [0, 1], [.1], .5)
    with pytest.raises(ValueError):
        moving_block_confidence_intervals(
            np.zeros((2, 4)), np.zeros((3, 4)), iterations=100)
