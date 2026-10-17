"""Non-destructive unit checks for retrospective CASH_OUT diagnosis."""
import numpy as np
import pytest

from src.full_context_cashout_diagnostic import (
    cashout_observation, distribution, type_diagnostic,
)


def test_zero_eligible_explains_absence_of_reviews():
    info = type_diagnostic(
        np.array([1, 0, 1], dtype=np.uint8),
        np.array([0.1, 0.2, 0.4]),
        np.array([False, False, False]),
        0.5,
    )
    assert info["frauds"] == 2
    assert info["eligible"] == 0
    assert info["eligible_frauds"] == 0
    assert "ZERO_CASHOUT_SCORE" in cashout_observation(info)


def test_eligible_but_no_review_is_separate_failure_mode():
    info = type_diagnostic(
        np.array([0, 1], dtype=np.uint8),
        np.array([0.1, 0.9]),
        np.array([False, False]),
        0.5,
    )
    assert info["eligible"] == 1
    assert info["eligible_frauds"] == 1
    assert info["eligible_not_reviewed"] == 1
    assert "PREFIX" in cashout_observation(info).upper()


def test_summary_counts_ties_and_no_row_level_data():
    info = type_diagnostic(
        np.array([0, 1, 0, 1], dtype=np.uint8),
        np.array([0.2, 0.5, 0.5, 0.9]),
        np.array([False, False, True, True]),
        0.5,
    )
    assert info["exact_cutoff_ties"] == 2
    assert info["eligible"] == 3
    assert info["reviews"] == 2
    assert info["true_positives"] == 1
    assert info["eligible_not_reviewed"] == 1
    assert info["reviewed_below_threshold"] == 0
    assert info["score_distribution_fraud"]["max"] == pytest.approx(0.9)
    assert "event_ids" not in info


def test_invalid_nonfinite_score_rejected():
    with pytest.raises(ValueError):
        type_diagnostic([0, 1], [0.1, float("nan")], [False, False], .5)
    with pytest.raises(ValueError):
        distribution([float("inf")])


def test_empty_type_is_safe():
    r = type_diagnostic([], [], [], .5)
    assert r["rows"] == 0
    assert r["eligible"] == 0
    assert r["score_distribution_all"] is None
