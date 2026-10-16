import numpy as np
import pytest

from src.review_audit import (
    ranked_indices,
    summarize_selection,
    top_k_selection,
)


def test_review_count_and_confusion():
    labels = np.array([1, 0, 1, 0, 0], dtype=np.uint8)
    chosen = np.array([True, True, False, False, False])
    report = summarize_selection(labels, chosen)
    assert report["reviews"] == 2
    assert report["true_positives"] == 1
    assert report["false_positives"] == 1
    assert report["false_negatives"] == 1
    assert report["precision"] == 0.5
    assert report["recall"] == 0.5


def test_tied_scores_prefer_earlier_events():
    scores = np.array([0.8, 0.9, 0.9, 0.1])
    ranked = ranked_indices(scores)
    assert list(ranked) == [1, 2, 0, 3]
    selected = top_k_selection(ranked, 4, 0.5)
    assert list(selected) == [False, True, True, False]


def test_budget_uses_ceil():
    selected = top_k_selection(ranked_indices(np.arange(5)), 5, 0.21)
    assert selected.sum() == 2


def test_empty_group_has_no_division_by_zero():
    report = summarize_selection(np.array([], dtype=np.uint8), np.array([], dtype=bool))
    assert report["reviews"] == 0
    assert report["recall"] is None


def test_rejects_nonfinite_scores_and_invalid_budgets():
    with pytest.raises(ValueError):
        ranked_indices([0.1, float("nan")])
    with pytest.raises(ValueError):
        top_k_selection(np.array([0]), 1, 0)
