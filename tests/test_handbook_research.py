import numpy as np
import pytest

from src.handbook_research import (
    BASIC_NAMES, BEHAVIOR_NAMES, PastOnlyActivity,
    ranking_metrics, temporal_masks,
)


def event(history, t, customer="C1", terminal="T1", amount=100):
    return history.extract(timestamp=t, customer=customer,
                           terminal=terminal, amount=amount)


def observe(history, t, customer="C1", terminal="T1", amount=100):
    history.observe(timestamp=t, customer=customer,
                    terminal=terminal, amount=amount)


def test_history_uses_only_previous_transactions():
    h = PastOnlyActivity()
    basic, enriched = event(h, 100)
    assert len(basic) == len(BASIC_NAMES) == 4
    assert len(enriched) == len(BEHAVIOR_NAMES) == 12
    assert enriched[4] == enriched[7] == enriched[10] == 0
    observe(h, 100)
    second = event(h, 100)[1]
    assert second[4] == pytest.approx(np.log1p(1))
    assert second[7] == pytest.approx(np.log1p(1))
    assert second[8] == pytest.approx(np.log1p(1))
    assert second[9] == pytest.approx(np.log1p(1))
    assert second[10] == pytest.approx(np.log1p(1))


def test_same_timestamp_is_ordered_by_observation():
    h = PastOnlyActivity()
    a = event(h, 10)[1]
    b = event(h, 10)[1]
    assert np.allclose(a, b)
    observe(h, 10)
    assert event(h, 10)[1][4] > 0


def test_expiry_and_nonmonotonic_time():
    h = PastOnlyActivity(window_seconds=10)
    observe(h, 20)
    assert event(h, 29)[1][4] > 0
    assert event(h, 30)[1][4] == 0
    with pytest.raises(ValueError):
        event(h, 29)


def test_graph_unique_counts_and_window():
    h = PastOnlyActivity(window_seconds=100)
    observe(h, 10, "C1", "T1")
    observe(h, 11, "C2", "T1")
    observe(h, 12, "C1", "T2")
    x = event(h, 13, "C1", "T1")[1]
    assert x[8] == pytest.approx(np.log1p(2))
    assert x[9] == pytest.approx(np.log1p(2))
    assert x[10] == pytest.approx(np.log1p(1))
    y = event(h, 112, "C1", "T1")[1]
    assert y[8] == 0
    assert y[9] == 0
    assert y[10] == 0


def test_validation_guards():
    h = PastOnlyActivity()
    with pytest.raises(ValueError):
        event(h, 1, amount=-1)
    with pytest.raises(ValueError):
        observe(h, 1, amount=float("nan"))
    with pytest.raises(ValueError):
        PastOnlyActivity(window_seconds=0)


def test_temporal_splits_are_disjoint():
    days = np.repeat(np.arange(10), 5)
    masks, bounds = temporal_masks(days, 10)
    assert bounds == (6, 8)
    assert [int(mask.sum()) for mask in masks] == [30, 10, 10]
    assert np.all(np.sum(np.stack(masks), axis=0) == 1)


def test_top_budget_metrics_deterministic_ties():
    y = np.array([1, 0, 1, 0])
    scores = np.array([.5, .5, .5, .1])
    metrics = ranking_metrics(y, scores, .5)
    assert metrics["reviews"] == 2
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1


def test_rejects_invalid_scores():
    with pytest.raises(ValueError):
        ranking_metrics(np.array([0, 1]), np.array([float("nan"), 1]))
    with pytest.raises(ValueError):
        ranking_metrics(np.array([0, 2]), np.array([.1, .2]))
