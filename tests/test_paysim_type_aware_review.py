import math

import numpy as np
import pytest

from src.features import TYPE_TO_INDEX
from src.paysim_type_aware_review import (
    cutoff_at_quota,
    describe,
    policy_thresholds,
    select_online,
)

T = TYPE_TO_INDEX["TRANSFER"]
C = TYPE_TO_INDEX["CASH_OUT"]
P = TYPE_TO_INDEX["PAYMENT"]


def test_score_cutoff_is_label_free_and_deterministic():
    assert cutoff_at_quota(np.arange(100.0), 10) == 90.0
    assert cutoff_at_quota(np.arange(100.0), 0) == math.inf
    assert cutoff_at_quota(np.arange(100.0), 100) == -math.inf


def test_cutoff_rejects_nan_and_negative_quota():
    with pytest.raises(ValueError):
        cutoff_at_quota(np.array([0.1, np.nan]), 1)
    with pytest.raises(ValueError):
        cutoff_at_quota(np.array([0.1]), -1)


def test_tuned_allocations_include_only_target_groups():
    scores = np.arange(200, dtype=float)
    kinds = np.array([T] * 100 + [C] * 100)
    th = policy_thresholds(scores, kinds, transfer_share=.25, budget=.1)
    assert th["TRANSFER"] == 95.0
    assert th["CASH_OUT"] == 185.0
    global_scores = policy_thresholds(scores, kinds, budget=.1)
    assert global_scores == {"GLOBAL": 180.0}


def test_global_eligible_cutoff_ignores_irrelevant_type():
    scores = np.arange(200, dtype=float)
    kinds = np.array([T] * 100 + [P] * 100)
    cutoffs = policy_thresholds(
        scores, kinds, restrict_eligible=True, budget=.1
    )
    assert cutoffs["GLOBAL"] == 80.0


def test_online_global_cap_never_exceeds_any_block_prefix():
    n = 550
    chosen = select_online(
        np.ones(n), np.full(n, P),
        {"GLOBAL": 0}, budget=.1, block_size=200,
    )
    for i in range(n):
        beginning = (i // 200) * 200
        reviews = int(chosen[beginning:i + 1].sum())
        assert reviews <= math.floor((i - beginning + 1) * .1 + 1e-10)
    assert int(chosen.sum()) == 55


def test_online_type_quotas_do_not_backfill_unused_capacity():
    chosen = select_online(
        np.ones(300), np.full(300, C),
        {"TRANSFER": 0, "CASH_OUT": 0},
        transfer_share=.5, budget=.01, block_size=300,
    )
    assert int(chosen.sum()) == 1  # cap=3, round(3 * .5)=2, C cap=1


def test_online_selection_is_causal_under_future_extension():
    rng = np.random.default_rng(23)
    scores = rng.random(440)
    kinds = rng.choice([T, C, P], size=440)
    cutoffs = {"TRANSFER": .2, "CASH_OUT": .3}
    first = select_online(
        scores[:275], kinds[:275], cutoffs, transfer_share=.5,
        budget=.04, block_size=150,
    )
    extended = select_online(
        scores, kinds, cutoffs, transfer_share=.5,
        budget=.04, block_size=150,
    )
    np.testing.assert_array_equal(first, extended[:275])


def test_describe_counts_type_specific_true_and_false_positives():
    labels = np.array([1, 0, 1, 0, 0, 1], dtype=int)
    types = np.array([T, T, C, C, P, C])
    scores = np.array([.9, .1, .8, .2, .05, .7])
    chosen = np.array([True, True, True, False, False, False])
    d = describe(labels, types, scores, chosen)
    assert d["true_positives"] == 2
    assert d["false_positives"] == 1
    assert d["by_type"]["TRANSFER"]["false_positives"] == 1
    assert d["by_type"]["CASH_OUT"]["true_positives"] == 1


def test_invalid_stream_inputs_rejected():
    with pytest.raises(ValueError):
        select_online(np.array([math.nan]), np.array([T]), {"GLOBAL": .5})
