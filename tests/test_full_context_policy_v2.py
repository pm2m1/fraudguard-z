import numpy as np
import pytest

from src.full_context_policy_v2 import inclusive_threshold
from src.full_context_policy import decisions_with_prefix_cap


def test_inclusive_boundary_retains_tied_candidates():
    scores = np.array([.1, .2, .2, .2, .9])
    # 40% capacity = 2, 2nd highest cutoff = .2, all .2 tie.
    assert inclusive_threshold(scores, 0.4) == pytest.approx(.2)
    assert np.count_nonzero(scores >= inclusive_threshold(scores, .4)) == 4


def test_v2_prefix_cap_with_all_scores_tied():
    scores = np.full(526517, .25)
    threshold = inclusive_threshold(scores)
    assert threshold == .25
    selected = decisions_with_prefix_cap(scores, threshold)
    assert selected.sum() == 5265
    assert not selected[:99].any()
    for n in (1, 99, 100, 199, 200, 1000, len(scores)):
        assert selected[:n].sum() <= int(np.floor(.01*n))


def test_invalid_threshold_input():
    with pytest.raises(ValueError):
        inclusive_threshold([0, float("nan")])
    with pytest.raises(ValueError):
        inclusive_threshold([], 0.01)
