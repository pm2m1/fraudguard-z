import numpy as np
import pytest

from src.paysim_type_aware_review import (
    threshold_tie_audit, paired_block_stability
)


def test_cutoff_tie_audit():
    out = threshold_tie_audit([.2, .5, .5, .8], .5)
    assert out == {"above": 1, "equal": 2, "at_or_above": 3}


def test_cutoff_tie_audit_infinite_bounds():
    assert threshold_tie_audit([1, 2], float("inf"))["at_or_above"] == 0
    assert threshold_tie_audit([1, 2], float("-inf"))["above"] == 2


def test_paired_bootstrap_is_repeatable():
    labels = np.array([1, 0, 1, 0, 1, 0])
    base = np.array([1, 0, 0, 0, 0, 0], dtype=bool)
    other = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
    x = paired_block_stability(labels, base, other, block_size=2, draws=200)
    y = paired_block_stability(labels, base, other, block_size=2, draws=200)
    assert x == y
    assert x["observed_extra_fraud_detections"] == 2


def test_paired_bootstrap_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        paired_block_stability([1], [True, False], [True])
