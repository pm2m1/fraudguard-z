import numpy as np


def test_unsigned_uint8_detection_counts_require_signed_python_difference():
    # Regression for failed seed_104729__default paired comparison.
    y = np.array([1, 1, 0, 0], dtype=np.uint8)
    history = np.array([False, False, False, False])
    static = np.array([True, True, False, False])
    difference = int(y[history].sum()) - int(y[static].sum())
    assert difference == -2


def test_positive_and_zero_differences():
    y = np.array([1, 1, 0], dtype=np.uint8)
    h = np.array([True, True, False])
    s = np.array([True, False, False])
    assert int(y[h].sum()) - int(y[s].sum()) == 1
    assert int(y[s].sum()) - int(y[s].sum()) == 0
