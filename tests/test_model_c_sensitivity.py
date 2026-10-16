import numpy as np
import pytest

from src.model_c_sensitivity import (
    alter_current_financial_features, select_top_k, summarize_sensitivity,
)
from src.model_c import MODEL_C_FEATURE_NAMES

def make_row(amount=100, balance=100):
    row = np.zeros((1, len(MODEL_C_FEATURE_NAMES)), dtype=np.float32)
    row[0, 0] = np.log1p(amount)
    row[0, 6] = np.log1p(balance)
    row[0, 8] = np.log1p(amount / (balance + 1))
    row[0, 9] = float(balance > 0 and .95 * balance <= amount <= 1.01 * balance)
    row[0, 10] = float(amount > balance)
    row[0, 11] = 1.5
    return row

def test_matching_balance_breaks_after_amount_reduction():
    before = make_row()
    after = alter_current_financial_features(before, amount_factor=0.9)
    assert before[0, 9] == 1
    assert after[0, 9] == 0
    assert after[0, 11] == before[0, 11]

def test_joint_scaling_preserves_relationship():
    before = make_row()
    after = alter_current_financial_features(before, amount_factor=0.9, balance_factor=0.9)
    assert after[0, 9] == 1
    assert after[0, 10] == 0

def test_no_perturbation_reconstructs_original():
    before = make_row(200, 400)
    after = alter_current_financial_features(before)
    np.testing.assert_allclose(before, after, atol=1e-5)

def test_bad_parameters_and_shape_rejected():
    with pytest.raises(ValueError):
        alter_current_financial_features(make_row(), balance_factor=0)
    with pytest.raises(ValueError):
        alter_current_financial_features(np.zeros((2, 8)))

def test_deterministic_top_k_ties():
    assert select_top_k([0.5, 0.5, 0.3], 1).tolist() == [True, False, False]

def test_overlap_and_rule_retention():
    x = np.array([0.8, 0.4, 0.2])
    changed = np.array([0.1, 0.9, 0.2])
    old = np.array([True, True, False])
    new = np.array([False, True, True])
    rule = np.array([True, False, False])
    result = summarize_sensitivity(x, changed, old, new, rule)
    assert result["top_1pct_overlap_count"] == 1
    assert result["fraction_original_rule_reviews_retained"] == 0
