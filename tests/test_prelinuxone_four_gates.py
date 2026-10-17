import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.full_context_margin_attribution import explain_margin
from src.full_context_calibration import INDICES
from src.full_context_shadow_policy import ShadowThresholds, simulate_decisions


def test_shadow_three_actions_and_prefix_budget():
    cfg = ShadowThresholds(review=.3, block=.8, max_review_fraction=.5)
    scores = [.9,.9,.5,.1,.95,.35]
    actions=simulate_decisions(scores,cfg)
    assert actions == [
        "ALLOW_SIMULATION_ONLY", "WOULD_BLOCK_SIMULATION_ONLY",
        "ALLOW_SIMULATION_ONLY", "ALLOW_SIMULATION_ONLY",
        "WOULD_BLOCK_SIMULATION_ONLY", "WOULD_REVIEW_SIMULATION_ONLY"]
    assert all("SIMULATION_ONLY" in a for a in actions)


def test_shadow_policy_rejects_unsafe_configuration():
    for r,b in ((.5,.4),(.5,.5),(0,.5),(.5,1.5)):
        with pytest.raises(ValueError):
            ShadowThresholds(review=r,block=b)
    cfg=ShadowThresholds(.2,.8)
    for invalid in (float("nan"),float("inf"),-1,1.2):
        with pytest.raises(ValueError):
            simulate_decisions([invalid],cfg)


def test_exact_margin_attribution_reconstructs_fitted_pipeline():
    rng=np.random.default_rng(2026)
    X=rng.normal(size=(60,len(INDICES))).astype(np.float32)
    y=np.array([0,1]*30)
    pipeline=make_pipeline(StandardScaler(),LogisticRegression()).fit(X,y)
    full=np.zeros(21,dtype=np.float32)
    full[list(INDICES)]=X[7]
    outcome=explain_margin(pipeline,full)
    assert outcome["not_probability_or_causal_attribution"] is True
    assert len(outcome["items"]) == len(INDICES)
    contribution_sum = sum(z["contribution"] for z in outcome["items"]) + outcome["intercept"]
    # First verify the contribution decomposition to float64 arithmetic precision.
    assert contribution_sum == pytest.approx(outcome["reconstructed_margin"], abs=1e-12)
    # The sklearn decision margin can differ by float32/BLAS rounding.
    assert outcome["margin"] == pytest.approx(
        float(pipeline.decision_function(X[7:8])[0]), abs=1e-12
    )
    assert outcome["reconstruction_abs_error"] == pytest.approx(
        abs(contribution_sum - outcome["margin"]), abs=1e-12
    )
    assert outcome["reconstruction_abs_error"] <= outcome["reconstruction_tolerance"]


def test_attribution_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        explain_margin(None,np.zeros(21))
    with pytest.raises(ValueError):
        explain_margin(object(),np.zeros(21))
    with pytest.raises(ValueError):
        explain_margin(object(),np.zeros(20))


def test_meaningful_attribution_mismatch_still_fails(monkeypatch):
    rng = np.random.default_rng(2027)
    X = rng.normal(size=(30, len(INDICES))).astype(np.float32)
    y = np.array([0, 1] * 15)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    full = np.zeros(21, dtype=np.float32)
    full[list(INDICES)] = X[0]
    original = pipeline.decision_function
    monkeypatch.setattr(
        pipeline, "decision_function",
        lambda values: original(values) + 0.01,
    )
    with pytest.raises(RuntimeError, match="does not reconstruct"):
        explain_margin(pipeline, full)
