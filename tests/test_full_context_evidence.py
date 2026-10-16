import json

import pytest
import numpy as np

from src.full_context import FullContextHistory
from src.full_context_evidence import build_evidence, canonical_evidence_json


def event(step, orig="A", dest="B", kind="TRANSFER", amount=10):
    return {"step": step, "type": kind, "nameOrig": orig, "nameDest": dest,
            "amount": amount, "oldbalanceOrg": 100}


def test_empty_history_reports_observed_zero_not_fraud_claim():
    history = FullContextHistory()
    evidence = build_evidence(history.features(event(1)))
    lookup = {item["code"]: item for item in evidence["items"]}
    assert lookup["RECIPIENT_PRIOR_ORIGINATORS"]["observed"] == 0
    assert lookup["PAIR_FIRST_OBSERVED"]["observed"] is True
    assert evidence["model_attribution"] is False
    assert evidence["fraud_causation_claim"] is False
    assert all(item["evidence_kind"] == "observation_not_model_attribution"
               for item in evidence["items"])


def test_only_prior_committed_events_appear_in_evidence():
    history = FullContextHistory()
    first = event(1)
    history.features(first)
    history.observe(first)
    second = event(1, orig="C", dest="B")
    before = build_evidence(history.features(second))
    lookup = {x["code"]: x["observed"] for x in before["items"]}
    assert lookup["RECIPIENT_PRIOR_ORIGINATORS"] == 1
    assert lookup["RECIPIENT_INCOMING_6H"] == 1
    assert lookup["PAIR_FIRST_OBSERVED"] is True
    history.observe(second)
    third = event(2)
    after = build_evidence(history.features(third))
    lookup = {x["code"]: x["observed"] for x in after["items"]}
    assert lookup["RECIPIENT_PRIOR_ORIGINATORS"] == 2
    assert lookup["PAIR_PRIOR_ACTIVITY"] == 1
    assert lookup["PAIR_SEEN_BEFORE"] is False
    assert json.loads(canonical_evidence_json(history.features(third))) == after


def test_expired_recent_but_lifetime_counts_remain():
    history = FullContextHistory()
    old = event(1)
    history.features(old)
    history.observe(old)
    evidence = build_evidence(history.features(event(7)))
    lookup = {x["code"]: x["observed"] for x in evidence["items"]}
    assert lookup["SENDER_OUTGOING_6H"] == 0
    assert lookup["RECIPIENT_INCOMING_6H"] == 0
    assert lookup["PAIR_PRIOR_ACTIVITY"] == 1


def test_reject_malformed_features():
    history = FullContextHistory()
    good = list(history.features(event(1)))
    with pytest.raises(ValueError):
        build_evidence(good[:-1])
    wrong = good.copy()
    wrong[17] = float("nan")
    with pytest.raises(ValueError):
        build_evidence(wrong)
    wrong = good.copy()
    wrong[15] = 0.0
    with pytest.raises(ValueError):
        build_evidence(wrong)


def test_raw_window_counts_vs_log_lifetime_counts_and_float32():
    history = FullContextHistory()
    for prior in (
        event(1, orig="A", dest="B"),
        event(1, orig="C", dest="B"),
        event(1, orig="B", dest="X", kind="CASH_OUT"),
    ):
        history.features(prior)
        history.observe(prior)
    # Exercise the same float32 representation used by the data materializer.
    vector = np.asarray(history.features(event(2, orig="A", dest="B")),
                        dtype=np.float32)
    evidence = build_evidence(vector)
    observed = {item["code"]: item["observed"] for item in evidence["items"]}
    assert observed["RECIPIENT_PRIOR_ORIGINATORS"] == 2   # log1p-encoded
    assert observed["SENDER_PRIOR_ACTIVITY"] == 1        # log1p-encoded
    assert observed["PAIR_PRIOR_ACTIVITY"] == 1          # log1p-encoded
    assert observed["SENDER_OUTGOING_6H"] == 1           # raw integer
    assert observed["RECIPIENT_INCOMING_6H"] == 2        # raw integer
    assert observed["RECIPIENT_AS_CASHOUT_ORIGIN_6H"] == 1
    assert observed["PAIR_RECENT_24H"] == 1              # raw integer


def test_reject_fractional_raw_counter():
    history = FullContextHistory()
    vector = list(history.features(event(1)))
    vector[18] = 1.718281828  # Illegal raw count; must not be expm1 decoded.
    with pytest.raises(ValueError, match="Malformed historical counter"):
        build_evidence(vector)
