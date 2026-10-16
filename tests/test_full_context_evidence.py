import json

import pytest

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
