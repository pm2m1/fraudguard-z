"""Deterministic, observed-only PaySim history/network evidence.

Evidence is a human-readable statement about committed-prior event state,
NOT model feature attribution, a causal explanation, fraud proof or reason
to block. Uses no labels, learned thresholds, future balances or raw IDs.

The original 21-column feature schema remains unchanged.
"""
import json
from math import expm1, isfinite

from .full_context import FULL_FEATURE_NAMES

EVIDENCE_SCHEMA_ID = "fgz_observed_context_v1"


def _as_vector(values):
    if isinstance(values, dict):
        if set(values) != set(FULL_FEATURE_NAMES):
            raise ValueError("Evidence needs the exact full feature schema")
        values = [values[name] for name in FULL_FEATURE_NAMES]
    if len(values) != len(FULL_FEATURE_NAMES):
        raise ValueError("Evidence needs exactly 21 columns")
    result = tuple(float(v) for v in values)
    if not all(isfinite(v) for v in result):
        raise ValueError("Nonfinite evidence feature")
    return result


def _nonnegative_int(raw, feature):
    # Lifetime cardinalities are stored as log1p counts; a rounded estimate
    # recovers a human-readable *integer* and tolerates float32 conversion.
    if raw < 0:
        raise ValueError(f"Negative {feature}")
    estimated = expm1(raw)
    value = round(estimated)
    if value < 0 or abs(estimated - value) > max(0.1, abs(value) * 2e-5):
        raise ValueError(f"Malformed historical counter {feature}")
    return int(value)


def _item(code, feature, observed, explanation, unit):
    return {
        "code": code,
        "feature": feature,
        "observed": observed,
        "unit": unit,
        "description": explanation,
        "evidence_kind": "observation_not_model_attribution",
    }


def build_evidence(features):
    """Return ordered, JSON-serializable context observations for one event.

    All counters are computed from *past committed events* before the current
    event is observed. No empirical fraud threshold is applied.
    """
    vector = _as_vector(features)
    kind_flags = vector[1:6]
    if any(abs(v - round(v)) > 1e-8 or round(v) not in (0, 1) for v in kind_flags):
        raise ValueError("Malformed type indicators")
    if sum(round(v) for v in kind_flags) != 1:
        raise ValueError("Transaction type is not one-hot")
    sender_count = _nonnegative_int(vector[9], "log_sender_prior_out_count")
    pair_count = _nonnegative_int(vector[13], "log_pair_prior_count")
    prior_originators = _nonnegative_int(vector[17], "log_recipient_distinct_source_count")
    sender6 = _nonnegative_int(vector[10], "sender_out_count_6h")
    incoming6 = _nonnegative_int(vector[18], "recipient_in_count_6h")
    cashout6 = _nonnegative_int(vector[19], "recipient_cashout_count_6h")
    prior_pair_24 = _nonnegative_int(vector[14], "pair_count_24h")
    flag = vector[15]
    if flag not in (0.0, 1.0) or bool(flag) != (pair_count == 0):
        raise ValueError("Inconsistent first-time recipient flag")
    share = vector[20]
    if share < 0 or share > 1:
        raise ValueError("Invalid pair share")
    items = [
        _item(
            "RECIPIENT_PRIOR_ORIGINATORS", "log_recipient_distinct_source_count",
            prior_originators,
            "Distinct originator accounts previously observed with this destination",
            "accounts",
        ),
        _item(
            "SENDER_PRIOR_ACTIVITY", "log_sender_prior_out_count", sender_count,
            "Previous transactions where this originator was the sender",
            "transactions",
        ),
        _item(
            "PAIR_PRIOR_ACTIVITY", "log_pair_prior_count", pair_count,
            "Previous originator-to-destination transactions",
            "transactions",
        ),
        _item(
            "PAIR_FIRST_OBSERVED" if flag else "PAIR_SEEN_BEFORE",
            "first_time_recipient", bool(flag),
            "Whether this originator-to-destination pair has no prior events",
            "boolean",
        ),
        _item(
            "SENDER_OUTGOING_6H", "sender_out_count_6h", sender6,
            "Originator transaction count in the preceding six PaySim step-hours",
            "transactions",
        ),
        _item(
            "RECIPIENT_INCOMING_6H", "recipient_in_count_6h", incoming6,
            "Previous transactions reaching this destination in six PaySim step-hours",
            "transactions",
        ),
        _item(
            "RECIPIENT_AS_CASHOUT_ORIGIN_6H", "recipient_cashout_count_6h",
            cashout6,
            "Previous CASH_OUT events originating from this destination account over six step-hours",
            "transactions",
        ),
        _item(
            "PAIR_RECENT_24H", "pair_count_24h", prior_pair_24,
            "Previous originator-to-destination transactions in 24 PaySim step-hours",
            "transactions",
        ),
        _item(
            "SENDER_DEST_SHARE_24H", "sender_current_dest_share_24h", float(share),
            "Share of the originator's previous 24-step-hour transactions to this destination",
            "fraction",
        ),
    ]
    return {
        "evidence_schema_id": EVIDENCE_SCHEMA_ID,
        "scope": "PaySim synthetic past-only observed context",
        "model_attribution": False,
        "fraud_causation_claim": False,
        "items": items,
    }


def canonical_evidence_json(features):
    return json.dumps(build_evidence(features), sort_keys=True, separators=(",", ":"))
