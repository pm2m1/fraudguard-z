import math

import pytest

from src.full_context import FULL_FEATURE_NAMES, FullContextHistory


def event(step, sender="A", dest="B", kind="TRANSFER", amount=10, balance=100):
    return {"step": step, "nameOrig": sender, "nameDest": dest,
            "type": kind, "amount": amount, "oldbalanceOrg": balance}


def as_features(state, row):
    return dict(zip(FULL_FEATURE_NAMES, state.features(row)))


def test_frozen_width_and_input_validation():
    h = FullContextHistory()
    assert len(h.features(event(1))) == 21
    with pytest.raises(ValueError):
        h.features(event(2, amount=-1))
    with pytest.raises(ValueError):
        h.features(event(2, kind="INVALID"))


def test_past_only_first_and_same_step_second_event():
    h = FullContextHistory()
    row = event(1)
    before = as_features(h, row)
    assert before["sender_out_count_6h"] == 0
    assert before["first_time_recipient"] == 1
    assert before["recipient_in_count_6h"] == 0
    assert before["sender_current_dest_share_24h"] == 0
    assert as_features(h, row) == before  # no observation on features()
    h.observe(row)
    second = as_features(h, event(1, amount=20))
    assert second["sender_out_count_6h"] == 1
    assert second["recipient_in_count_6h"] == 1
    assert second["first_time_recipient"] == 0
    assert second["pair_count_24h"] == 1
    assert second["sender_current_dest_share_24h"] == 1
    assert second["sender_amount_vs_hist_mean"] == pytest.approx(math.log(21 / 11))


def test_boundary_expiration_and_lifetime_pair_history():
    h = FullContextHistory()
    first = event(1)
    h.observe(first)
    assert as_features(h, event(6))["sender_out_count_6h"] == 1
    assert as_features(h, event(7))["sender_out_count_6h"] == 0
    assert as_features(h, event(24))["pair_count_24h"] == 1
    final = as_features(h, event(25))
    assert final["pair_count_24h"] == 0
    assert final["first_time_recipient"] == 0
    assert final["log_pair_prior_count"] == pytest.approx(math.log1p(1))


def test_network_distinct_and_cashout_history():
    h = FullContextHistory()
    h.observe(event(1, sender="A", dest="B", kind="TRANSFER"))
    h.observe(event(2, sender="C", dest="B", kind="CASH_OUT"))
    h.observe(event(3, sender="B", dest="D", kind="CASH_OUT"))
    f = as_features(h, event(4, sender="A", dest="B"))
    assert f["log_recipient_distinct_source_count"] == pytest.approx(math.log1p(2))
    assert f["log_sender_distinct_dest_count"] == pytest.approx(math.log1p(1))
    assert f["recipient_cashout_count_6h"] == 1
    assert f["sender_current_dest_share_24h"] == 1


def test_cashout_expiration_and_order():
    h = FullContextHistory()
    h.observe(event(1, sender="B", dest="C", kind="CASH_OUT"))
    assert as_features(h, event(6, dest="B"))["recipient_cashout_count_6h"] == 1
    assert as_features(h, event(7, dest="B"))["recipient_cashout_count_6h"] == 0
    with pytest.raises(ValueError, match="Out-of-order"):
        h.features(event(6))


def test_history_can_be_rebuilt_from_committed_events():
    events = [
        event(1, sender="A", dest="B"),
        event(1, sender="C", dest="B"),
        event(2, sender="A", dest="D", kind="CASH_OUT"),
    ]
    h = FullContextHistory()
    for row in events:
        h.features(row)
        h.observe(row)
    rebuilt = FullContextHistory()
    for row in events:
        rebuilt.observe(row)
    query = event(3, sender="A", dest="B", amount=100)
    assert h.features(query) == rebuilt.features(query)
