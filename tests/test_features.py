import math

import pytest

from src.features import (
    RecipientHistory,
    make_feature_row,
    BASE_FEATURE_NAMES,
    MODEL_B_FEATURE_NAMES,
)


def transaction(step, recipient, txn_type="TRANSFER",
                amount=100, balance=500, fraud=0):
    return {
        "step": str(step),
        "type": txn_type,
        "amount": str(amount),
        "oldbalanceOrg": str(balance),
        "nameDest": recipient,
        "isFraud": str(fraud),
        "newbalanceOrig": "0",
        "newbalanceDest": "0",
    }


def test_first_recipient_visit_has_zero_history():
    history = RecipientHistory()
    features = make_feature_row(transaction(1, "X"), history)

    assert len(features) == 8
    assert features[-1] == 0


def test_current_event_is_not_counted_before_observation():
    history = RecipientHistory()
    row = transaction(1, "X")

    assert make_feature_row(row, history)[-1] == 0
    assert make_feature_row(row, history)[-1] == 0

    history.observe("X", 1)

    assert make_feature_row(row, history)[-1] == pytest.approx(
        math.log1p(1)
    )


def test_same_step_events_use_csv_order():
    history = RecipientHistory()

    row = transaction(2, "X")
    assert make_feature_row(row, history)[-1] == 0
    history.observe("X", 2)

    assert make_feature_row(row, history)[-1] == pytest.approx(
        math.log1p(1)
    )
    history.observe("X", 2)

    assert make_feature_row(row, history)[-1] == pytest.approx(
        math.log1p(2)
    )


def test_six_step_window_expires_correctly():
    history = RecipientHistory()

    for step in (1, 2, 3):
        row = transaction(step, "X")
        make_feature_row(row, history)
        history.observe("X", step)

    # At step 7, step 1 is excluded.
    row7 = transaction(7, "X")
    assert make_feature_row(row7, history)[-1] == pytest.approx(
        math.log1p(2)
    )

    # Do not observe step 7.
    # At step 8, step 2 also expires.
    row8 = transaction(8, "X")
    assert make_feature_row(row8, history)[-1] == pytest.approx(
        math.log1p(1)
    )


def test_future_events_are_not_visible():
    history = RecipientHistory()
    row = transaction(1, "X")
    features = make_feature_row(row, history)

    assert features[-1] == 0

    history.observe("X", 1)

    with pytest.raises(ValueError):
        make_feature_row(transaction(0, "X"), history)


def test_labels_and_post_balances_do_not_affect_features():
    history = RecipientHistory()

    a = transaction(5, "X", fraud=0)
    b = transaction(5, "X", fraud=1)

    b["newbalanceOrig"] = "999999"
    b["newbalanceDest"] = "123456"
    b["isFlaggedFraud"] = "1"

    assert make_feature_row(a, history) == make_feature_row(b, history)


def test_model_a_is_prefix_of_model_b():
    history = RecipientHistory()
    features = make_feature_row(
        transaction(1, "X", "CASH_OUT", 250, 400),
        history,
    )

    assert len(BASE_FEATURE_NAMES) == 7
    assert len(MODEL_B_FEATURE_NAMES) == 8

    assert features[0] == pytest.approx(math.log1p(250))
    assert features[2] == 1.0  # CASH_OUT one-hot
    assert features[6] == pytest.approx(math.log1p(400))


def test_invalid_amount_is_rejected():
    history = RecipientHistory()

    with pytest.raises(ValueError):
        make_feature_row(
            transaction(1, "X", amount=-10),
            history,
        )
