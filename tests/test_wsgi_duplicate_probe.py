import pytest

from src.wsgi_duplicate_probe import (
    PAYLOAD, check_preflight, pct, validate_duplicate, validate_local_url,
)


def reference():
    return {
        "event_id": "wsgi-demo-1",
        "step": 1,
        "source_row_index": 0,
        "txn_type": "TRANSFER",
        "amount": 100,
        "risk": .01,
        "decision": "ALLOW",
        "n3": 0,
    }


def test_preflight_accepts_matching_single_event():
    row = reference()
    result = check_preflight(
        {"healthy": True, "committed_events": 1,
         "last_position": [1, 0]},
        {"transactions": [row]},
    )
    assert result == row


def test_preflight_rejects_foreign_database():
    with pytest.raises(RuntimeError):
        check_preflight(
            {"healthy": True, "committed_events": 2,
             "last_position": [1, 1]},
            {"transactions": [reference()]},
        )


def test_preflight_rejects_unexpected_event():
    row = reference()
    row["event_id"] = "different"
    with pytest.raises(RuntimeError):
        check_preflight(
            {"healthy": True, "committed_events": 1,
             "last_position": [1, 0]},
            {"transactions": [row]},
        )


def test_validate_duplicate_requires_same_decision():
    ref = reference()
    ok = {k: ref[k] for k in ("event_id", "decision", "n3", "risk")}
    ok["duplicate"] = True
    validate_duplicate(200, ok, ref)
    with pytest.raises(RuntimeError):
        validate_duplicate(201, ok, ref)
    with pytest.raises(RuntimeError):
        validate_duplicate(200, {**ok, "risk": .5}, ref)


def test_local_url_only():
    assert validate_local_url("http://127.0.0.1:5050") == "http://127.0.0.1:5050"
    for bad in (
        "http://example.com:5050",
        "https://127.0.0.1:5050",
        "http://127.0.0.1:5050/path",
        "http://admin:secret@127.0.0.1:5050",
    ):
        with pytest.raises(ValueError):
            validate_local_url(bad)


def test_percentiles():
    assert pct([1, 2, 3, 4], .5) == 2.5
    assert pct([1, 2, 3, 4], .95) == pytest.approx(3.85)
