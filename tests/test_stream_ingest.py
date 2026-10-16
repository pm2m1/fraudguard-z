import pytest

from src.stream_ingest import validate_resume, run_bounded, LocalAPI


def test_empty_stream_start():
    assert validate_resume(
        {"healthy": True, "committed_events": 0, "last_position": [-1, -1]}, [],
        resume=False
    ) == 0


def test_safe_prefix_resume():
    assert validate_resume(
        {"healthy": True, "committed_events": 2, "last_position": [1, 1]},
        [{"event_id": "paysim-1", "source_row_index": 1, "step": 1}],
        resume=True,
    ) == 2


@pytest.mark.parametrize("latest", [
    [{"event_id": "ui-1", "source_row_index": 1, "step": 1}],
    [{"event_id": "paysim-1", "source_row_index": 999, "step": 1}],
    [],
])
def test_refuses_unverified_resume(latest):
    with pytest.raises(RuntimeError):
        validate_resume(
            {"healthy": True, "committed_events": 2, "last_position": [1, 1]},
            latest, resume=True,
        )


def test_bounded_consumer_preserves_order():
    inputs = [{"event_id": f"paysim-{i}"} for i in range(50)]
    observed = []
    def submit(item):
        observed.append(item["event_id"])
        return {"event_id": item["event_id"], "decision": "ALLOW",
                "n3": 0, "duplicate": False}
    report = run_bounded(inputs, submit, queue_capacity=1, progress_every=100)
    assert observed == [e["event_id"] for e in inputs]
    assert report["new_transactions"] == 50
    assert report["queue_capacity"] == 1


def test_bad_response_fails_without_hanging():
    inputs = ({"event_id": f"paysim-{i}"} for i in range(10000))
    with pytest.raises(RuntimeError, match="event ID mismatch"):
        run_bounded(inputs, lambda event: {
            "event_id": "wrong", "decision": "ALLOW", "n3": 0
        }, queue_capacity=1)


def test_rejects_nonlocal_api():
    with pytest.raises(ValueError):
        LocalAPI("https://example.com")


def test_no_resume_without_flag():
    with pytest.raises(RuntimeError, match="not empty"):
        validate_resume(
            {"healthy": True, "committed_events": 1, "last_position": [1, 0]},
            [{"event_id": "paysim-0", "source_row_index": 0, "step": 1}],
            resume=False,
        )
