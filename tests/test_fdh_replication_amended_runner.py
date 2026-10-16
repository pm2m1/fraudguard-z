import json
import pytest

from src import fdh_replication_amended_runner as amended


def test_amended_paths_do_not_replace_original():
    assert amended.LOCK != amended.Path("reports/fdh_replication_lock.json")
    assert amended.START != amended.Path("reports/fdh_replication_started.json")
    assert amended.RUN_DIR != amended.Path("reports/fdh_replication_runs")


def test_amended_run_plan_retains_touched_seed():
    assert len(amended.run_plan()) == 9
    assert amended.run_plan()[0]["id"] == "seed_104729__default"


def test_amended_lock_marks_exploratory(monkeypatch):
    monkeypatch.setattr(amended, "file_hash", lambda path: "hash")
    monkeypatch.setattr(amended, "inspect_original_incident", lambda: "markerhash")
    specification = amended.build_lock()
    assert specification["original_failed_run"] == "seed_104729__default"
    assert specification["evidence_class"] == "EXPLORATORY_AMENDED_NOT_UNTOUCHED"


def test_amended_start_marker_requires_original_incident(monkeypatch, tmp_path):
    monkeypatch.setattr(amended, "START", tmp_path / "amended_started.json")
    monkeypatch.setattr(amended, "OUTPUT", tmp_path / "amended_summary.json")
    monkeypatch.setattr(amended, "RUN_DIR", tmp_path / "amended_runs")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="Original failure marker"):
        amended.no_prior_run()


def test_prepare_amended_lock_never_generates(monkeypatch, tmp_path):
    monkeypatch.setattr(amended, "LOCK", tmp_path / "amended_lock.json")
    monkeypatch.setattr(amended, "START", tmp_path / "amended_started.json")
    monkeypatch.setattr(amended, "OUTPUT", tmp_path / "amended_summary.json")
    monkeypatch.setattr(amended, "RUN_DIR", tmp_path / "amended_runs")
    monkeypatch.setattr(amended, "no_prior_run", lambda: None)
    monkeypatch.setattr(amended, "build_lock", lambda: {"status": "amended"})
    monkeypatch.setattr(amended, "generate_events", lambda *a, **k: (_ for _ in ()).throw(AssertionError("generated")))
    amended.prepare_lock()
    assert json.loads(amended.LOCK.read_text()) == {"status": "amended"}


def test_original_incident_marker_hash_mismatch(monkeypatch, tmp_path):
    original_lock = tmp_path / "frozen.json"
    original_lock.write_text('{"status":"PRE_GENERATION"}')
    original_start = tmp_path / "started.json"
    original_start.write_text(json.dumps({
        "started_at_utc": "2026-10-17T01:03:00+00:00",
        "lock_sha256": "wrong",
    }))
    original_dir = tmp_path / "runs"
    original_dir.mkdir()
    monkeypatch.setattr(amended, "ORIGINAL_LOCK", original_lock)
    monkeypatch.setattr(amended, "ORIGINAL_START", original_start)
    monkeypatch.setattr(amended, "ORIGINAL_DIR", original_dir)
    monkeypatch.setattr(amended, "ORIGINAL_OUTPUT", tmp_path / "summary.json")
    monkeypatch.setattr(amended, "git_head_bytes", lambda p: p.read_bytes())
    with pytest.raises(RuntimeError, match="does not match original lock"):
        amended.inspect_original_incident()


def test_original_incident_marker_hash_match(monkeypatch, tmp_path):
    import hashlib
    original_lock = tmp_path / "frozen.json"
    original_lock.write_text('{"status":"PRE_GENERATION"}')
    hash_value = hashlib.sha256(original_lock.read_bytes()).hexdigest()
    original_start = tmp_path / "started.json"
    original_start.write_text(json.dumps({
        "started_at_utc": "2026-10-17T01:03:00+00:00",
        "lock_sha256": hash_value,
    }))
    original_dir = tmp_path / "runs"
    original_dir.mkdir()
    monkeypatch.setattr(amended, "ORIGINAL_LOCK", original_lock)
    monkeypatch.setattr(amended, "ORIGINAL_START", original_start)
    monkeypatch.setattr(amended, "ORIGINAL_DIR", original_dir)
    monkeypatch.setattr(amended, "ORIGINAL_OUTPUT", tmp_path / "summary.json")
    monkeypatch.setattr(amended, "git_head_bytes", lambda p: p.read_bytes())
    assert amended.inspect_original_incident() == hashlib.sha256(original_start.read_bytes()).hexdigest()
