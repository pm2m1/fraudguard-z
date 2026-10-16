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
    specification = amended.build_lock()
    assert specification["original_failed_run"] == "seed_104729__default"
    assert specification["evidence_class"] == "EXPLORATORY_AMENDED_NOT_UNTOUCHED"


def test_amended_start_marker_requires_original_incident(monkeypatch, tmp_path):
    monkeypatch.setattr(amended, "START", tmp_path / "amended_started.json")
    monkeypatch.setattr(amended, "OUTPUT", tmp_path / "amended_summary.json")
    monkeypatch.setattr(amended, "RUN_DIR", tmp_path / "amended_runs")
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
