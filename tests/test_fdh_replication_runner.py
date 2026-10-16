import json
from pathlib import Path

import pytest

from src import fdh_replication_runner as runner


def test_run_matrix_fixed_and_unique():
    plan = runner.run_plan()
    assert len(plan) == 9
    assert len({p["id"] for p in plan}) == 9
    assert {p["seed"] for p in plan} == {104729, 130363, 155921}


def test_frozen_split_contract():
    spec = runner.protocol()
    assert spec["training_days"] == [0, 53]
    assert spec["tuning_days"] == [54, 71]
    assert spec["final_days"] == [72, 89]
    assert spec["review_budget"] == .01
    assert spec["no_model_selection_after_generation"] is True


def test_prepare_lock_does_not_generate(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "LOCK", tmp_path / "lock.json")
    monkeypatch.setattr(runner, "START", tmp_path / "started.json")
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "summary.json")
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path / "runs")
    monkeypatch.setattr(runner, "build_lock", lambda: {"status": "PRE_GENERATION"})
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not generate/evaluate")
    monkeypatch.setattr(runner, "generate_events", forbidden)
    monkeypatch.setattr(runner, "evaluate_events", forbidden)
    runner.prepare_lock()
    assert json.loads(runner.LOCK.read_text()) == {"status": "PRE_GENERATION"}
    with pytest.raises(RuntimeError, match="already exists"):
        runner.prepare_lock()


def test_check_lock_blocks_started_run(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "START", tmp_path / "started.json")
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "summary.json")
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path / "runs")
    runner.START.write_text("{}")
    with pytest.raises(RuntimeError, match="Previous replication"):
        runner.no_prior_run()


def test_lock_fails_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "LOCK", tmp_path / "missing.json")
    with pytest.raises(RuntimeError, match="Missing lock"):
        runner.verify_lock()


def test_descriptive_summary_preserves_all_three_seeds():
    reports = []
    for p in runner.run_plan():
        reports.append({
            "seed": p["seed"], "regime": p["regime"],
            "evaluation": {
                "paired_comparison": {
                    "extra_detected_history_minus_static": 2
                },
                "model_results": {
                    "hgb_history": {"evaluation": {"reviews": 10}},
                    "hgb_static": {"evaluation": {"reviews": 11}},
                },
            },
        })
    summary = runner.summarise(reports)
    assert len(summary) == 3
    assert all(v["positive_seed_count"] == 3 for v in summary.values())
    assert all(v["review_differences_per_seed"] == [-1, -1, -1]
               for v in summary.values())
