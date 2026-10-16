"""Amended, disclosed single-execution FDH-inspired synthetic replication.

Run --prepare-lock only after the generator/evaluator tests pass.
Commit the lock and verify it before considering --run.
"""
import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy
import sklearn

from .fdh_seeded_generator import (
    REGIMES, SEEDS, SimulationConfig, fingerprint, generate_events,
    generator_manifest,
)
from .fdh_replication_evaluator import (
    BLOCK, BUDGET, FINAL_END, HISTORY_WINDOW_DAYS,
    LABEL_DELAY_DAYS, MODEL_NAMES, TRAIN_END, TUNE_END, evaluate_events,
)

LOCK = Path("reports/fdh_amended_01_lock.json")
START = Path("reports/fdh_amended_01_started.json")
OUTPUT = Path("reports/fdh_amended_01_summary.json")
RUN_DIR = Path("reports/fdh_amended_01_runs")
ORIGINAL_LOCK = Path("reports/fdh_replication_lock.json")
ORIGINAL_START = Path("reports/fdh_replication_started.json")
ORIGINAL_DIR = Path("reports/fdh_replication_runs")
ORIGINAL_OUTPUT = Path("reports/fdh_replication_summary.json")
SOURCE_FILES = (
    "src/fdh_seeded_generator.py",
    "src/fdh_replication_evaluator.py",
    "src/fdh_replication_amended_runner.py",
    "src/fdh_replication_runner.py",
    "notes/fdh_replication_incident_01.md",
    "notes/fdh_amended_01_protocol.md",
    "notes/fdh_replication_evaluator_contract.md",
    "notes/fdh_replication_run_protocol.md",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_plan():
    return [
        {"seed": seed, "regime": regime,
         "id": f"seed_{seed}__{regime}"}
        for seed in SEEDS for regime in REGIMES
    ]


def protocol():
    return {
        "generator": generator_manifest(),
        "training_days": [0, TRAIN_END - 1],
        "tuning_days": [TRAIN_END, TUNE_END - 1],
        "final_days": [TUNE_END, FINAL_END - 1],
        "history_window_days": HISTORY_WINDOW_DAYS,
        "label_delay_days": LABEL_DELAY_DAYS,
        "models": list(MODEL_NAMES),
        "review_budget": BUDGET,
        "review_block_size": BLOCK,
        "score_cutoffs": "top 1 percent of tuning scores, no tuning labels",
        "ties": "chronological arrival order with per-prefix cap",
        "primary": "history HGB minus static HGB: detections and actual reviews",
        "no_model_selection_after_generation": True,
        "runs": run_plan(),
    }


def build_lock():
    return {
        "status": "AMENDED_AFTER_FAILED_FIRST_RUN",
        "original_lock_commit": "d233ce7",
        "original_failed_run": "seed_104729__default",
        "original_lock_sha256": file_hash(ORIGINAL_LOCK),
        "original_marker_sha256": inspect_original_incident(),
        "evidence_class": "EXPLORATORY_AMENDED_NOT_UNTOUCHED",
        "sources": {name: file_hash(name) for name in SOURCE_FILES},
        "protocol": protocol(),
        "environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "note": (
            "Original first run reached evaluator and failed; never represent "
            "this repeat as untouched confirmatory evidence. "
            "Dataset fingerprints recorded after generation. "
            "Three independent seeds across three paired regimes "
            "of one simplified synthetic simulator."
        ),
    }


def git_head_bytes(path):
    try:
        return subprocess.run(
            ["git", "show", "HEAD:" + str(path)],
            capture_output=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Lock and sources must be committed") from exc


def verify_lock():
    if not LOCK.is_file():
        raise RuntimeError("Missing lock: prepare, inspect and commit it first")
    if LOCK.read_bytes() != git_head_bytes(LOCK):
        raise RuntimeError("Lock differs from committed HEAD")
    for name in SOURCE_FILES:
        if Path(name).read_bytes() != git_head_bytes(name):
            raise RuntimeError(f"Source differs from committed HEAD: {name}")
    lock = json.loads(LOCK.read_text())
    if lock != build_lock():
        raise RuntimeError("Source, settings or dependency versions changed")
    return lock


def inspect_original_incident():
    """Validate preserved original marker against the original committed lock."""
    if not ORIGINAL_START.is_file() or not ORIGINAL_DIR.is_dir():
        raise RuntimeError("Original failure marker/directory missing; preserve incident evidence")
    if not ORIGINAL_LOCK.is_file():
        raise RuntimeError("Original frozen lock missing")
    if ORIGINAL_LOCK.read_bytes() != git_head_bytes(ORIGINAL_LOCK):
        raise RuntimeError("Original frozen lock differs from committed HEAD")
    try:
        marker = json.loads(ORIGINAL_START.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Original failure marker malformed") from exc
    if (
        not isinstance(marker.get("started_at_utc"), str)
        or not marker["started_at_utc"]
        or marker.get("lock_sha256") != file_hash(ORIGINAL_LOCK)
    ):
        raise RuntimeError("Original failure marker does not match original lock")
    if any(ORIGINAL_DIR.iterdir()) or ORIGINAL_OUTPUT.exists():
        raise RuntimeError("Original incident reports differ from documented first-run failure")
    return file_hash(ORIGINAL_START)


def no_prior_run():
    inspect_original_incident()
    if START.exists() or OUTPUT.exists() or RUN_DIR.exists():
        raise RuntimeError("Previous replication run artifacts exist")


def prepare_lock():
    if LOCK.exists():
        raise RuntimeError("Lock already exists; refusing overwrite")
    no_prior_run()
    value = build_lock()  # No calls to generate_events or evaluate_events
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
    print(f"Prepared {LOCK}. No synthetic events were generated.")


def check_lock():
    value = verify_lock()
    no_prior_run()
    print("PASS: committed source, protocol, dependencies and lock match")
    print("PASS: no prior replication run marker")
    print("Run count:", len(value["protocol"]["runs"]))


def summarise(reports):
    by_regime = {}
    for regime in REGIMES:
        rows = [row for row in reports if row["regime"] == regime]
        if len(rows) != len(SEEDS):
            raise RuntimeError("Missing predeclared run")
        detections = [
            row["evaluation"]["paired_comparison"][
                "extra_detected_history_minus_static"
            ] for row in rows
        ]
        reviews = [
            row["evaluation"]["model_results"]["hgb_history"]["evaluation"][
                "reviews"
            ] - row["evaluation"]["model_results"]["hgb_static"]["evaluation"][
                "reviews"
            ] for row in rows
        ]
        by_regime[regime] = {
            "seeds": list(SEEDS),
            "extra_fraud_detections_per_seed": detections,
            "review_differences_per_seed": reviews,
            "mean_detection_difference_descriptive": float(numpy.mean(detections)),
            "minimum_detection_difference": min(detections),
            "maximum_detection_difference": max(detections),
            "positive_seed_count": sum(value > 0 for value in detections),
            "warning": "Only three seeds; descriptive, not IID confidence bounds",
        }
    return by_regime


def execute_once():
    no_prior_run()
    lock = verify_lock()
    START.parent.mkdir(parents=True, exist_ok=True)
    with START.open("x") as f:
        json.dump({
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "lock_sha256": file_hash(LOCK),
            "note": "Preserve marker even if a run fails; no silent reruns",
        }, f, indent=2)
        f.write("\n")
    RUN_DIR.mkdir(exist_ok=False)
    reports = []
    for spec in lock["protocol"]["runs"]:
        seed, regime, run_id = spec["seed"], spec["regime"], spec["id"]
        print("Running:", run_id, flush=True)
        events = generate_events(seed, regime, SimulationConfig())
        record = {
            "id": run_id,
            "seed": seed,
            "regime": regime,
            "dataset_fingerprint_sha256": fingerprint(events),
            "events": len(events),
            "evaluation": evaluate_events(events),
        }
        with (RUN_DIR / (run_id + ".json")).open("x") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
        reports.append(record)

    if len(set(r["dataset_fingerprint_sha256"] for r in reports)) != len(reports):
        raise RuntimeError("Duplicate generated datasets; preserve artifacts")
    with OUTPUT.open("x") as f:
        json.dump({
            "status": "COMPLETED_EXPLORATORY_AMENDED_REPLICATION",
            "original_failed_run": "seed_104729__default",
            "not_untouched_confirmatory": True,
            "lock_sha256": file_hash(LOCK),
            "total_runs": len(reports),
            "by_regime": summarise(reports),
            "run_fingerprints": {
                r["id"]: r["dataset_fingerprint_sha256"] for r in reports
            },
            "limits": (
                "Three independent seeds and three paired regimes from one "
                "synthetic simulator. Not real-world validation. "
                "No model or review-policy promotion."
            ),
        }, f, indent=2)
        f.write("\n")
    print("Saved amended reports:", OUTPUT, "and nine detailed run reports.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-lock", action="store_true")
    modes.add_argument("--check-lock", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare_lock:
        prepare_lock()
    elif args.check_lock:
        check_lock()
    else:
        execute_once()


if __name__ == "__main__":
    main()
