"""Phase 6F: 10k/25k train-prefix full-context replay engineering acceptance.

No fraud labels, model fitting, calibration, final holdout, network listener,
automatic BLOCK, or production claims. See notes/full_context_scale_acceptance_gate.md.
"""
import argparse
import base64
import csv
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from .full_context import FULL_FEATURE_NAMES, FullContextHistory
from .full_context_evidence import build_evidence
from .full_context_journal import FEATURE_SCHEMA_SHA256
from .full_context_materialize import DEFAULT_CSV, DEFAULT_MANIFEST, file_sha256
from .serve_full_context_lab import open_local_lab

REFERENCE_DIR = Path("data/processed/full_context_dev_v1")
MIN_COUNT, MAX_COUNT = 10, 25_000


def validate_inputs(count, password, privacy_key, csv_path, reference_dir,
                    manifest_path, output):
    if type(count) is not int or not MIN_COUNT <= count <= MAX_COUNT:
        raise ValueError("Phase 6F count must be between 10 and 25000")
    if not isinstance(password, str) or len(password) < 16:
        raise RuntimeError("Phase 6F requires a password of at least 16 characters")
    if (not isinstance(privacy_key, str)
            or len(privacy_key.encode("utf-8")) < 32):
        raise RuntimeError("Phase 6F privacy key must have at least 32 UTF-8 bytes")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Phase 6F refuses to overwrite an existing report")
    if not csv_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Original PaySim CSV and frozen split manifest required")
    metadata_path = reference_dir / "materialization.json"
    feature_path = reference_dir / "features.npy"
    if (not metadata_path.is_file() or not feature_path.is_file()
            or metadata_path.is_symlink() or feature_path.is_symlink()):
        raise FileNotFoundError("Verified development reference features required")
    meta = json.loads(metadata_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (meta.get("status") != "DEVELOPMENT_ONLY_FEATURE_MATERIALIZATION"
            or meta.get("schema") != list(FULL_FEATURE_NAMES)
            or meta.get("schema_sha256") != FEATURE_SCHEMA_SHA256
            or meta.get("split_manifest_sha256") != file_sha256(manifest_path)
            or meta.get("dtype") != "float32"
            or meta.get("rows") != manifest.get("development_row_count")
            or meta.get("partitions") != manifest.get("partitions")
            or manifest.get("dev_max_step") != 372
            or manifest.get("final_holdout_start_step") != 373
            or manifest.get("partitions", {}).get("train", {}).get("rows", 0) < count):
        raise RuntimeError("Unrecognized or inconsistent development reference provenance")
    # Read-only mmap: never load labels.npy or any other partition's labels.
    X = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if (X.ndim != 2 or X.shape != (meta["rows"], len(FULL_FEATURE_NAMES))
            or X.dtype != np.dtype("float32")):
        raise RuntimeError("Development feature matrix shape/dtype mismatch")
    return X, meta


def train_source_events(csv_path, count):
    """Consume only the requested CSV prefix; never access isFraud."""
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"step", "type", "amount", "oldbalanceOrg",
                    "nameOrig", "nameDest"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Missing required PaySim CSV columns")
        for i in range(count):
            try:
                row = next(reader)
            except StopIteration:
                raise ValueError("CSV shorter than requested training prefix") from None
            step = int(row["step"])
            if not 1 <= step <= 238:
                raise RuntimeError("Train-prefix boundary crossed; refusing replay")
            yield {
                "event_id": f"phase6f-train-row-{i}",
                "step": step,
                "source_row_index": i,
                "type": row["type"],
                "amount": float(row["amount"]),
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "nameOrig": row["nameOrig"],
                "nameDest": row["nameDest"],
            }


def db_sizes(db_path):
    return {
        name: (path.stat().st_size if path.is_file() else 0)
        for name, path in (
            ("main", db_path),
            ("wal", Path(str(db_path) + "-wal")),
            ("shm", Path(str(db_path) + "-shm")),
        )
    }


def validate_checkpoint(app, journal, headers, count, reviewed, db_path):
    with app.test_client() as client:
        stats = client.get("/stats", headers=headers)
        health = client.get("/health", headers=headers)
        if stats.status_code != 200 or health.status_code != 200:
            raise RuntimeError("Health/stats request failed at checkpoint")
        s, h = stats.json, health.json
        if (s["total"] != count or s["reviewed"] != reviewed
                or s["allowed"] != count - reviewed or s["blocked"] != 0
                or h["committed_events"] != count
                or h["reviewed_events"] != reviewed or not h["healthy"]
                or h["block_enabled"] or not h["simulation_only"]):
            raise RuntimeError("Checkpoint counter or safety mismatch")
    if journal.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise RuntimeError("SQLite quick_check failed")
    return {"events": count, "reviews": reviewed, "sqlite_bytes": db_sizes(db_path)}


def run(csv_path=DEFAULT_CSV, reference_dir=REFERENCE_DIR,
        manifest_path=DEFAULT_MANIFEST, output=None, count=10_000,
        password=None, privacy_key=None):
    csv_path, reference_dir, manifest_path = (
        Path(csv_path), Path(reference_dir), Path(manifest_path)
    )
    output = (Path(output) if output is not None else Path(
        f"data/processed/full_context_scale_{count}_v1.json"
    ))
    X, meta = validate_inputs(
        count, password, privacy_key, csv_path, reference_dir,
        manifest_path, output
    )

    all_started = time.perf_counter()
    delays_ms = []
    snapshots, restarts = [], []
    expected_state = FullContextHistory()
    reviewed = 0
    expected_model = expected_policy = None
    last_event = None
    checkpoints = {max(1, count // 4), count // 2,
                   (3 * count) // 4, count}

    with tempfile.TemporaryDirectory(prefix="fgz_phase6f_") as temporary:
        db_path = Path(temporary) / "replay.sqlite3"
        app, journal = open_local_lab(db_path, privacy_key, password)
        expected_model, expected_policy = journal.model_version, journal.policy_version
        token = base64.b64encode(f"fraudguard:{password}".encode("utf-8")).decode()
        headers = {"Authorization": "Basic " + token}

        def verify_replay_and_reopen(at_count, last):
            nonlocal app, journal
            journal.close()
            start = time.perf_counter_ns()
            app, journal = open_local_lab(db_path, privacy_key, password)
            duration = (time.perf_counter_ns() - start) / 1e6
            if (journal.model_version != expected_model
                    or journal.policy_version != expected_policy):
                raise RuntimeError("Model/policy changed across restart")
            validate_checkpoint(app, journal, headers, at_count, reviewed, db_path)
            with app.test_client() as client:
                duplicate = client.post(
                    "/transactions/score", headers=headers, json=last
                )
                if (duplicate.status_code != 200
                        or duplicate.json.get("duplicate") is not True):
                    raise RuntimeError("Idempotent decision did not survive restart")
                evidence = client.get(
                    f"/transactions/{last['event_id']}/evidence", headers=headers
                )
                if (evidence.status_code != 200
                        or evidence.json.get("evidence") !=
                        duplicate.json.get("evidence")):
                    raise RuntimeError("Stored evidence did not survive restart")
            validate_checkpoint(app, journal, headers, at_count, reviewed, db_path)
            restarts.append({"after_events": at_count, "rebuild_ms": duration})

        try:
            with app.test_client() as client:
                if client.get("/health").status_code != 401:
                    raise RuntimeError("Unauthenticated access was not denied")
            for i, event in enumerate(train_source_events(csv_path, count)):
                expected_raw = expected_state.features(event)
                ref = np.asarray(X[i], dtype=np.float32)
                # Local independent running-state features vs frozen materialization.
                if not np.allclose(ref, expected_raw, rtol=1e-6, atol=1e-6):
                    raise RuntimeError(f"Source/reference mismatch at row {i}")
                expected_score = float(journal.score_calibrated(
                    ref.reshape(1, len(FULL_FEATURE_NAMES))
                ))
                with app.test_client() as client:
                    start_ns = time.perf_counter_ns()
                    response = client.post(
                        "/transactions/score", headers=headers, json=event
                    )
                    delays_ms.append((time.perf_counter_ns() - start_ns) / 1e6)
                if response.status_code != 201:
                    raise RuntimeError(
                        f"New replay event {i} returned HTTP {response.status_code}"
                    )
                item = response.json
                if (item["duplicate"] or not item["simulation_only"]
                        or item["block_enabled"]
                        or item["model_version"] != expected_model
                        or item["policy_version"] != expected_policy
                        or item["decision"] not in ("ALLOW", "REVIEW")):
                    raise RuntimeError(f"Unsafe or inconsistent decision at row {i}")
                actual = np.asarray(
                    [item["features"][name] for name in FULL_FEATURE_NAMES],
                    dtype=np.float64,
                )
                if not np.allclose(actual, ref, rtol=1e-6, atol=1e-6):
                    raise RuntimeError(f"21-feature mismatch at row {i}")
                if not np.isclose(
                    float(item["risk"]), expected_score, rtol=1e-6, atol=1e-12
                ):
                    raise RuntimeError(f"Verified model score mismatch at row {i}")
                if item["evidence"] != build_evidence(expected_raw):
                    raise RuntimeError(f"Evidence mismatch at row {i}")
                reviewed += item["decision"] == "REVIEW"
                if reviewed > (i + 1) // 100:
                    raise RuntimeError(f"Prefix REVIEW budget exceeded at row {i}")
                expected_state.observe(event)
                last_event = event
                position = i + 1
                if position in checkpoints:
                    snapshots.append(validate_checkpoint(
                        app, journal, headers, position, reviewed, db_path
                    ))
                    if position == count // 2:
                        verify_replay_and_reopen(position, last_event)
            verify_replay_and_reopen(count, last_event)
        finally:
            journal.close()

    arr = np.asarray(delays_ms, dtype=np.float64)
    result = {
        "status": "PASS_LOCAL_ENGINEERING_SCALE_ACCEPTANCE",
        "scope": "PaySim training-prefix engineering only; no labels loaded",
        "transaction_count": count,
        "feature_count": len(FULL_FEATURE_NAMES),
        "verified_feature_rows": count,
        "schema_sha256": FEATURE_SCHEMA_SHA256,
        "split_manifest_sha256": meta["split_manifest_sha256"],
        "reviewed": int(reviewed),
        "allowed": int(count - reviewed),
        "blocked": 0,
        "block_enabled": False,
        "model_version": expected_model,
        "policy_version": expected_policy,
        "checkpoints": snapshots,
        "restart_measurements": restarts,
        "latency_method": "in-process Flask test_client POST including SQLite commit",
        "post_latency_ms": {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(statistics.mean(delays_ms)),
            "max": float(max(delays_ms)),
        },
        "overall_wall_seconds_including_replay_validation": (
            time.perf_counter() - all_started
        ),
        "limitations": (
            "Neither a network, concurrency, LinuxONE nor production benchmark. "
            "No detection/recall claim; reference shares feature semantics. "
            "Original final holdout untouched, experimental BLOCK disabled."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(
        csv_path=args.csv, reference_dir=args.reference_dir,
        manifest_path=args.manifest, output=args.output, count=args.count,
        password=os.getenv("FGZ_AUTH_PASSWORD"),
        privacy_key=os.getenv("FGZ_PRIVACY_KEY"),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
