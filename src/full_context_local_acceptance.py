"""Bounded train-prefix in-process acceptance, no fraud labels or test holdout.

Use trusted, locally generated model artifacts only. Not a network benchmark.
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

from .full_context import FullContextHistory, FULL_FEATURE_NAMES
from .full_context_evidence import build_evidence
from .full_context_materialize import DEFAULT_CSV
from .serve_full_context_lab import open_local_lab

DEFAULT_OUTPUT = Path("data/processed/full_context_local_acceptance_v1.json")


def source_events(path, count):
    if not 2 <= count <= 2000:
        raise ValueError("Bounded acceptance accepts 2..2000 events")
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"step", "type", "amount", "oldbalanceOrg", "nameOrig", "nameDest"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Missing PaySim source columns")
        # Never inspect labels or read any row after the requested prefix.
        for i in range(count):
            try:
                row = next(reader)
            except StopIteration:
                raise ValueError("CSV prefix is shorter than requested") from None
            if int(row["step"]) > 238:
                raise RuntimeError("Train-prefix boundary crossed; refusing")
            yield {
                "event_id": f"acceptance-train-row-{i}",
                "source_row_index": i,
                "step": int(row["step"]),
                "type": row["type"],
                "amount": float(row["amount"]),
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "nameOrig": row["nameOrig"],
                "nameDest": row["nameDest"],
            }


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run(csv_path=DEFAULT_CSV, output=DEFAULT_OUTPUT, count=300,
        password=None, privacy_key=None):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Acceptance report exists; refusing overwrite")
    if not password or len(password) < 16 or not privacy_key or len(privacy_key) < 16:
        raise RuntimeError("Acceptance requires password and pseudonymization key")
    start_wall = time.perf_counter()
    delays = []
    model_version = policy_version = None
    independent = FullContextHistory()
    reviewed = 0
    last_event = None
    with tempfile.TemporaryDirectory(prefix="fgz_acceptance_") as folder:
        path = Path(folder) / "test.sqlite3"
        app, journal = open_local_lab(path, privacy_key, password)
        token = base64.b64encode(f"fraudguard:{password}".encode()).decode()
        headers = {"Authorization": "Basic " + token}
        restarts = []
        try:
            assert app.test_client().get("/health").status_code == 401
            with app.test_client() as client:
                assert client.get("/", headers=headers).status_code == 200
                assert client.get("/health", headers=headers).json["simulation_only"] is True
            for ordinal, event in enumerate(source_events(csv_path, count), start=1):
                if ordinal == count // 2 + 1:
                    journal.close()
                    before = time.perf_counter_ns()
                    app, journal = open_local_lab(path, privacy_key, password)
                    restarts.append((time.perf_counter_ns()-before)/1e6)
                    with app.test_client() as client:
                        status = client.get("/health", headers=headers).json
                        assert status["committed_events"] == ordinal-1
                        assert status["reviewed_events"] == reviewed
                        duplicate = client.post(
                            "/transactions/score", headers=headers, json=last_event
                        )
                        assert duplicate.status_code == 200 and duplicate.json["duplicate"]
                        assert client.get("/stats", headers=headers).json["reviewed"] == reviewed
                expected = independent.features(event)
                with app.test_client() as client:
                    tick = time.perf_counter_ns()
                    response = client.post(
                        "/transactions/score", headers=headers, json=event
                    )
                    delays.append((time.perf_counter_ns() - tick)/1e6)
                    if response.status_code != 201:
                        raise RuntimeError(f"Acceptance transaction HTTP {response.status_code}")
                    item = response.json
                    received = np.asarray(
                        [item["features"][name] for name in FULL_FEATURE_NAMES],
                        dtype=np.float64
                    )
                    if not np.allclose(received, expected, rtol=1e-6, atol=1e-6):
                        raise RuntimeError("Independent feature replay mismatch")
                    if item["evidence"] != build_evidence(expected):
                        raise RuntimeError("Observed evidence mismatch")
                    if item["block_enabled"] or not item["simulation_only"]:
                        raise RuntimeError("Simulation safety invariant failed")
                    if item["decision"] not in ("ALLOW", "REVIEW"):
                        raise RuntimeError("Unrecognized decision")
                    if item["decision"] == "REVIEW":
                        reviewed += 1
                    if reviewed > ordinal // 100:
                        raise RuntimeError("Prefix capacity exceeded")
                    model_version = item["model_version"]
                    policy_version = item["policy_version"]
                independent.observe(event)
                last_event = event
            with app.test_client() as client:
                stats = client.get("/stats", headers=headers).json
                health = client.get("/health", headers=headers).json
                recent = client.get("/transactions/recent?limit=3", headers=headers)
                evidence = client.get(
                    f"/transactions/{last_event['event_id']}/evidence", headers=headers
                )
                if (stats["total"] != count or stats["reviewed"] != reviewed
                        or stats["blocked"] != 0 or not health["healthy"]
                        or recent.status_code != 200 or not recent.json["transactions"]
                        or evidence.status_code != 200):
                    raise RuntimeError("Final API acceptance state mismatch")
        finally:
            journal.close()
    report = {
        "status": "LOCAL_ENGINEERING_ACCEPTANCE",
        "transaction_count": count,
        "reviewed": reviewed,
        "allowed": count-reviewed,
        "blocked": 0,
        "block_enabled": False,
        "model_version": model_version,
        "policy_version": policy_version,
        "restart_count": len(restarts),
        "restart_rebuild_ms": restarts,
        "timing_method": "in_process_flask_test_client_post_including_sqlite_commit",
        "post_latency_ms": {
            "p50": percentile(delays, 50),
            "p95": percentile(delays, 95),
            "p99": percentile(delays, 99),
            "mean": float(statistics.mean(delays)),
            "min": float(min(delays)),
            "max": float(max(delays)),
        },
        "overall_wall_seconds_including_rebuild": time.perf_counter()-start_wall,
        "warning": "Train-prefix engineering only. No labels or holdout. Cold-start contextual state; no fraud accuracy, network latency, LinuxONE or production claims."
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(run(args.csv, args.output, args.count,
                         os.getenv("FGZ_AUTH_PASSWORD"),
                         os.getenv("FGZ_PRIVACY_KEY")), indent=2))


if __name__ == "__main__":
    main()
