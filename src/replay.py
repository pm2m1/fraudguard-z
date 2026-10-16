"""Replay PaySim transactions through the local FraudGuard API.

This script does not train or change the model. It streams raw dataset rows
through the same endpoint used by the dashboard.
"""

import argparse
import csv
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

DATA_PATH = Path("data/raw/PS_20174392719_1491204439457_log.csv")
REPORT_PATH = Path("reports/replay_benchmark.json")
REQUIRED_COLUMNS = {
    "step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg"
}


def iter_transactions(path, limit):
    if limit < 1:
        raise ValueError("Transaction limit must be positive")

    previous_step = -1
    count = 0
    with Path(path).open(newline="") as file:
        reader = csv.DictReader(file)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError("PaySim CSV is missing required columns")

        for source_row_index, row in enumerate(reader):
            if count >= limit:
                break

            step = int(row["step"])
            if step < previous_step:
                raise ValueError("PaySim rows are out of chronological order")
            previous_step = step

            yield {
                "event_id": f"paysim-{source_row_index}",
                "source_row_index": source_row_index,
                "step": step,
                "type": row["type"],
                "amount": float(row["amount"]),
                "nameOrig": row["nameOrig"],
                "nameDest": row["nameDest"],
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
            }
            count += 1

    if count != limit:
        raise ValueError(f"CSV contains only {count} rows, expected {limit}")


def percentile(sorted_values, fraction):
    if not sorted_values:
        return 0.0
    location = (len(sorted_values) - 1) * fraction
    lower = int(location)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = location - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def run_replay(events, submit, *, delay_ms=0.0, progress_every=100):
    if delay_ms < 0:
        raise ValueError("Delay must be nonnegative")
    if progress_every < 1:
        raise ValueError("Progress interval must be positive")

    latencies = []
    decisions = Counter()
    previous_recipient_count_positive = 0
    count = 0
    started_at = datetime.now(timezone.utc).isoformat()
    begin = time.perf_counter()

    for event in events:
        start = time.perf_counter()
        result = submit(event)
        latency_ms = (time.perf_counter() - start) * 1000

        if result.get("event_id") != event["event_id"]:
            raise RuntimeError("API returned a different event ID")
        if result.get("duplicate"):
            raise RuntimeError("Replay encountered a duplicate. Use an empty replay database")
        if result.get("decision") not in ("ALLOW", "REVIEW"):
            raise RuntimeError("API returned an unexpected decision")

        count += 1
        latencies.append(latency_ms)
        decisions[result["decision"]] += 1
        previous_recipient_count_positive += int(result["n3"] > 0)

        if count % progress_every == 0:
            print(f"Processed {count:,} transactions", flush=True)
        if delay_ms:
            time.sleep(delay_ms / 1000)

    elapsed = time.perf_counter() - begin
    if count == 0:
        raise ValueError("No transactions were replayed")

    latencies.sort()
    report = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "records": count,
        "duration_seconds": round(elapsed, 3),
        "effective_transactions_per_second": round(count / elapsed, 3),
        "http_latency_ms": {
            "mean": round(sum(latencies) / count, 3),
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "p99": round(percentile(latencies, 0.99), 3),
        },
        "decisions": dict(decisions),
        "transactions_with_n3_positive": previous_recipient_count_positive,
        "delay_ms_per_transaction": delay_ms,
        "client_system": platform.system(),
        "client_architecture": platform.machine(),
        "notes": "Client-observed HTTP latency includes API, model and SQLite; measured serially.",
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--csv", type=Path, default=DATA_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--url", default="http://127.0.0.1:5050")
    args = parser.parse_args()

    if args.count < 1 or args.count > 100_000:
        parser.error("--count must be between 1 and 100000")
    if args.delay_ms < 0:
        parser.error("--delay-ms must be nonnegative")

    parsed = urlsplit(args.url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path not in ("", "/"):
        parser.error("Replay only supports a local HTTP server at 127.0.0.1")

    url = args.url.rstrip("/")
    opener = build_opener(ProxyHandler({}))

    def json_request(path, payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = Request(
            url + path,
            data=body,
            headers={"Content-Type": "application/json"} if body is not None else {},
            method="POST" if body is not None else "GET",
        )
        try:
            with opener.open(req, timeout=15) as response:
                return json.load(response)
        except HTTPError as error:
            detail = error.read(500).decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {error.code} from {path}: {detail}") from error
        except URLError as error:
            raise RuntimeError(f"Cannot reach API: {error.reason}") from error

    health = json_request("/health")
    if (
        not health.get("healthy")
        or health.get("committed_events") != 0
        or health.get("last_position") != [-1, -1]
    ):
        raise RuntimeError(
            "Replay needs a healthy empty database. "
            "Start the server with FGZ_DB_PATH=data/runtime/replay.sqlite3 "
            "and use a new database filename for each run."
        )

    print(f"Replaying {args.count:,} transactions through {url}")
    report = run_replay(
        iter_transactions(args.csv, args.count),
        lambda event: json_request("/transactions/score", event),
        delay_ms=args.delay_ms,
        progress_every=args.progress_every,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    print("\nReplay complete")
    print(f"Throughput: {report['effective_transactions_per_second']:.1f} txn/s")
    print(f"p50 HTTP latency: {report['http_latency_ms']['p50']:.2f} ms")
    print(f"p95 HTTP latency: {report['http_latency_ms']['p95']:.2f} ms")
    print(f"p99 HTTP latency: {report['http_latency_ms']['p99']:.2f} ms")
    print(f"Decisions: {report['decisions']}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
