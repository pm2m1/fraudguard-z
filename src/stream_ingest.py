"""Bounded, resumable PaySim HTTP ingestion for the single-owner decision engine.

This is a controlled local replay, not a general-purpose message broker.
The producer reads the CSV into a bounded queue; one consumer preserves
transaction order and relies on server-side idempotency for retry safety.
"""
import argparse
import base64
import itertools
import json
import platform
import queue
import threading
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .replay import DATA_PATH, iter_transactions, percentile

END = object()
RETRYABLE_HTTP = frozenset({429, 500, 502, 503, 504})


def validate_resume(health, latest, *, resume):
    """Return already-committed PaySim row count, or fail closed."""
    if not health.get("healthy"):
        raise RuntimeError("Scoring engine is not healthy")
    committed = health.get("committed_events")
    last = health.get("last_position")
    if type(committed) is not int or committed < 0:
        raise RuntimeError("Invalid committed-event count")
    if committed == 0:
        if last != [-1, -1]:
            raise RuntimeError("Empty database has unexpected position")
        return 0
    if not resume:
        raise RuntimeError("Database is not empty; use --resume only for a PaySim-only stream")
    if not isinstance(latest, list) or len(latest) != 1:
        raise RuntimeError("Cannot verify stream ownership of database")
    event = latest[0]
    expected_id = f"paysim-{committed - 1}"
    if (
        event.get("event_id") != expected_id
        or event.get("source_row_index") != committed - 1
        or last != [event.get("step"), committed - 1]
    ):
        raise RuntimeError("Database contains noncontiguous or non-PaySim events; refusing resume")
    return committed


def run_bounded(events, submit, *, queue_capacity=64, progress_every=1000):
    if queue_capacity < 1 or progress_every < 1:
        raise ValueError("Queue capacity and progress interval must be positive")

    pending = queue.Queue(maxsize=queue_capacity)
    stopped = threading.Event()
    producer_errors = []
    stats = Counter()
    latencies = []
    began = time.perf_counter()
    started_utc = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat()

    def offer(value):
        while not stopped.is_set():
            try:
                pending.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce():
        try:
            for event in events:
                if not offer(event):
                    return
        except Exception as error:
            producer_errors.append(error)
        finally:
            offer(END)

    worker = threading.Thread(target=produce, name="paysim-producer", daemon=True)
    worker.start()
    error = None
    try:
        while True:
            item = pending.get()
            if item is END:
                break
            started = time.perf_counter()
            result = submit(item)
            latencies.append((time.perf_counter() - started) * 1000)
            if result.get("event_id") != item["event_id"]:
                raise RuntimeError("API event ID mismatch")
            if result.get("decision") not in ("ALLOW", "REVIEW"):
                raise RuntimeError("API returned invalid decision")
            if not isinstance(result.get("n3"), int) or result["n3"] < 0:
                raise RuntimeError("API returned invalid history count")
            stats["completed"] += 1
            stats[result["decision"]] += 1
            stats["duplicates_after_retry"] += int(bool(result.get("duplicate")))
            if stats["completed"] % progress_every == 0:
                print(f"Committed {stats['completed']:,} events", flush=True)
    except Exception as exc:
        error = exc
    finally:
        stopped.set()
        worker.join(timeout=5)
    if worker.is_alive():
        raise RuntimeError("Producer did not stop safely")
    if error is not None:
        raise error
    if producer_errors:
        raise RuntimeError("CSV producer stopped") from producer_errors[0]
    if not stats["completed"]:
        raise ValueError("No new transactions were streamed")

    duration = time.perf_counter() - began
    latencies.sort()
    return {
        "status": "completed",
        "client_architecture": platform.machine(),
        "client_system": platform.system(),
        "started_utc": started_utc,
        "new_transactions": stats["completed"],
        "decisions": {"ALLOW": stats["ALLOW"], "REVIEW": stats["REVIEW"]},
        "duplicate_responses_after_retry": stats["duplicates_after_retry"],
        "queue_capacity": queue_capacity,
        "single_ordered_consumer": True,
        "duration_seconds": round(duration, 3),
        "throughput_tps": round(stats["completed"] / duration, 3),
        "http_latency_ms": {
            "p50": round(percentile(latencies, .50), 3),
            "p95": round(percentile(latencies, .95), 3),
            "p99": round(percentile(latencies, .99), 3),
        },
        "limitations": (
            "Serial local HTTP requests, one consumer; not a concurrency or "
            "sustained-ingestion benchmark. Safe resume requires PaySim-only "
            "database and unchanged CSV. Network timeouts can be retried because "
            "the engine stores an idempotent event decision."
        ),
    }


class LocalAPI:
    def __init__(self, url, password=None, attempts=3, retry_delay=.2):
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path not in ("", "/"):
            raise ValueError("This stream tool only supports HTTP localhost 127.0.0.1")
        if attempts < 1 or retry_delay < 0:
            raise ValueError("Invalid retry configuration")
        self.base = url.rstrip("/")
        self.opener = build_opener(ProxyHandler({}))
        self.attempts = attempts
        self.retry_delay = retry_delay
        self.headers = {}
        if password:
            token = base64.b64encode(f"fraudguard:{password}".encode()).decode()
            self.headers["Authorization"] = "Basic " + token

    def request(self, path, event=None):
        payload = json.dumps(event).encode() if event is not None else None
        failed_attempt = False
        for attempt in range(self.attempts):
            headers = dict(self.headers)
            if payload is not None:
                headers["Content-Type"] = "application/json"
            request = Request(
                self.base + path, data=payload, headers=headers,
                method="POST" if payload is not None else "GET",
            )
            try:
                with self.opener.open(request, timeout=15) as response:
                    result = json.load(response)
                if event is not None and result.get("duplicate") and not failed_attempt:
                    raise RuntimeError("Unexpected duplicate without a prior transport failure")
                return result
            except HTTPError as exc:
                if exc.code not in RETRYABLE_HTTP or attempt == self.attempts - 1:
                    raise RuntimeError(f"HTTP {exc.code} at {path}") from exc
            except (URLError, TimeoutError, ConnectionError) as exc:
                if attempt == self.attempts - 1:
                    raise RuntimeError("Local API failed after retries") from exc
            failed_attempt = True
            time.sleep(self.retry_delay * (2 ** attempt))
        raise RuntimeError("Exhausted local API attempts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DATA_PATH)
    parser.add_argument("--count", type=int, default=5000,
                        help="Total PaySim prefix length, including already committed rows")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--queue-size", type=int, default=64)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--report", type=Path, default=Path("reports/bounded_stream.json"))
    parser.add_argument("--url", default="http://127.0.0.1:5050")
    args = parser.parse_args()
    if args.count < 1 or args.count > 1000000:
        parser.error("--count must be between 1 and 1000000")
    if args.queue_size < 1 or args.queue_size > 10000:
        parser.error("--queue-size must be between 1 and 10000")

    import os
    client = LocalAPI(args.url, os.environ.get("FGZ_AUTH_PASSWORD"),
                      attempts=args.retries)
    health = client.request("/health")
    latest = client.request("/transactions/recent?limit=1")["transactions"]
    already = validate_resume(health, latest, resume=args.resume)
    if already >= args.count:
        raise RuntimeError(f"Nothing to stream: {already} events already committed")
    print(f"Continuing PaySim prefix [{already}, {args.count}) with queue={args.queue_size}")
    events = itertools.islice(iter_transactions(args.csv, args.count), already, args.count)
    report = run_bounded(
        events,
        lambda event: client.request("/transactions/score", event),
        queue_capacity=args.queue_size,
    )
    end = client.request("/health")
    if end.get("committed_events") != args.count:
        raise RuntimeError("Server event count mismatches requested prefix")
    report["total_committed_events"] = args.count
    report["resumed_from_row"] = already
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Processed {report['new_transactions']} new transactions")
    print(f"Throughput: {report['throughput_tps']:.1f} txn/s")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
