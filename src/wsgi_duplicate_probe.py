"""Bounded local duplicate-request load and idempotency probe.

Run ONLY against the dedicated one-event WSGI demo database. This probes
concurrent duplicate HTTP requests to a single-worker Gunicorn service,
NOT concurrent distinct payment submissions or ML inference throughput.
Secrets are never printed, placed on the command line, or saved in reports.
"""
import argparse
import base64
import getpass
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

PAYLOAD = {
    "event_id": "wsgi-demo-1",
    "source_row_index": 0,
    "step": 1,
    "type": "TRANSFER",
    "amount": 100,
    "oldbalanceOrg": 500,
    "nameOrig": "C_DEMO_1",
    "nameDest": "C_DEMO_2",
}


def validate_local_url(value):
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query or parsed.fragment
    ):
        raise ValueError("Probe requires HTTP 127.0.0.1 without credentials or paths")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid local port") from exc
    if port is None:
        raise ValueError("Explicit local port required")
    return f"http://127.0.0.1:{port}"


def check_preflight(health, recent):
    """Reject unexpected databases so no new event can be inserted accidentally."""
    transactions = recent.get("transactions")
    if (
        health.get("healthy") is not True
        or health.get("committed_events") != 1
        or health.get("last_position") != [1, 0]
        or not isinstance(transactions, list)
        or len(transactions) != 1
    ):
        raise RuntimeError("Not the expected single-event WSGI demo database")
    row = transactions[0]
    if (
        row.get("event_id") != PAYLOAD["event_id"]
        or row.get("step") != PAYLOAD["step"]
        or row.get("source_row_index") != PAYLOAD["source_row_index"]
        or row.get("txn_type") != PAYLOAD["type"]
        or row.get("amount") != PAYLOAD["amount"]
        or row.get("n3") != 0
        or row.get("decision") not in ("ALLOW", "REVIEW")
        or type(row.get("risk")) not in (int, float)
    ):
        raise RuntimeError("The stored demo event does not match expected metadata")
    return row


def validate_duplicate(status, response, reference):
    if (
        status != 200
        or response.get("event_id") != PAYLOAD["event_id"]
        or response.get("duplicate") is not True
        or response.get("decision") != reference["decision"]
        or response.get("n3") != reference["n3"]
        or response.get("risk") != reference["risk"]
    ):
        raise RuntimeError("Inconsistent response for committed duplicate event")


def http_request(url, password, path, body=None):
    headers = {
        "Authorization": "Basic " + base64.b64encode(
            ("fraudguard:" + password).encode("utf-8")
        ).decode("ascii")
    }
    payload = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    opener = build_opener(ProxyHandler({}))
    request = Request(
        url + path, data=payload, headers=headers,
        method="POST" if body is not None else "GET",
    )
    started = time.perf_counter()
    try:
        with opener.open(request, timeout=20) as result:
            return result.status, json.load(result), (time.perf_counter() - started) * 1000
    except HTTPError as exc:
        # HTTP status only; never include Authorization or URLs containing secrets.
        try:
            data = json.load(exc)
        except (ValueError, UnicodeDecodeError):
            data = {}
        return exc.code, data, (time.perf_counter() - started) * 1000


def pct(sorted_values, fraction):
    if not sorted_values:
        raise ValueError("Cannot calculate percentile of empty vector")
    idx = (len(sorted_values) - 1) * fraction
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo)


def probe(url, password, requests=200, concurrency=8):
    if not password:
        raise ValueError("Authentication required")
    if not 1 <= concurrency <= 32 or not 1 <= requests <= 2000:
        raise ValueError("Unsafe probe limits")
    url = validate_local_url(url)

    unauthenticated = http_request_unauthed(url, "/health")
    if unauthenticated != 401:
        raise RuntimeError("Unauthenticated /health must return HTTP 401")
    status, health, _ = http_request(url, password, "/health")
    if status != 200:
        raise RuntimeError(f"Authenticated health request returned {status}")
    status, recent, _ = http_request(
        url, password, "/transactions/recent?limit=1"
    )
    if status != 200:
        raise RuntimeError(f"Could not check recent demo event: HTTP {status}")
    reference = check_preflight(health, recent)

    # A conflicting reuse of a committed event ID must not rewrite its record.
    changed = {**PAYLOAD, "amount": PAYLOAD["amount"] + 1}
    status, _, _ = http_request(url, password, "/transactions/score", changed)
    if status != 409:
        raise RuntimeError(f"Conflicting event ID should return 409, got {status}")

    results = []
    started = time.perf_counter()
    # Requests are independent, but all are intentional retries of the SAME
    # previously committed event. No fresh transactions are submitted.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(http_request, url, password,
                        "/transactions/score", PAYLOAD)
            for _ in range(requests)
        ]
        for future in as_completed(futures):
            status, body, latency = future.result()
            validate_duplicate(status, body, reference)
            results.append(latency)
    elapsed = time.perf_counter() - started
    status, after, _ = http_request(url, password, "/health")
    if (
        status != 200 or after.get("committed_events") != 1
        or after.get("last_position") != [1, 0]
    ):
        raise RuntimeError("Committed transaction count or order changed")

    results.sort()
    return {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "client_architecture": platform.machine(),
        "client_system": platform.system(),
        "target": url,
        "concurrency": concurrency,
        "duplicate_requests": requests,
        "conflicting_event_rejected": True,
        "unauthenticated_health_rejected": True,
        "all_duplicate_responses_consistent": True,
        "committed_events_before": 1,
        "committed_events_after": 1,
        "wall_seconds": round(elapsed, 3),
        "duplicate_responses_per_second": round(requests / elapsed, 2),
        "http_latency_ms": {
            "p50": round(pct(results, .5), 3),
            "p95": round(pct(results, .95), 3),
            "p99": round(pct(results, .99), 3),
        },
        "limitations": (
            "Single-worker localhost Gunicorn; repeated ALREADY-COMMITTED "
            "event only. Measures duplicate-response handling under concurrent "
            "HTTP clients, NOT sustained fresh-transaction ingestion, "
            "concurrent unique-writer semantics, online ML throughput, "
            "external-network performance, TLS, or LinuxONE."
        ),
    }


def http_request_unauthed(url, path):
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(url + path, timeout=10) as response:
            return response.status
    except HTTPError as error:
        return error.code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5050")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--report", type=Path, default=Path("reports/wsgi_duplicate_probe.json")
    )
    args = parser.parse_args()
    password = os.environ.get("FGZ_AUTH_PASSWORD") or getpass.getpass(
        "Server password (hidden): "
    )
    report = probe(args.url, password, args.requests, args.concurrency)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Report saved: {args.report}")


if __name__ == "__main__":
    main()
