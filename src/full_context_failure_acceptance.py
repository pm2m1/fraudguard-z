"""Phase 6I: isolated, in-process fault and recovery acceptance.

Test-only deterministic scorer, disposable SQLite, no real payment routing,
no original PaySim labels or trained models. No network listener.
"""
import argparse
import base64
import json
import sqlite3
import tempfile
import time
from pathlib import Path

from .full_context_api import create_full_context_app
from .full_context_journal import FullContextJournal

OUTPUT = Path("data/processed/full_context_failure_acceptance_v1.json")
PASSWORD = "phase6i-synthetic-test-password-2026"
KEY = "phase6i-synthetic-test-privacy-key-32-bytes"
MODEL = "DETERMINISTIC_TEST_SCORER_NOT_FRAUD_MODEL"
POLICY = "ALLOW_ONLY_TEST_POLICY"


def event(i):
    return {
        "event_id": f"phase6i-{i}", "source_row_index": i - 1,
        "step": i, "type": "TRANSFER", "amount": 20.0,
        "oldbalanceOrg": 100.0,
        "nameOrig": "private-source-fixture",
        "nameDest": "private-destination-fixture",
    }


def open_fixture(path, *, key=KEY, model=MODEL):
    journal = FullContextJournal(
        path, lambda X: 0.01, lambda risk: "ALLOW",
        model_version=model, policy_version=POLICY, privacy_key=key,
    )
    app = create_full_context_app(
        journal, auth_password=PASSWORD, scoring_source="SYNTHETIC_TEST_ONLY"
    )
    auth = {"Authorization": "Basic " + base64.b64encode(
        f"fraudguard:{PASSWORD}".encode()).decode()}
    return app, journal, auth


def must(value, detail):
    if not value:
        raise AssertionError(detail)


def run(output=OUTPUT):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Existing Phase 6I report cannot be overwritten")
    checks = []
    started = time.perf_counter()

    def check(name, fn):
        fn()
        checks.append({"name": name, "passed": True})

    with tempfile.TemporaryDirectory(prefix="fgz_phase6i_") as tmp:
        root = Path(tmp)
        db = root / "normal.sqlite3"
        app, j, auth = open_fixture(db)
        try:
            with app.test_client() as c:
                def auth_gate():
                    for path in ("/", "/stats", "/health", "/transactions/recent",
                                 "/static/full_context_dashboard.js"):
                        must(c.get(path).status_code == 401, "Missing auth accepted")
                    must(c.post("/transactions/score", json=event(1)).status_code == 401,
                         "Anonymous scoring accepted")
                    bad_auth = {"Authorization": "Basic " + base64.b64encode(
                        b"fraudguard:wrong-password").decode()}
                    must(c.get("/health", headers=bad_auth).status_code == 401,
                         "Wrong password accepted")
                check("auth_and_static_asset_protection", auth_gate)

                def origins_headers():
                    r = c.post("/transactions/score", json=event(1),
                               headers={**auth, "Origin": "https://evil.example"})
                    must(r.status_code == 403, "Cross-origin mutation accepted")
                    must(j.status()["committed_events"] == 0, "Rejected mutation committed")
                    r = c.get("/", headers=auth)
                    for field in ("Content-Security-Policy", "X-Frame-Options",
                                  "X-Content-Type-Options", "Referrer-Policy",
                                  "Cache-Control"):
                        must(field in r.headers, f"Missing {field}")
                    must(r.headers["Cache-Control"] == "no-store", "Caching enabled")
                check("origin_protection_and_security_headers", origins_headers)

                def malformed():
                    bodies = ([], {"bad": 1})
                    for payload in bodies:
                        must(c.post("/transactions/score", json=payload,
                                    headers=auth).status_code == 400, "Bad shape accepted")
                    for change in ({"type": "UNKNOWN"}, {"step": 0},
                                   {"amount": -1}, {"oldbalanceOrg": -1},
                                   {"source_row_index": -1}):
                        e = {**event(1), **change}
                        must(c.post("/transactions/score", json=e,
                                    headers=auth).status_code == 400,
                             "Bad transaction accepted")
                    oversize = c.post(
                        "/transactions/score",
                        data=b"x" * (32 * 1024 + 1),
                        content_type="application/json", headers=auth)
                    must(oversize.status_code == 413, "Oversize payload accepted")
                    must(j.status()["committed_events"] == 0, "Rejected input committed")
                check("invalid_and_oversized_input", malformed)

                first = c.post("/transactions/score", json=event(1), headers=auth)
                must(first.status_code == 201, "Initial event did not commit")

                def duplicate_conflicts():
                    again = c.post("/transactions/score", json=event(1), headers=auth)
                    must(again.status_code == 200 and again.json["duplicate"],
                         "Duplicate was not idempotent")
                    must(again.json["evidence"] == first.json["evidence"],
                         "Duplicate evidence changed")
                    conflict = {**event(1), "amount": 30}
                    must(c.post("/transactions/score", json=conflict,
                                headers=auth).status_code == 409,
                         "Conflicting duplicate accepted")
                    earlier = {**event(2), "step": 1, "source_row_index": 0}
                    must(c.post("/transactions/score", json=earlier,
                                headers=auth).status_code == 409,
                         "Out-of-order event accepted")
                    must(j.status()["committed_events"] == 1,
                         "Invalid retry mutated journal")
                check("duplicates_conflicts_order", duplicate_conflicts)

                def history_privacy():
                    r = c.post("/transactions/score", json=event(2), headers=auth)
                    must(r.status_code == 201, "Second chronological event rejected")
                    must(r.json["evidence"]["items"][0]["observed"] == 1,
                         "Past-only context missing")
                    recent = c.get("/transactions/recent", headers=auth)
                    body = recent.get_data(as_text=True)
                    must(recent.status_code == 200, "Recent records unavailable")
                    for private in ("private-source-fixture", "private-destination-fixture"):
                        must(private not in body, "Account ID leaked in response")
                    saved = j.db.execute(
                        "SELECT originator,recipient FROM decisions LIMIT 1"
                    ).fetchone()
                    must(saved["originator"].startswith("acct_v1_") and
                         saved["recipient"].startswith("acct_v1_"),
                         "Unpseudonymized journal identity")
                    must(c.get("/stats", headers=auth).json["blocked"] == 0,
                         "BLOCK invariant violated")
                check("history_evidence_privacy_no_block", history_privacy)
        finally:
            j.close()

        def restart_integrity():
            app2, j2, headers = open_fixture(db)
            try:
                with app2.test_client() as c:
                    health = c.get("/health", headers=headers)
                    must(health.status_code == 200 and
                         health.json["committed_events"] == 2,
                         "Persistence count mismatch")
                    must(c.get("/transactions/phase6i-2/evidence",
                               headers=headers).status_code == 200,
                         "Evidence unavailable after restart")
                    r = c.post("/transactions/score", json=event(2), headers=headers)
                    must(r.status_code == 200 and r.json["duplicate"],
                         "Restart replay not idempotent")
                    must(c.get("/stats", headers=headers).json["total"] == 2,
                         "Restart duplicate committed again")
            finally:
                j2.close()
            for kwargs in ({"key": "another-test-key-of-length-32-bytes!!!!"},
                           {"model": "DIFFERENT_TEST_MODEL"}):
                try:
                    _, bad, _ = open_fixture(db, **kwargs)
                except RuntimeError:
                    pass
                else:
                    bad.close()
                    raise AssertionError("Mismatched identity/model reopened database")
        check("restart_duplicate_evidence_version_key", restart_integrity)

        def evidence_tamper():
            tamper_db = root / "tamper.sqlite3"
            app2, j2, headers = open_fixture(tamper_db)
            try:
                with app2.test_client() as c:
                    must(c.post("/transactions/score", json=event(1),
                                headers=headers).status_code == 201, "Fixture commit failed")
                    j2.db.execute("UPDATE decisions SET evidence_json='{}' WHERE event_id=?",
                                  (event(1)["event_id"],))
                    must(c.get("/transactions/phase6i-1/evidence",
                               headers=headers).status_code == 503,
                         "Tampered evidence leaked")
                    must(c.get("/transactions/recent", headers=headers).status_code == 503,
                         "Tampered evidence accepted in recent list")
            finally:
                j2.close()
            try:
                _, bad, _ = open_fixture(tamper_db)
            except RuntimeError:
                pass
            else:
                bad.close()
                raise AssertionError("Tampered evidence recovered silently")
        check("tamper_detected_on_reads_and_restart", evidence_tamper)

        def postcommit_failure():
            crash_db = root / "crash.sqlite3"
            app2, j2, headers = open_fixture(crash_db)
            try:
                def crash(_):
                    raise RuntimeError("test-only injected post-commit failure")
                j2.history.observe = crash
                with app2.test_client() as c:
                    r = c.post("/transactions/score", json=event(1), headers=headers)
                    must(r.status_code == 503 and r.json.get("error"),
                         "Post-commit failure was not fail-closed")
                    must(not j2.status()["healthy"], "Broken state still healthy")
                    must(j2.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1,
                         "Durable committed transaction was lost")
                    must(c.post("/transactions/score", json=event(2),
                                headers=headers).status_code == 503,
                         "Unhealthy journal accepted new event")
            finally:
                j2.close()
            app3, j3, headers = open_fixture(crash_db)
            try:
                with app3.test_client() as c:
                    must(c.post("/transactions/score", json=event(1),
                                headers=headers).status_code == 200,
                         "Committed event not idempotent after recovery")
                    must(c.post("/transactions/score", json=event(2),
                                headers=headers).status_code == 201,
                         "Recovery refused a subsequent valid event")
                    must(c.get("/stats", headers=headers).json["total"] == 2,
                         "Recovery count mismatch")
            finally:
                j3.close()
        check("postcommit_failure_recovery", postcommit_failure)

    report = {
        "status": "PASS_LOCAL_IN_PROCESS_FAILURE_ACCEPTANCE",
        "scope": "TEMPORARY_SYNTHETIC_TEST_FIXTURES_ONLY",
        "checks_passed": len(checks),
        "checks": checks,
        "wall_seconds": time.perf_counter() - started,
        "blocked": 0,
        "production_validated": False,
        "warning": (
            "Not a concurrent or network/TLS/WSGI/LinuxONE acceptance. "
            "No model accuracy claim. Original PaySim/data artifacts untouched; "
            "existing CASH_OUT detection limitation unresolved."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()
    report = run(args.output)
    print("STATUS:", report["status"])
    for c in report["checks"]:
        print("PASS", c["name"])
    print("CHECKS PASSED:", report["checks_passed"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
