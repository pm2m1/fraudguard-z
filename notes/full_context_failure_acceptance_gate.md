# Phase 6I — In-process failure and recovery acceptance (pre-LinuxONE)

This gate follows local dashboard/manual restart checks of Phase 6H. It is
**engineering-only**, not a fraud model, policy, financial security certification,
network benchmark, concurrency test or LinuxONE deployment gate.

## Isolation and invariants
- No original PaySim CSV, labels, held-out partitions or model binaries.
- Inject a clearly marked deterministic **test-only** scorer/policy, never
  a model presented as calibrated. Use only exclusively created temporary
  SQLite databases; all corruptions and injected failures are confined to them.
- Keep the actual 21-feature contract and evidence journal code under test.
- Authentication remains mandatory; BLOCK disabled; recorded data must not
  leak raw sender/destination account IDs.
- Never re-use the user's manually verified dashboard database. The runner
  starts no network server, is not a production attack tool and stores only
  aggregate check names/results, not raw IDs or credentials.
- Do not overwrite existing reports or previously archived scale/CASH_OUT results.

## Predeclared checks
1. No-auth and wrong-auth requests receive 401, and static assets require auth.
2. Same-origin authenticated transaction accepted; supplied cross-origin
   Origin rejected without state changes. Verify security headers/no-store.
3. Malformed, out-of-range, invalid-type and oversized HTTP bodies rejected;
   no rejected input commits.
4. Identical duplicate receives HTTP 200, immutable decision/evidence, and
   unchanged counts. Conflicting duplicate and out-of-order events return 409.
5. Two chronological events preserve prior-activity observations and yield no
   raw account IDs in responses or SQLite account columns.
6. Full process-level-equivalent journal close/reopen: same key/model/policy
   restores exactly the journal and evidence; duplicate replay after restart
   is idempotent. Wrong key and changed model/policy refused.
7. Evidence hash tamper in **disposable** DB yields 503 on evidence reads;
   reconstruction refuses corrupted journal. No file repair attempted.
8. Inject post-commit history.observe failure. HTTP request must return 503
   with no unearned ALLOW. Durable row survives, and reopen/rebuild restores
   idempotency, history and the ability to commit a later valid event.
9. Unhealthy journal returns 503 and does not accept new decisions.

## Remaining limitations
- A live journal does not automatically verify all historical SQLite rows on
  each new POST; direct local database tampering may be detected only during
  evidence reads or full restart. This is a current threat-model limitation,
  not a proved global tamper-evident audit chain.
- All tests are in-process Flask client calls, no network TLS/rate limiting,
  WSGI, proxy/host allowlisting, RBAC, concurrent workers, operator accounts,
  network outage injection or IBM LinuxONE. Public-facing deployment is not
  approved. Synthetic Basic credentials are **fixtures**, never secrets.
- An exploratory policy that caught 0/206 CASH_OUT frauds remains unfit for
  real-world transaction blocking.
