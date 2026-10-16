# Full-context localhost analyst API / dashboard — phase 5B gate

This task adds a **separate** Flask application factory for the
`FullContextJournal` evidence-v2 schema and a new analyst dashboard.
It MUST NOT change `src/api.py`, `src/serve.py`, Model B,
its database, or policy v1/v2 experiment artifacts.

- Flask app factory takes an already-initialized journal with an injected
  scorer/policy. This phase does not silently choose or promote any
  experimental model or review threshold.
- Require a nonempty Basic-auth password (>=16 chars) even for localhost.
  Bind any future dev-server CLI only to 127.0.0.1. Keep responses no-store;
  never expose the app publicly using Flask development server.
- GET `/`: separate clearly labeled *LOCAL SIMULATION ONLY* dashboard.
  Displays latest **recorded** decision data, actual context evidence,
  model/policy IDs, status, count summaries, and client-observed HTTP
  roundtrip duration (never call it model inference latency).
- GET `/health`: journal health, committed rows, versions; unhealthy =>503.
- POST `/transactions/score`: commit event through existing journal,
  never bypass its duplicate, order, idempotency, pseudonymization or
  fail-closed paths; return true stored evidence. 201 new, 200 duplicate,
  400 invalid, 409 event conflict/order, 503 journal unavailable.
- GET `/transactions/recent?limit=...`: bounded 1..100, no raw
  accounts, include observed summary, stored model/policy IDs and risk
  as an **experimental simulation score**. Fail closed on evidence
  checksum mismatch or missing schema, not silently serve tampered records.
- GET `/transactions/<event_id>/evidence`: JSON evidence and
  explicit `model_attribution=false`, checksum-checked, no raw IDs.
- GET `/stats`: real SQLite counts/mean score, blocks=0,
  `block_enabled=false`. Never claim BLOCK capability is operational.
- Always expose the scoring source as a caller-supplied label; tests can
  use synthetic deterministic scorer **only as test fixtures**.
  Do not ship a fake-score launcher as a calibrated production-like demo.
- Authentication and privacy tests must cover all endpoints; include
  restart and duplicate evidence, malformed payloads, unhealthy journal,
  injection-safe rendering, no-store headers and no legacy service changes.

The current v2 exploratory review policy detected 97 of 412 frauds on
previously inspected PaySim policy data, with zero CASH_OUT detections.
It is not operationally validated and MUST NOT automatically power
this API. BLOCK remains disabled throughout. Local acceptance and
an independently specified future evaluation precede LinuxONE.
