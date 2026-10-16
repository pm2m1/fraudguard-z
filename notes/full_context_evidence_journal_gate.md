# Full-context evidence journal schema v2 — engineering gate

Status: local engineering only, not a model/policy promotion.

The original Model B Flask service and SQLite databases are untouched.
The separate experimental `FullContextJournal` receives **new**
columns to atomically commit observed-only evidence in each decision:

- `evidence_json`: deterministic canonical JSON from
  `full_context_evidence.canonical_evidence_json(pre_event_features)`
- `evidence_sha256`: SHA256 of the exact UTF-8 evidence JSON.

Both fields are written in the **same SQLite transaction** as event,
features, risk, decision and model/policy identifiers, before updating
in-memory history. Return the stored evidence on duplicates without
observing history again. On restart, verify feature replay, canonical
evidence equality and digest for every committed row; reject mismatches.

Bump journal schema ID from `fgz_full_context_v1` to
`fgz_full_context_evidence_v2`. The revised class accepts NEW journal
databases only; it must refuse old v1 journals without migrating or
rewriting their bytes. No changes to the frozen 21 numeric features.
Never store raw account identifiers or labels in evidence.

The checksum is a data-integrity check, **not** an HMAC, tamper-proof
ledger or authentication mechanism: a privileged DB editor could
rewrite both evidence and its digest. SQLite application permissions,
signing/append-only infrastructure and audit retention remain distinct
future controls.

BLOCK remains disabled. Neither policy v1 nor v2 is promoted or
automatically installed in the runtime. This task does not authorize
LinuxONE deployment. Verify idempotency, restart, stored evidence
integrity, nonpromotion and old-schema refusal in tests before API work.
