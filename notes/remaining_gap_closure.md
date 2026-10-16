# Remaining gap closure: execution plan and evidence gates

Baseline: main commit d32f585, 66 passing tests, frozen production Model B.
This branch is experimental and must not be merged without a passing test run.

## 3. Bounded streaming and recovery (first deliverable)

`src/stream_ingest.py` adds a bounded producer/consumer queue, ordered single-consumer scoring, idempotent retry of transient local HTTP errors, and conservative prefix-resume checks. Unlike `src/replay.py`, it can continue a partial PaySim-only stream after restarting with the **same database and CSV prefix**, provided the stored last event ID, position, and count match. This is a local prototype, not a distributed broker, multi-client transaction system, or exactly-once guarantee.

Commands (run from repository root, model files present):

```bash
python -m pytest tests -q
# Terminal 1: protected single-owner local service with a NEW filename
FGZ_DB_PATH=data/runtime/stream_demo_v1.sqlite3 python -m src.serve
# Terminal 2, same authenticated environment:
python -m src.stream_ingest --count 5000 --queue-size 32 --report reports/stream_5000.json
```

On a deliberately stopped partial stream, restart the server with the **same** database path and original privacy key, then run `python -m src.stream_ingest --count 5000 --queue-size 32 --resume`. Don't run the producer concurrently from two terminals. Before interpreting an output report, verify matching stats and transaction history. Backpressure here is via bounded queue, not a durable queue. The worker uses a daemon thread; if the client process crashes while the queue contains events, those not committed must be replayed from the original CSV. The SQL database remains the authoritative checkpoint.

## 2 and 4. Model quality and technical contribution

Model C's 0.9917 validation PR-AUC is largely explained by the synthetic exact-sender-balance shortcut. Do not promote it, report it as real-world accuracy, or rerun the already inspected holdout to justify an improvement. Prefer an explicit original systems contribution: recovery-safe and auditable, calibrated fraud review on IBM Z. Any stronger ML model must beat baselines on an **independent** dataset with comparable feature availability, fraud prevalence and chronological setup. Report type-specific recall, false positives and review budgets. If network features do not improve quality, say so.

## 5. Portability and independent evidence

- Identify a legitimate third-party or separately generated transaction benchmark, check licensing, and adapt its fields without leaking features.
- Capture a reproducible dependency lock for each architecture.
- Provision a real s390x LinuxONE system, transfer the private repo and model binaries securely, verify checksum, dependency compatibility, equivalent sample scores, tests, and p50/p95/p99 latencies there.
- Model artifacts loaded with joblib are trusted-code artifacts; never load files from unverified sources.

## 6. Deployment controls

Authentication and HMAC pseudonymization are implemented for local demos; production TLS, encrypted storage/backups, secret management, log redaction, and authorization policies are not yet verified. `src/serve.py` intentionally refuses production mode. Build and validate a single-worker authenticated TLS-protected WSGI service before external access.

## Evidence gate

Do not label any gap **closed** until test reports, runtime measurements and deployment/security checks are recorded. Keep raw PaySim data, private keys, live SQLite files, and model binaries out of Git.
