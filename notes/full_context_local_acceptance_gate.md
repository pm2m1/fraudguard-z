# Full-context Phase 6C: bounded local chronological acceptance protocol

Status: **engineering acceptance**, NOT new fraud-model validation.
Predeclared before implementing/running this acceptance command.

## Inputs and strict exclusion
- Use only the **first N chronological source rows**, N=300 by default,
  from the locally provided PaySim CSV. N is bounded 2..2,000.
- Require all rows have step <=238 (train partition); reject a row beyond
  step 238. Never read the calibration, policy, or step>=373 final
  holdout. Do not evaluate or use `isFraud`/labels.
- Use a fresh TemporaryDirectory SQLite DB; do NOT reuse a Model B,
  prior experimental, or interactive dashboard DB. Delete acceptance
  DB on normal exit and failure.
- The recorded PaySim source rows are for parity/reliability/latency,
  not an independent candidate evaluation or claims of detection accuracy.
- No per-account identifiers or individual transaction scores in the
  output; only aggregate engineering measurements.
- Load the checksum-validated *real fitted* logistic/Platt/policy artifacts
  through `open_local_lab`; do not use a fabricated scorer in the CLI.

## Fixed acceptance steps
1. Create the localhost **Flask test client in-process**, with real
   Basic authentication and a high-entropy private pseudonymization key
   passed through environment. Do not start an HTTP listener.
2. Verify no-auth request yields 401 and authenticated health, stats,
   and dashboard return 200 with simulation-only labeling.
3. Replay N events in original CSV order and original source-row-index;
   create deterministic synthetic `event_id` based only on the row
   index. Use the raw account fields only inside the process.
4. In parallel compute an independent fresh `FullContextHistory`
   sequential prefix: for each event take `features(event)` before
   `observe(event)`. Verify all 21 response features, fitted scorer
   output to precision, observed evidence, model/policy IDs and
   BLOCK disabled. Never use a future event in history.
5. Restart the SQLite journal+Flask client after `N//2` committed
   events; verify status, review counts, and durable prefix-history
   parity. Replay the last committed event as a duplicate and verify
   it consumes zero review budget. Check evidence lookup.
6. Enforce `reviewed <= floor(.01*committed)` on every prefix; never
   expect at least one REVIEW in a cold-start bounded run.
7. Measure event submission duration with `perf_counter_ns`
   around the **in-process Flask test_client POST** including SQLite
   durable commit. Report p50/p95/p99, mean, min/max, and throughput.
   This is NOT network RTT, model-only inference, or LinuxONE latency.
   Exclude initialization/rebuild from this timing; separately record
   elapsed wall time and journal restart/rebuild time.
8. Verify final stats/health, a nonempty recent-events response, true
   observed evidence lookup, and zero automatic BLOCK.
9. Write a fresh JSON report with `STATUS=LOCAL_ENGINEERING_ACCEPTANCE`,
   exact sample size, aggregate decisions, model/policy versions,
   timing method, and limitations. Refuse overwrite (exclusive create).
   If any invariant fails, raise and do not publish a success report.

## Further gates
- Actual live HTTP/browser review and a larger correctly primed,
  independent history replay are still outstanding.
- No unsanctioned policy tuning, alternate labels, reopened prior test,
  real-world blocking, production mode, or LinuxONE deployment.
- Preserve all archived Policy v1/v2 and Handbook results unchanged.
