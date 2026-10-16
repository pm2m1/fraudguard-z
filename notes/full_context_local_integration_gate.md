# Phase 6A — isolated full-context localhost integration gate

Status: **engineering-only local simulation**, not operational approval.
This plan is frozen before implementing the local launcher. The previous
119-test and 133-test suites are already verified on the developer Mac.

## Frozen artifacts and non-promotion

- Base model: `logistic_type_history_network`, recorded SHA256
  `e83d413613fc8f0adf3058f932bd910bdf5496c7e7718e97b2ac4c7baffdbdf5`.
- Calibration: original Platt object SHA256
  `cfaaf94597953622318a991d9fbe5dd6af15c8e9430f8d04c02ef4813c64989d`.
- Simulation REVIEW rule: *post-hoc* policy v2, calibration threshold
  0.00528866346387857, inclusive ties, arrival order, prefix review cap
  `floor(0.01 * journal_committed_events)`. SHA256 of the policy JSON
  is recorded as the runtime policy version; verify all provenance,
  threshold/rate/status/block flags and model/calibrator hashes at startup.
- Model receives the selected original columns
  `(1,2,3,4,5,9,10,11,13,14,15,16,17,18,19,20)` from the frozen
  21-column in-memory feature engine, then margin-to-sigmoid calibrated risk.
  No retraining, new model thresholds, dataset access, or hidden policy tuning.
- The run must be explicitly opted into via environment; localhost-only
  bind 127.0.0.1:5051, single process, no production mode, required
  >=16-character HTTP Basic auth, required account pseudonymization key.
- Use a separate `data/runtime/full_context_lab_v2.sqlite3` database.
  Never reuse legacy Model B database or prior experimental journal versions.

## Prefix capacity and durability

A stateless `choose_action(risk)` callback requires the journal's durable
counts. The journal shall retain `reviewed_events` as an in-memory projection
of committed SQLite rows, rebuilt after restart and advanced only after
successful SQLite commit and history observation. The exact journal uses
one owner/lock, and policy is bound to it; the policy computes the
**next** ordinal = committed_events+1 and requires
`reviewed_events < floor(0.01 * next_ordinal)` to REVIEW.
No decisions are made based on labels, retrospective quantiles,
future transactions or type-specific quotas. The chosen event is not
counted before it commits. Duplicates must not consume review slots.

This is a local cold-start experiment: feature history starts empty in
a fresh database. It must not be compared directly with offline PaySim
policy-period outcomes because the latter use all earlier development
events as historical context. A later acceptance phase must replay
the correct historical prefix and test score parity.

## Honest outcomes and testing

The previous v2 retrospective analysis found only 97/412 frauds (23.54%)
and no CASH_OUT detections. The API must declare simulation-only,
not-real-world-risk, no real blocking, and model-attribution unavailable.
Zero BLOCK actions always. No live payments, production, IBM Z, or LinuxONE
claim.

Test source model and calibration checksum tampering, incompatible policy
file/threshold/provenance, missing env keys, production refusal, restart
preserving prefix review budget, duplicate idempotency, real scorer
shape/provenance, and local API response. No changes to old Model B service
or immutable research results. Run the full local suite before launch.
