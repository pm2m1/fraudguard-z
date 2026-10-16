# FraudGuard Z — Complete original P0 before LinuxONE

## Intent and safety boundary

This branch extends the original intended stateful, network-aware, cost-sensitive system while leaving `main`'s frozen Model B, calibration, threshold, SQLite schema and ALL previously inspected final test results unchanged.

Only when the new *full context* feature schema, offline training/serving equivalence, fresh evaluation, three-way policy, evidence codes, audit design and local dashboard all pass their independent gates can the team attempt LinuxONE deployment. This branch is an engineering effort, **not** a retroactive test improvement. Existing PaySim final holdout is already inspected; do not claim any new evaluation on it as untouched. Freeze an independently generated corpus/seed and protocol for future confirmatory work.

## Frozen logical input schema: 17 feature groups, 21 numeric values

T1 `log_amount`; T2 five one-hot transaction types (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER); T3 `log_sender_pre_balance`; T4 `amount_to_sender_balance` (ratio amount/(balance+1) clipped to 1e6, a fixed engineering bound); T5 `sender_balance_zero`.

B1 `log_sender_prior_out_count`; B2 `sender_out_count_6h`; B3 `log_sender_out_amount_6h`; B4 `sender_amount_vs_hist_mean`; B5 `log_pair_prior_count`; B6 `pair_count_24h`; B7 `first_time_recipient`.

N1 `log_sender_distinct_dest_count`; N2 `log_recipient_distinct_source_count`; N3 `recipient_in_count_6h`; N4 `recipient_cashout_count_6h` (past CASH_OUT transactions where current destination was the originator); N5 `sender_current_dest_share_24h`.

Semantics: PaySim `step` is an hourly simulation bucket; prior six-hour events satisfy `prior_step > current_step - 6`; prior 24-hour events satisfy `prior_step > current_step - 24`. Events in the same step count as past only after prior replay order was committed. Identity originator/destination is based on `nameOrig`/`nameDest` across all PaySim transaction types; it does not claim verified economic fund direction. No current event participates in its own features. No label, post-transaction balance, global future aggregation, `isFlaggedFraud` or raw account string enters numeric features.

## Required predeployment gates (ordered)

1. **Feature schema and state:** implement 21-value past-only context in new isolated engine; test same-step ordering, boundary expiry, restart replay, duplicates, exact pair/recipient counts, cardinality and memory behavior. Keep unchanged Model B on main.
2. **Candidate training:** generate full-feature matrices chronologically from exactly the same feature implementation used online. Train logistic and HistGradientBoosting variants; compare feature ablations against transaction-only and behavioral-only. No train/test overlap, no leakage, no arbitrary model advancement. Report all negative results.
3. **Calibration and intervention:** freeze calibrator on separate chronological calibration partition, choose REVIEW threshold from separate policy partition under review budget; choose BLOCK threshold separately with a conservative upper bound on legitimate block errors and a documented cost assumption. **If constraints cannot be met, the validated policy must permit zero BLOCK decisions.** The BLOCK action is always simulation-only. Never reuse Model B review threshold as a block cutoff, or use uncalibrated Model C scores as probabilities.
4. **Evidence/audit and fail-closed:** deterministic observed-evidence reasons with observed values and training-only thresholds. Persist immutable audit linkage, model/feature/policy version hashes and decisions. Prevent duplicate state updates; commit to SQLite before in-memory observation; rebuild only committed events; fail closed on inconsistent or unavailable state. Existing Model B SQLite databases are not migrated in place.
5. **Analyst UI:** show history and network context, calibrated risk, ALLOW/REVIEW/BLOCK, observed evidence and model attribution as separate concepts, model/policy versions, current replay, reasons, cumulative reviews/blocks and local latency; never fabricate risk, intervention or LinuxONE origin.
6. **Local integration and proof:** API, database, restart/replay and dashboard tests, fresh measured system throughput and p50/p95/p99, multi-transaction demo with real data; record constraints, failure behavior, all passing tests. Only after this phase may LinuxONE access/deployment validation begin.

## P2 excluded until P0 verified

Kafka/Redis/Spark/Flink/Neo4j/GNN/LLM, automated threshold feedback, drift monitors, hash-chain audit, full production identity/role systems, external payment blocking and commercial IBM acceleration are not part of current P0. TLS, authorization, retention, secret management and encryption must be verified before exposure outside localhost. Official hackathon pre-kickoff code restrictions and dataset licensing remain unresolved: work must never be falsely represented as developed after the kickoff.

## Acceptance criteria

- Full feature schema length/order and pre-event-only semantics tested.
- Training/serving equivalent on the same committed event prefix.
- Fresh split/fit/calibrate/policy design reproducible and signed off before new holdout.
- No live block without validated conservative threshold and an explicit simulation-only configuration.
- No event can be double counted, processed out of order, or silently allowed on state failure.
- Audit and dashboard display actual data, not placeholders; Model B main remains frozen.
- Evidence documented in version-controlled reports; project eligible under competition rules.
- **Do not deploy to LinuxONE before all selected local P0 criteria pass.**
