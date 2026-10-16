# Locked FDH-inspired replication run protocol

**PRE-GENERATION.** The driver has not been run on full settings, and no nine-run final outcomes exist.

## Freeze
- Exactly three independent seeds: 104729, 130363, 155921; exactly three paired regimes: default, no_amount_shortcut, final_period_shift.
- Generator: src/fdh_seeded_generator.py, SimulationConfig defaults, independently seeded substreams. This is a simplified FDH-inspired simulator, not original Handbook execution.
- Evaluator: src/fdh_replication_evaluator.py. HGB static vs HGB history; fixed hyperparameters and features, training days 0–53, tuning days 54–71 (unlabeled score cutoff), final days 72–89.
- Streaming per-model maximum review budget 1% of every prefix within 10,000-event blocks; no backfill. Report actual review counts, detected/missed fraud, false alerts, precision, recall, ranking PR-AUC and fraud scenario breakdown. Labels delayed seven days only for outcome accounting.
- Generator regimes within a seed are paired and correlated. Nine runs do not mean nine independent sources. The no_amount_shortcut regime still includes transaction amount multiplication by scenario 3.
- Results are from synthetic scenarios with imposed fraud rules, not real-world validation, novel algorithm confirmation, PaySim CASH_OUT proof, or grounds for production promotion.

## Procedure (must follow order)
1. Run the full test suite. Review the driver and tests, including no-generation checks for prepare mode.
2. Invoke python -m src.fdh_replication_runner --prepare-lock **once**. It only inspects source bytes, dependencies, protocol and run-plan parameters. No generated transactions or labels are read. Commit and push reports/fdh_replication_lock.json.
3. Invoke python -m src.fdh_replication_runner --check-lock. Require exact committed lock, source, dependency and settings matches; no prior run artifacts.
4. Only after explicit human approval, invoke python -m src.fdh_replication_runner --run once. It creates reports/fdh_replication_started.json BEFORE generation, then records per-run reports and fingerprints; if it fails, preserve the marker and investigate without resetting or re-running silently.
5. Summarize every seed/regime, including failures and actual review workloads. No post-final reranking, threshold selection, changed seeds, or revised claims of untouched evidence.

## Integrity and storage
- Raw independent synthetic transaction rows are not automatically persisted by the driver; fingerprints are computed from canonical serialized event tuples during a run. Record runtime environment in lock. Individual JSON reports include paired outcomes for reproducibility and method inspection. Outputs/markers are ignored by Git until deliberately archived.
- Archived BankSim and Handbook results, frozen Model B and PaySim holdout are unchanged.
- The upstream Fraud Detection Handbook notebook code is GPL-3.0. Our generator implementation was written separately from published scenario concepts; preserve attribution and perform a license review before including source in an external datathon deliverable.
