# FraudGuard Z: archived BankSim audit and forward robustness plan

## Completed one-time evidence — immutable
- Source branch `research/prospective-banksim` contains commit `44971f2` with both original final results and the evaluation-start marker.
- The original source was a separate **synthetic simulator**, not real bank payments or PaySim CASH_OUT.
- The history HGB detected 1,024/1,440 frauds with 1,141 reviews versus static HGB 934/1,440 with 1,201 reviews. These are **observations**, not a transferable model or proof of algorithmic novelty.
- No rerun, threshold adjustment, or retrospective model selection on BankSim steps 144–179 is permitted. Never delete the original run-start marker to circumvent the one-time check.

## Stage A: archive-only consistency audit
Run `python -m src.banksim_integrity_audit` using only committed JSON files. Verify reported totals, precision, recall, per-step review sums, aggregate cap, delayed-label counts and dataset/evaluator identities. This cannot verify the per-10k prefix cap or individual fraud detections because individual decisions and ground truth were not preserved in the report. Do **not** infer a paired significance test from aggregate totals.

## Stage B: separate future-data replication, NOT an extension of BankSim
1. Obtain **independently generated** simulation runs with distinct documented random seeds and controllable fraud regimes. Reusing or resampling BankSim rows, or using its steps 144–179 again, does not create unseen data.
2. Before generating or accessing any replication labels, freeze and commit: simulator version and license, exact seeds, data-generation settings, injected fraud regimes, feature schema, model families/hyperparameters, training and cutoff periods, block size, label delay, eligibility logic, tie rules, baseline and history comparisons, and primary metrics.
3. Use at least three separate run seeds and explicit stable, higher-fraud-prevalence, and distribution-shift regimes if the simulator supports them. Every reported condition must identify its simulator configuration; avoid assuming replicates are independent if they share latent patterns.
4. Preserve a true chronological or generation-order holdout per run, fit models only on training events, tune unlabeled cutoffs on dedicated earlier data, never adapt to the final labels. Evaluate equal review-budget policies **and report actual reviews used**. A predeclared matched-review secondary comparison can be useful but must not use outcomes for threshold selection.
5. Record event-level **pseudonymized** scores and decisions only if licensing and privacy allow; retain per-step actual labels in controlled research storage for paired analysis. Do not publish raw customer or merchant IDs.
6. Quantify variation by independent simulation seed or fraud-regime replicate. Avoid an event-level IID confidence interval when transactions share entities and time dependence.
7. Keep experimental models distinct from frozen production Model B. No automated blocking or production model promotion. IBM LinuxONE work remains deferred.
8. If seedable independent generation is unavailable, halt replication and report this limitation rather than repackaging resampled BankSim data as new evidence.

## What evidence would support advancement?
Consistent improvements in frauds detected at comparable review workload, with uncertainty across genuinely separate runs; robustness to changed category, merchant and transaction amount distributions; transparent failure cases. Even then the result is **synthetic generalization**, not real-world or algorithmic novelty by itself.

## Audit caveat
The integrity audit is post hoc verification of pre-existing report arithmetic. It does not create new independent evaluation evidence.
