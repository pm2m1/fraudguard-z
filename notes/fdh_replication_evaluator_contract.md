# FDH-inspired replication: frozen evaluator candidate (pre-generation)

**Status: evaluator code and toy tests added; NO full generator runs have been created. The run lock and driver do not exist. Do not generate full datasets.**

## Exact chronological contract
- Generator produces 90 daily steps and 3 fixed seeds × 3 regimes.
- Day 0–53: training only (labels available for fitting).
- Day 54–71: unlabeled cutoff setting only, using the top 1% of model scores; tuning fraud labels not used in policy.
- Day 72–89: one-time final-period frozen scoring, no fitting, no threshold changes.
- Strictly original event order with same-day observations recorded after feature extraction. Active past history includes days in the last seven calendar steps, excluding expired observations.
- Static predictors: log amount, sine/cosine transaction clock time, sine/cosine day of week. History adds log counts and monetary sums for customer and terminal and log repeated customer-terminal pair count over the prior seven days.
- Compare EXACTLY two HGB models using identical hyperparameters: max_iter 100, max_leaf_nodes 15, max_bins 128, learning_rate 0.07, l2 10, balanced class weights, random_state 42, early_stopping=False. Do not calibrate model scores.
- Each model sets its own score threshold using the *unlabeled* tuning period. Final decisions use score >= fixed threshold and a cap of floor(0.01 × every prefix of each fixed 10,000-event block). No backfill; no use of future scores, event outcomes, or fraud labels.
- Delayed labels become reportable only seven days after transaction time; no online model/policy adaptation.
- Primary: detected frauds and actual review volume for history HGB versus static HGB across each seed/regime. Also report false alerts, precision, recall, PR-AUC ranking diagnostic, breakdown by fraud scenario and individual day, and within-run paired discordance. Because there are only three seeds per regime, summarize uncertainty descriptively; do not claim independent IID confidence bounds.
- All data outputs excluded from git. Store generator/evaluator SHA-256, runtime dependency versions, dataset hashes, seed and regime in a committed manifest before any full generation. Write one-time per-seed/per-regime run markers to avoid silent reuse. Never use hidden final labels to select alternate seeds, settings or model variants.

## Major scientific caveats to review
- The simplified synthetic generator produces far fewer events than the original Handbook archive; the size and positive prevalence must not be adjusted after peeking at final fraud outcomes.
- No-amount-shortcut regime only removes the direct scenario-1 label; scenario 3 still multiplies amount, introducing an indirect shortcut.
- Final-period-shift regime changes spending and compromise assignment; it may both shift fraud prevalence and make histories more or less useful. This is designed stress testing, not an unseen organic shift.
- Independent seeded runs do not imply independent simulator families.
- The original archived Handbook dataset, one-time BankSim final result and PaySim final holdout remain frozen.

## Next gate
Test \`python -m pytest tests -q\`, review mistakes in unit tests and code, finalize lock and run driver *without generating any full data yet*. Document third-party license obligations and avoid copying GPL notebook implementation unless licensing and distribution terms are compatible.
