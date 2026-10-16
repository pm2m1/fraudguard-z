# FraudGuard Z full-context policy gate — frozen before policy labels

## Status and model contract

**Simulation-only, exploratory policy tuning.** Do NOT overwrite original
Model B, its calibrator, threshold, final test or SQLite. No live payment actions.
The original PaySim final holdout (step >= 373) has been inspected in earlier
work and is NOT an untouched confirmation set.

The frozen base model is `logistic_type_history_network`, using original
21-column indices `(1,2,3,4,5,9,10,11,13,14,15,16,17,18,19,20)`;
Platt calibrator on its decision margins was fitted ONLY on calibration
rows [4,263,279, 4,759,371), yielding slope 1.5872824512785568.
Verify model SHA256 `e83d413613fc8f0adf3058f932bd910bdf5496c7e7718e97b2ac4c7baffdbdf5`,
calibrator SHA256 `cfaaf94597953622318a991d9fbe5dd6af15c8e9430f8d04c02ef4813c64989d`
against the recorded local comparison/calibration metadata and bytes
before any joblib deserialization. The model is an exploratory PaySim
prototype; it is not proven to generalize. N2 is a contextual association,
not a causal fraud indicator.

## Dedicated policy partition

Open only rows [4,759,371, 5,285,888) once under the following protocol:
526,517 events, 412 frauds per the existing frozen split manifest.
Do NOT refit either model or calibrator; do not use final held-out rows.
No manual tuning after inspecting policy results. Use a NEW local ignored
directory `data/processed/full_context_policy_logistic_v1`, never overwrite.

## Frozen REVIEW operating point

1. Predict calibrated risk for every policy row in frozen replay order.
2. Review capacity target = floor(0.01 * policy rows) = **5,265** reviews.
3. Determine one fixed risk threshold at the global policy-period score
   quantile corresponding to that capacity, with a deterministic
   conservative tie guard: require `score >= threshold` and increase the
   threshold with `nextafter(threshold, +inf)` if ties would exceed the cap.
   No type-specific thresholds, score engineering or fraud-label-based
   threshold search. This is **in-sample policy threshold fitting**, not
   independent evaluation and not a prospective calibrated performance test.
4. Simulate a strict causal-prefix capacity guard at decision time:
   for event ordinal `i` from 1 to N, REVIEW only if risk clears the
   frozen threshold AND the total REVIEW count after this event would
   be <= floor(0.01 * i). Otherwise ALLOW. The guard is global across types.
   It may review fewer than 1%, especially for early events or tied scores.
5. Summarize both the threshold-only eligible count and the sequential
   prefix-capped actual outcomes (TP, FP, recall, precision), and a
   type breakdown computed **using the same global decisions**. Include
   total risk mean, observed prevalence, Brier score and AP for the
   policy period; these are on the **policy-tuning set**, not an untouched
   test. Clearly label them exploratory.

## BLOCK safety gate

**BLOCK_DISABLED_UNVALIDATED.** `block_enabled=false`,
`block_threshold=null`, `blocks=0`, `false_blocks=0`.
Do not infer block safety from zero blocks; there has been no block trial.
No automatic BLOCK in journal, API or payments. A future independently
specified block protocol needs an explicit cost model, acceptable upper
confidence bound for legitimate blocks, separate evidence from new
validation, and explicit simulation-only sign-off. Zero BLOCK is a valid
final outcome. No reuse of legacy Model B threshold.

## Persistence and evidence

Store `policy.json` with feature/model/calibration hashes, threshold,
budget, metrics and policy version. Store no raw accounts or row scores.
All runtime use must verify provenance and must be tested separately
for restart/idempotency and audit evidence. This gate does not integrate
with existing Flask service and does not deploy to LinuxONE.
