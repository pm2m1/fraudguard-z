# Phase 6N-B: BankSim-native external-domain benchmark (predeclared)

## Scope
A separately trained BankSim model can test our *methodology* on a second synthetic data domain. It **cannot** independently validate the frozen PaySim-trained logistic model, its 21-column feature schema, its calibration, its REVIEW policy, or CASH_OUT detection. No mixing, merging, retraining, or updating any PaySim model/policy/artifact.

Input: local `data/external/banksim/bs140513_032310.csv`; reference SHA-256
`e37006f76d993bfaec3a02d717b4f0bdc1ebfa5d36449e91fb7c07e225278377`
(594643 rows, 180 steps 0..179 from **label-blind audit**). Provenance/licensing must be manually reviewed before public use. Never commit raw records, customer/merchant IDs or model binaries.

## Frozen temporal split by step (not random split)

- TRAIN: steps 0..107 inclusive
- VALIDATION: 108..125
- CALIBRATION: 126..143
- POLICY DIAGNOSTIC: 144..161
- FINAL TEST: 162..179

The FINAL TEST label is not accessed during preparation, candidate selection,
or calibration. Run the final-test command only once after the candidate/model,
calibrator, fixed review threshold and policy are frozen. Never tune any of
those using policy or final-test outcomes. Once inspected, the holdout ceases
to be untouched. An immutable report file is refused if it exists already.

## Fixed numeric schema

A transaction-only baseline: `log1p(amount)` and one-hot encodings for the
15 exact BankSim category literals observed in the unlabeled audit. No
gender, age, zipcode, user/merchant ID numeric encoding or fraud label.

Extended model adds strict pre-event counts and statistics:
`log1p(customer_prior_count)`, `log1p(merchant_prior_count)`,
`log1p(customer_merchant_prior_count)`,
`log1p(customer_prior_mean_amount)`,
`log1p(merchant_prior_mean_amount)`,
`log1p(customer_recent_7_step_count)`,
`log1p(merchant_recent_7_step_count)` and
`log1p(pair_recent_7_step_count)`.
Same-step earlier rows count; recent step window uses
`prior_step > current_step - 7`; no future and no current-row self-count.
BankSim steps are dataset-relative units, not assumed PaySim hours.
Customer-to-merchant edges are *not* account-to-account money transfers.
No label, future balance, post-payment information or model feedback
enters any predictor.

## Frozen fitting and scoring

Train two logistic StandardScaler pipelines with `class_weight=balanced`,
`solver=lbfgs`, `max_iter=300`, `C=1`, `random_state=2026`.
Select larger validation average precision; exact ties select the
transaction-only baseline. Fit one Platt sigmoid on CALIBRATION decision
margins, requiring both labels present and a positive slope. Take the
fixed global REVIEW cutoff from the top 1% of calibration probabilities
and apply an arrival-order prefix 1% capacity guard. No BLOCK decisions.

Record validation/model/calibration metadata and artifact SHA256 and do
not refit. The final test reports frozen-policy precision, recall,
false-positive rate, PR-AUC, by-category results, and limitation notices.
This is a new BankSim model evaluated on BankSim, **not** external
validation of the prior PaySim model. All results are synthetic-only.

## Execution gates

1. `prepare`: header, SHA, order, feature generation and non-test labels;
   write only into an absent ignored output directory.
2. `fit`: fit only train, select only validation, calibrate only calibration,
   compute non-label threshold; freeze metadata and hashes.
3. `final-test`: explicit once-only final-label access after successful
   gate, with nonoverwrite and frozen-artifact verification.
4. Tests: same-step history, seven-step expiration, shape/order, no
   pretest-label access, no mixed PaySim artifacts, reproducible decisions,
   stage refusal on overwrite/invalid provenance.

## Conditions for claims
An untouched BankSim test measures within-BankSim generalization at a
chronological boundary. It is a *separate external-domain benchmark
exercise*; no "real-world fraud accuracy", LinuxONE performance,
fraud BLOCK safety, or PaySim-model generalization claims.
