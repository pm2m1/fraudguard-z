# Phase 6G.2 — prior incoming TRANSFER -> subsequent CASH_OUT link

Status: pre-execution observational protocol. This experiment is a retrospective
diagnostic on previously inspected PaySim **training and validation** rows, not
an independent holdout, trained model, calibration or policy change.

## Scope, hypotheses, boundaries

- Dataset: exact existing PaySim CSV whose frozen split is
  `reports/split_manifest.json`, **only rows 0..4,263,278** (steps 1–306),
  with training at steps 1–238 and validation at steps 239–306. Read exactly
  those rows; do not iterate/read steps 307+, calibration, policy or
  step-373+ final holdout.
- Hypothesis H1: some CASH_OUT originators received one or more prior
  TRANSFERs (where `TRANSFER.nameDest == CASH_OUT.nameOrig`) in a
  preceding rolling six or twenty-four step-hour window.
- Hypothesis H2: this observable link differs between fraudulent versus
  legitimate CASH_OUT events **within each of the two inspected periods**.
  H2 may fail even if H1 is true. Exclude any claim about actual economic
  fund flow or fraud causation: PaySim identity fields are simulation data.
- Never treat the account joining condition as confirmed mule activity.
  Same-step events count as preceding only after earlier CSV-order events.
  At step t, exclude events with step <= t-6 (six-hour) or <= t-24
  (twenty-four-hour). Remove expired ledger entries before querying.
- Track counts of prior incoming TRANSFER to current CASH_OUT originator
  over six/twenty-four step-hours, and age in step-hours since most recent
  incoming TRANSFER within 24 hours. No current event enters its history.
- Distinguish full CASH_OUT population and an *exploratory* subset
  excluding the known PaySim shortcut
  `oldbalanceOrg > 0 and abs(amount - oldbalanceOrg) < 0.01`.
  Also predeclare broad current amount strata: <=100, (100,1000], >1000,
  in dataset-native units. These are diagnostics for confounding, not
  predictors or fitted cutoffs.
- Report per-period counts, frauds, linked cases, fraud rates among linked
  and unlinked CASH_OUTs, and 6h/24h recent count and age summaries.
  Don't report raw identifiers, row-level events, individual scores, or
  any per-account examples. No significance claims, no thresholds/policy
  selection and no automatic BLOCK.

## Implementation and verification

- Implement an independent, bounded sliding transfer-recipient ledger;
  the existing 21-feature engine and production Model B are untouched.
- Streaming CSV with only current row fields; do not load full CSV as a
  dataframe. Validate the full train+validation prefix against the frozen
  manifest counts and chronological boundaries, and refuse overwrite.
- Unit test: same-step read-before-write, expiry at exactly six/twenty-four
  steps, origin/destination direction, current-event exclusion,
  legitimate/fraud aggregation, shortcut subset, train/validation boundary,
  malformed ordering/counts, and no raw identifiers in output.
- On macOS: `python -m pytest tests/test_cashout_incoming_transfer_link.py -q`
  then all tests, then run
  `python -m src.cashout_incoming_transfer_link`.
- Results under ignored `data/processed/` until reviewed. Any archive
  should contain aggregate-only metrics and a clear post-hoc disclaimer.

## Interpretive gate

If prior TRANSFER links are rare, absent, or non-discriminating, reject
this particular hypothesis, do not manufacture a positive result.
If seemingly informative, perform an additional **separately specified**
feature-contract, stress-test and independent-simulator evaluation before
any model/policy replacement. Existing Model C/logistic artifacts, review
cutoff, calibration, all final evidence and LinuxONE gate remain frozen.
