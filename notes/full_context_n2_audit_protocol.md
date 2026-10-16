# FraudGuard Z — N2 concentration diagnostic (frozen before this audit)

## Status
EXPLORATORY VALIDATION AUDIT ONLY. Written after observing the network
add-one/drop-one results (baseline HGB 80/760 frauds and HGB+N2 398/760
at retrospective global top 1%, with full network also 398/760).
The original validation period has already been inspected. This audit does
not create an independent test or establish causal model reliance.

## Predeclared question

Does N2, `log_recipient_distinct_source_count` (numeric column 17),
concentrate fraud-labeled events into a small subset of historical-recipient
connectivity patterns? Does that concentration vary by PaySim transaction
type and by chronological development period? N2 counts unique prior
originators over the **entire observed prefix**, NOT a recent 6h window.
One transaction cannot modify its own N2 value.

Fixed N2 group definitions:
- 0 historical distinct senders
- 1 historical distinct sender
- 2–3 historical distinct senders
- 4+ historical distinct senders

Within frozen train steps 1–238 (rows 0:3,177,938) and frozen validation
steps 239–306 (rows 3,177,938:4,263,279), report counts, frauds, fraud
rates, fraction of all frauds, for all transactions and each of the five
PaySim transaction types. Separately report four consecutive equal-row-count
quarters of the validation partition by their fixed input row positions.
Do not compare raw AP between different prevalence regimes.

## Constraints

- Read ONLY `data/processed/full_context_dev_v1`, the immutable feature
  schema and frozen manifest. Do not open the raw CSV or stored account IDs.
- No model loading or fitting; no calibration or thresholds.
- Do NOT read any labels/features from the calibration or policy partitions.
  Do NOT read PaySim step>=373 or the original final holdout.
- Output only anonymized aggregate counts, no event identifiers or raw IDs.
- Do not apply N2 bin thresholds as a new fraud intervention policy.
- Report zero-count groups as null fraud rates.
- No change to Model B, candidate models, Handbook archives or audit reports.
- Write only to a NEW ignored file under
  `data/processed/full_context_n2_audit_v1.json`; refuse overwrite.
- This audit is the final focused diagnostic planned before returning to
  engineering gates: model freeze, calibration, safe simulated intervention,
  deterministic reason codes, API, dashboard and end-to-end acceptance.
