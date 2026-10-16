# Isolating the five network features — frozen exploratory diagnostic

This plan was written **after** inspecting the six-model
no-current-amount/balance sensitivity experiment, but **before** running the
specific add-one/drop-one diagnostic below. The previously inspected PaySim
validation partition makes it exploratory, NOT confirmatory evidence.

## Question and fixed design

The no-current-amount/balance HGB experiment captured 80/760 frauds at
retrospective global top 1% without N1–N5, versus 398/760 with all five
network columns. Test whether a single N feature explains the gap.

- Use only existing `data/processed/full_context_dev_v1`
  and frozen split manifest. No new feature generation or raw data read.
- Fit on frozen train rows 0:3,177,938. Diagnose on development validation
  rows 3,177,938:4,263,279.
- Never read, score, fit or tune on calibration, policy, or post-step-372
  final holdout.
- No current-transaction amount/balance or amount-vs-history ratio:
  exclude indices 0, 6, 7, 8, 12.
- Baseline: transaction type one-hots 1:6 plus B1, B2, B3, B5, B6,
  B7 (indices 9,10,11,13,14,15), total 11 columns.
- Network columns: N1 16, N2 17, N3 18, N4 19, N5 20.

For each of logistic regression and HGB, using parameters frozen in
`src/full_context_candidates.py`, fit exactly:
1. Baseline without N features.
2. Five independent add-one variants (baseline plus one N).
3. Five leave-one-out variants (baseline + all N except one).
4. Full network (baseline plus all five N).
This totals **12 variants per family / 24 model fits**.

## Metrics and limitations

Report development validation average precision, global retrospective
top 1% reviewed, TP/FP, recall, precision, and TP/FP split by PaySim
transaction type using the SAME global selection mask. Record model
SHA256 and exact feature names.

These are observational ablations. Overlapping/redundant network features
mean add-one and drop-one effects will not add linearly. Even if N4 drives
all gain, it could encode simulation artifacts, not a durable production
fraud pattern. Global hindsight top-k is not a prospective queue or live
decision threshold. Do not enable BLOCK, claim deployed fraud prevention
or choose any new model based solely on these results.

## Output

Only create a new ignored local directory:
`data/processed/full_context_network_attribution_v1`. Refuse overwrite.
Archive previous six-model comparisons, Handbook reports and Model B
unchanged. At the next gate, decide whether further **prospective**
stress/independent-simulator testing is needed before calibration.
