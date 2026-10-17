# Phase 6G — read-only diagnosis of the CASH_OUT policy blind spot

## Status and provenance

This research branch is based on the archived, completed Phase 6F branch
`feature/full-context-scale-acceptance`. Phase 6F archive commit: `7dfc91f`.
The 10,000 and 25,000 train-prefix engineering results remain unchanged.

The original exploratory v2 PaySim policy outcome was 0/206 CASH_OUT frauds
reviewed, 97/206 TRANSFER frauds reviewed and 5,264/526,517 total transactions
reviewed. The policy had already been inspected and is **not untouched evidence**.
Automatic BLOCK was disabled.

## Exact purpose

Determine whether CASH_OUT is never reviewed because:

1. No CASH_OUT calibrated score reaches the **already frozen global cutoff**, or
2. CASH_OUT events are cutoff eligible but review capacity and chronological
   arrival order prevent review, or
3. Both mechanisms contribute.

The policy has no type-exclusion rule. This diagnostic reproduces the exact
v2 decisions before writing a new, aggregate-only report.

## Strict boundaries

- Only frozen PaySim development matrix calibration rows
  [4,263,279, 4,759,371) and policy rows [4,759,371, 5,285,888).
- Only **policy** labels read; no training, validation, calibration or final
  holdout labels are needed. Final holdout beginning at step 373 is untouched.
- No new fitting, hyperparameters, new calibrator, threshold adjustments,
  alternative quotas, policy promotion, synthetic data or automatic BLOCK.
- The existing model, calibrator and policy artifacts are checked by hashes.
- Event-level scores, identities, originators, destinations, and row indices
  are never persisted. No overwrites.
- Quantiles, eligible counts, eligible true frauds, and cutoff-tie counts
  are descriptive and post-hoc, not statistical inference.

## Verification and commands

```sh
python -m pytest tests/test_full_context_cashout_diagnostic.py -q
python -m pytest tests -q
python -m src.full_context_cashout_diagnostic
```

Read `data/processed/full_context_cashout_diagnostic_v1.json` locally.
If it already exists, the evaluator must refuse overwriting it. Commit
only a reviewed aggregate research summary; do not upload original labels,
raw PaySim data, model binaries, account identifiers or secrets.

## Interpretation gate

If eligibility=0, investigate risk-ranking separation and threshold
mismatch, not simply allocation. If eligibility>0 but reviews=0, inspect
the causal prefix-cap/arrival constraints in a **separate predeclared
engineering exercise** without changing these frozen observations.

Neither outcome validates a new model or authorizes real blocking. A future
replacement policy requires new preregistered data and conservative
failure-cost and false-positive assessment before any deployment decision.
