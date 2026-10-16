# No-current-amount/balance shortcut audit — frozen exploratory protocol

Status: predeclared after observing the six-candidate validation comparison.
Exploratory sensitivity analysis only; NOT an independent confirmation, test,
model promotion, calibration run, or intervention/streaming policy.

## Fixed motivation and audit evidence

Full-context six-candidate run found transaction-only HGB validation AP
approximately 0.9955, capturing all 760 known frauds at retrospective top 1%.
Adding behavioral or network features had no additional TP at that capacity.
Existing archival `reports/model_c_shortcut_audit.json` showed the
`oldbalanceOrg > 0 and abs(amount-oldbalanceOrg) < 0.01` rule matches
741/760 validation frauds and 1 validation legitimate event. We do not rerun
or mutate that archive. Its labels are PaySim simulator outcomes.

## Fixed feature isolation

Input comes solely from previously materialized, frozen chronological
PaySim development features, `data/processed/full_context_dev_v1`.

Remove ALL features that use the *current* transaction amount or
sender balance, including indirect uses:
- Exclude `log_amount` (index 0),
  `log_sender_pre_balance` (index 6),
  `amount_to_sender_balance` (index 7),
  `sender_balance_zero` (index 8),
  and `sender_amount_vs_hist_mean` (index 12).
- Keep type indicators (indices 1–5).
- Add historical behavioral state (indices 9,10,11,13,14,15).
- Add historical counterparty/network state (indices 16–20).
- Historical sender amount aggregates (index 11) use *prior*
  events and are allowed; the study does not remove all amount-related
  historical structure or simulator artifacts.

Compare six variants using exact original parameters:
- logistic and HGB, each with type-only (5),
  type+behavior (11), and type+behavior+network (16) numeric columns.
- Frozen train rows 0..3177937 only for fit.
- Frozen validation rows 3177938..4263278 only for diagnostics.
- Calibration, policy and step>=373 final holdout MUST NOT be scored.
- Existing HGB or logistic candidate artifacts remain unchanged.

## Metrics and diagnostics

- Validation PR-AUC, label prevalence, top 1% retrospective TP/FP.
- Global-top-1%-selected TP/FP by PaySim transaction type;
  do not independently allocate 1% to each type.
- Candidate feature columns and model SHA256, fit duration.
- Rows and fraud counts for validation types.
- Do not interpret top-k as a deployable streaming threshold.
- Report negative outcomes and exact limitations.

## Output and gating

Use a fresh private local directory
`data/processed/full_context_no_current_shortcut_v1`; refuse overwrite.
Keep model artifacts local, ignored, and never change production Model B,
`full_context_candidates_v1`, archived JSONs or the three untracked
Handbook reports. A future calibrated three-way policy requires a separate
model-freeze decision, calibration, prospective robustness, and risk/cost
criteria. **Zero BLOCK is always an acceptable safety outcome.**
