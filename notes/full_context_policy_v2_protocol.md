# Policy v2 — post-hoc tie-handling sensitivity (NOT confirmatory)

## Why v1 failed
The completed and archived `full_context_policy_logistic_v1/policy.json`
reported 12 eligible/reviewed transactions against a capacity of 5,265,
with only 1/412 frauds flagged. Under `nextafter` tie protection, the
quantile-tied transactions were all excluded. v1 remains immutable.

## Versioned v2 rule
This is a **post-hoc engineering correction**, designed after viewing the
v1 policy results. It is NOT an independent holdout experiment, and no
improvement may be claimed as out-of-sample confirmation.

The underlying `logistic_type_history_network` model and Platt calibrator
remain unchanged and checksum-verified.

1. Take **calibration-only** model predictions from the previously frozen
   rows [4,263,279, 4,759,371). Choose the risk cutoff at the
   99th percentile (upper-tail selection) using the exact NumPy
   order-statistic boundary at `ceil(0.01*n)`.
2. Do **not** use `nextafter` to exclude all boundary ties. Eligible
   events have `risk >= threshold`, including exact ties.
3. In the chronological policy replay, admit an eligible event to
   REVIEW only if cumulative reviews remain strictly below
   `floor(0.01 * events_seen)` AFTER including this event.
   Equal scores are handled by arrival order. No lookahead and no
   random tie-breaking.
4. No backlog, type-specific intervention or use of fraud labels in the
   decision rule. This guard ensures an at-most-1% prefix budget.
5. Evaluate the already-inspected policy period **descriptively** for
   review counts and detected frauds, and label these as repeat-use
   exploratory diagnostics. Never call it fresh validation.
6. Refuse to overwrite v1 or any existing artifacts. Write a new report
   solely in ignored `data/processed/full_context_policy_logistic_v2`.
   No individual transaction data written; only aggregate results.
7. `block_enabled=false`, `block_threshold=null` and
   `blocks=0` unconditionally. No automatic blocking is authorized.

If the calibration-derived cutoff underutilizes the budget on policy rows
or captures few frauds, report that negative result rather than retuning
the cutoff on policy outcomes. The full application still needs
independent robustness, evidence, stateful API/dashboard and local
acceptance before IBM LinuxONE.
