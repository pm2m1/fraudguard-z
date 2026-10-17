# Phase 6N — MoMTSim lineage-compatibility decision (label blind)

**Decision:** NO-GO for direct independent validation of the frozen PaySim 21-feature scorer on either unmodified MoMTSim CSV. This decision may be reconsidered only on documented *semantic* evidence and a predeclared, tested adapter. No scoring, fraud-label inspection or PaySim model changes were performed.

## Reproducibility and aggregate findings

Results from the user's local, read-only audit
(`src/momtsim_lineage_coverage_audit.py`; local test suite 218/218 passing).
This record does not contain individual customer IDs or transactions.

| Input | SHA-256 | Rows | Direct invalid sender input | Later flagged by propagation | No known invalid lineage | Fraction no known invalid lineage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MoMTSim_20240722202413_1000_dataset.csv | `99fd07c3a9d3c4bd6d3462240058ca19d0d9e9284683f78bf77542ff7fcc05e7` | 4,225,958 | 42,574 | 2,703,875 | 1,479,509 | 35.0100% |
| synthetic_mobile_money_transaction_dataset.csv | `da951eb95735da96271740a3e66b676b342d3831ce3111cd19dbfa020d3bd0a7` | 1,720,181 | 44,402 | 1,664,295 | 11,484 | 0.6676% |

Direct invalidity in the existing PaySim event-validator contract is due to
negative `oldBalInitiator` values, despite nonnegative finite amounts.
Negative balances **may reflect legitimate simulation overdraft settings**;
the audit is not a finding of corruption or misconduct.

Both MoMTSim files also contain step 0 and the categories `DEPOSIT` and
`WITHDRAWAL`, whereas the frozen PaySim feature engine requires step >= 1
and recognizes `CASH_IN` and `CASH_OUT`.

### Important limitation: taint does not equal mathematical ineligibility

The lineage script marks *both sender and recipient* following an
unrepresentable sender-balance event, and it propagates that marker to
future transaction counterparts **without expiry**.

This is a **conservative connected-history screen**, not a feature-level
causal analysis. For example, the PaySim 21-feature engine uses some sender
histories, some recipient histories and windows of 6 or 24 steps; different
components are affected differently by an omitted event. The counts headed
"history_affected_after_invalid_event" are therefore **upper-bound flags
under the script's heuristic**, not demonstrated counts of transactions
whose 21 frozen input features would be wrong.

The reported "no known invalid lineage" fractions are not approved model
coverage estimates. They ignore still-unresolved differences in transaction
semantics, source account roles, step offsets, fraud definitions, and stateful
consequences of attempted / no-balance-change transactions.

### Why filtering is not an independent evaluation

Skipping negative-balance events discards those events from the stateful
replay. Later counts, history ratios and recipient/cash-out windows would be
computed over an altered population, and fraud outcomes of excluded rows
would not be represented. Treating negatives as zero or using the source
simulator's signed balance directly would change the frozen input contract
or feature distribution. Neither has been authorized.

Even though DEPOSIT/WITHDRAWAL may describe analogous actions to
CASH_IN/CASH_OUT, exact role and bookkeeping equivalence remains unproven.
A blind renaming would manufacture a new interpretation and cannot
establish external validation of the frozen original model.

**Do not score these sources under the current frozen full-context schema,
read their fraud labels to tune an adapter, or claim performance.**

## Research decision and next defensible alternatives

1. Seek a genuinely independent labeled event stream with the *full* PaySim
   feature semantics, nonnegative pre-sender balances and stable ordered
   account identities. Freeze its source and eligibility criteria **before**
   fraud labels are inspected. Verify that no source overlap exists.
2. A clearly **separate** MoMTSim-native benchmark could be implemented
   if useful, but it would need distinct training, labels/evaluation and
   honest synthetic-domain claims. BankSim already provides a separately
   completed synthetic benchmark; another such exercise is not a substitute
   for testing the original PaySim scorer.
3. If future source documentation justifies a lossless adapter, require
   comprehensive read-before-write history reconstruction, actor/type
   invariants, signed-balance handling, chronology and population-coverage
   acceptance on synthetic fixtures before any labels are opened.

## Stable scope

- Original PaySim model and policy: **unchanged**.
- MoMTSim fraud-label values: **not accessed by these audits**.
- BankSim final holdout: **consumed and frozen**, no repeat fresh claim.
- Automatic BLOCK: **disabled**.
- LinuxONE, IBM hardware benchmarks and final submission: **paused**.

Phase 6N's original independent-PaySim validation gate remains **OPEN**.
