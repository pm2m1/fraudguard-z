# Phase 6N — MoMTSim frozen-PaySim compatibility decision (label-blind)

**Status:** NOT APPROVED for direct full-population scoring with the frozen PaySim full-context model. This is a **compatibility gate**, not a fraud-performance result.

These facts were observed from user-run label-blind audits. No local MoMTSim
fraud label values were accessed, no MoMTSim model was fitted, and no
PaySim scorer/decision policy was changed.

## Audited source identities

| Data source | SHA-256 | Rows | Step span | Invalid frozen sender input rows |
| --- | --- | ---: | --- | ---: |
| MoMTSim_20240722202413_1000_dataset.csv | 99fd07c3a9d3c4bd6d3462240058ca19d0d9e9284683f78bf77542ff7fcc05e7 | 4,225,958 | 0..192 | 42,574 |
| synthetic_mobile_money_transaction_dataset.csv | da951eb95735da96271740a3e66b676b342d3831ce3111cd19dbfa020d3bd0a7 | 1,720,181 | 0..143 | 44,402 |

- All source amounts parsed as nonnegative finite numbers in both versions.
- Each invalid frozen sender input was a **negative** `oldBalInitiator`.
  The source may legitimately simulate overdrafts, so this is an incompatibility
  with the frozen **nonnegative** PaySim input contract, **not proof of corruption**.
- Large source: 43,222 rows with any invalid numeric under the audit's
  nonnegative rule; only 647 rows had *solely* post-transaction invalidity.
  Smaller source: 48,213 rows with any invalid numeric, only 3,697 were
  invalid solely in post-transaction balances.
- Both chronological order checks passed with no blank account IDs.
- Both include step 0. The PaySim `FullContextHistory` requires step >= 1.
  A uniform +1 offset could retain relative windows, but only if applied
  after source time-unit validation and declared **before** label access.
- MoMTSim has DEPOSIT / WITHDRAWAL / PAYMENT / TRANSFER / DEBIT. PaySim
  has CASH_IN / CASH_OUT / PAYMENT / TRANSFER / DEBIT.
- Dataset authors define DEPOSIT as loading funds into a mobile wallet,
  WITHDRAWAL as removing funds, TRANSFER as user-to-user funds transfer,
  PAYMENT as purchasing goods/services and DEBIT as moving wallet money
  into a bank account. This makes DEPOSIT -> CASH_IN and
  WITHDRAWAL -> CASH_OUT **plausible hypotheses**, not verified equivalents
  of PaySim's exact participant roles or fraud-label semantics.
- Large source PAYMENTS with unchanged sender balance among valid
  positive-amount rows: 773,298; smaller source: 25,036.
  Such events can have source-specific outcome/recording semantics;
  treating them as completed PaySim payments is not yet justified.
- Source authors' simulation includes overdraft-limit settings, a
  possible explanation for negative balances. Do not clamp, impute or
  treat them as corrupt automatically.

## Why this is not yet an external validation

The frozen PaySim scorer uses 21 features, including strict past-history
counts, window sums, transaction type indicators, pre-balance-derived values
and receiver-specific CASH_OUT counts. Problems:

1. Negative `oldBalInitiator` fails `validate_event`. Dropping those
   rows would also remove their earlier interactions from stateful history.
   Updating history while refusing to score requires a separate, explicitly
   tested strategy and cannot silently change the frozen feature engine.
2. Unsupported transaction types cannot be passed through without
   documented mapping and participant-role validation. Dropping DEPOSIT
   or WITHDRAWAL also corrupts later historical features.
3. Simulators differ in fraud generation, actions, customer populations,
   amount units, balances, and event-outcome semantics. Published synthetic
   MoMTSim fraud results cannot be equated with PaySim fraud performance.
4. Step alignment requires a documented consistent transformation and
   test of all 6- and 24-step window boundaries.

**Decision:** no external model scoring or fraud-label reading yet.
No retuning, retraining or threshold adjustments from MoMTSim outcomes.

## Follow-up compatibility gates

1. Document transaction-type and actor-role invariants for deposit,
   withdrawal, payment and debit, using the source simulator specification.
2. Predeclare negative-balance handling and the valid inference population,
   including how all source events contribute to historical state.
3. Test streaming feature equivalence and time-window boundaries on
   synthetic fixtures. Validate privacy and file/provenance hashes.
4. Only if a defensible frozen adapter exists, pre-register an evaluation
   that makes all coverage exclusions explicit, with a separate label-only
   access gate. Otherwise report external validation as **not established**.

## References

- Dataset: https://data.mendeley.com/datasets/zhj366m53p/2 (CC BY 4.0)
- Dataset description and transaction roles: https://pmc.ncbi.nlm.nih.gov/articles/PMC12036017/
- PaySim field semantics and synthetic transaction context:
  https://www.kaggle.com/datasets/ealaxi/paysim1
- Local source code: `src/full_context.py`, `src/features.py`,
  `src/momtsim_compatibility_audit.py`, `src/momtsim_field_diagnostic.py`.

Original PaySim model and policy remain unchanged. BankSim holdout remains
consumed and frozen. BLOCK, LinuxONE work and final submission remain paused.
