# Phase 6N — IBM SynDS access determination and AMLNet v2 fallback (2026-10-18)

**Current decision: NO EXTERNAL PAYSim VALIDATION RUN AUTHORIZED.** This investigation used published product/schema/dataset documentation only, not either candidate's row-level labels, and did not change any model, threshold, policy, feature code or evaluated holdout.

## IBM Synthetic Data Sets for Core Banking and Money Laundering

- IBM's public GitHub repository explicitly supplies **schemas**, not a verified freely downloadable full banking transaction CSV.
- Official `v1.1.0/bank_xfers.md` lists `Transaction_Date`, `Transaction_Time`, `From_Account`, `To_Account`, `Amount_Paid`, `From_Initial_Balance`, `Transaction_Type`, `Is_Laundering`, `Is_Instant_Payments_Fraud`, `Is_APP_Fraud`, and other fields.
- Its `In Kaggle` column says **no** for `From_Initial_Balance`, `Transaction_Type`, `Is_Instant_Payments_Fraud`, and `Is_APP_Fraud`, but **yes** for date/time, accounts, amounts, and `Is_Laundering`. Do not infer that another public copy includes these fields.
- The v1.1.0 change documentation mentions per-account transaction CSVs and a chronological bank-transfers CSV, but full chronology, history semantics, permissible category mapping and balance coverage have not been audited against actual files.
- IBM documents Trial, Pro, and Enterprise editions. Its Redbook calls Trial suitable for POCs and says trial copies **must be deleted** at the end of the trial. No guaranteed no-cost student entitlement or public full-download access has been verified here.
- **Confirmed IBM contact:** `aionz@us.ibm.com`. IBM's October 2025 instant-payments fraud blog explicitly offers limited-time trial enquiries through this address. Contact is a proposed user action, not performed here.
- The full schema's distinct fraud labels allow us to ask about more relevant *transaction fraud* targets than its `Is_Laundering` field. However, APP fraud or instant-payments fraud is still not automatically equivalent to PaySim's simulation fraud; no label substitution without a predeclared, justified target.
- The field `Transaction_Time` has much finer resolution than PaySim hourly `step`. A deterministic hourly conversion may be technically feasible, but would still need to preserve event ordering, account roles and replay context. No adapter is approved.

**IBM follow-up gates:** Ask in writing whether the complete Trial dataset includes the above fields and a full chronological stream, whether student participation in the IBM Z Datathon permits research use/public aggregate metrics, whether copying/model evaluation is allowed, and trial-retention/deletion obligations. If access is granted, audit **structure without fraud-label values first**. Do not request or download any data into GitHub.

## AMLNet Version 2.0 — fallback access and incompatibility

- Zenodo DOI: `10.5281/zenodo.21237971`; released July 7, 2026. An openly downloadable 764.6 MB CSV is advertised, with about 1.09M transactions and 1,411 suspicious ones (public dataset-level author summary; **no record-level labels inspected here**). The author states use under **CC BY-NC 4.0**; commercial licensing requires separate permission.
- Published columns include `step`, `type`, `amount`, `nameOrig`, `nameDest`, `oldbalanceOrg`, `newbalanceOrig`, `isFraud`, `isMoneyLaundering`, `laundering_typology`, and timestamp metadata.
- Crucially, source defines `step` as sequential simulation step **or transaction index**, not guaranteed hour. `hour` and timestamp metadata exist but do not establish equivalence to PaySim's 6-hour and 24-hour windows.
- Published types: `BPAY, CASH_OUT, DEBIT, EFTPOS, NPP, OSKO, PAYMENT, TRANSFER`. No documented CASH_IN type and multiple extra types relative to PaySim's frozen five-way one-hot schema.
- Source characterizes fraud labels as suspicious/fraudulent transactions including structuring, layering and integration typologies, an **AML target**, not proved equivalent to PaySim's transfer/cash-out fraud process.
- Numeric nonnegativity, history consistency and actor semantics require an actual label-blind source audit; **no compatibility or final test claim is established**. The generator source is not included in the public release.
- Therefore, AMLNet v2 is **research-download accessible** but **not approved for direct PaySim scorer evaluation**. A label-blind audit is only warranted if there is a plausible target/task alignment; it cannot fix task-label mismatch by renaming columns.

## Decision and low-cost next step

1. **Request IBM Trial access and exact feature/usage confirmation**, not the paid edition by default. Official contact `aionz@us.ibm.com`; IBM indicates one must delete trial copies at trial end.
2. Until IBM replies, do **not** treat AMLNet v2 as a ready replacement. Its license permits noncommercial academic research but dataset-specific task/feature differences are substantial.
3. If IBM does not supply a compatible, authorized source, report the external-generalization gate unresolved. Retain BankSim as a distinct frozen **within-BankSim** synthetic benchmark, not validation of the frozen PaySim model. No data repair or post-hoc model changes.

## Source references

- Official IBM banking schema: https://github.com/IBM/IBM-Synthetic-Data-Sets/blob/main/schemas/Core%20Banking%20and%20Money%20Laundering/v1.1.0/bank_xfers.md
- IBM repository describes schemas-only content: https://github.com/IBM/IBM-Synthetic-Data-Sets
- IBM 2025 Redbook: https://www.redbooks.ibm.com/redpapers/pdfs/redp5748.pdf
- IBM limited-time trial invitation: https://community.ibm.com/community/user/blogs/joy-deng/2025/10/16/instant-payments-fraud-detection-with-synds
- AMLNet v2 fixed dataset and license: https://zenodo.org/records/21237971
- Project baseline: `src/full_context.py`, `src/full_context_shortcut_sensitivity.py`, `notes/paysim_independent_dataset_discovery_2026_10_18.md`

## Unchanged project boundaries

No source labels inspected; no external model inference authorized. Original PaySim scorer, artifact IDs, schema and policy remain frozen; BLOCK remains disabled; BankSim holdout is consumed; MoMTSim two dataset audits remain label-blind; **LinuxONE / IBM hardware benchmarking and final submission packaging remain paused**.
