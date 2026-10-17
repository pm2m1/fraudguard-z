# Phase 6N — Independent PaySim-scorer dataset discovery (18 Oct 2026)

**Research result:** No publicly accessible, independently generated dataset
identified in this screening is **verified as a drop-in external validation**
for the frozen 21-feature PaySim full-context scorer. Do not score, access
candidate fraud labels, tune, refit, or modify frozen artifacts as a shortcut.

## Required frozen-model contract

`src/full_context.py` requires events in source order with hourly-like steps
starting at >=1, exact types CASH_IN/CASH_OUT/DEBIT/PAYMENT/TRANSFER,
persistent sender and recipient identities, nonnegative finite amount and
`oldbalanceOrg`, and deterministic read-before-write account history with
6- and 24-step windows. The currently selected logistic model uses **16 of
21** features and excludes the current amount/balance shortcut group
(indices 0, 6, 7, 8, 12); nonetheless the unmodified feature engine
validates the entire input event, so field semantics still matter.

Fraud labels must refer to transaction fraud of a sufficiently comparable
kind, not merely suspicious laundering or a risk score. Check copyright,
download/use permission, generator provenance and potential PaySim row
overlap before considering any independently labeled test.

## Candidate ranking (a research shortlist, not a list of approved datasets)

| Source | Verifiable strengths | Blocking issue | Verdict |
| --- | --- | --- | --- |
| IBM Synthetic Data Sets — Core Banking and Money Laundering | Official schema includes transaction time, account pair, amount, optional `From_Initial_Balance`, and several distinct fraud indicators (`Is_Instant_Payments_Fraud`, `Is_APP_Fraud`, etc.) | Public Kaggle variant **does not include** initial-balance and some fraud fields according to official schema; full product access/usage rights unverified; types differ from PaySim | **Highest-value conditional inquiry**, not compatible yet |
| AMLNet v2 (Zenodo DOI 10.5281/zenodo.21237971; July 2026) | Open download; ~1.09M records; independent AMLNet framework; `step`, `type`, `nameOrig`, `nameDest`, `oldbalanceOrg`, `isFraud`; CC BY-NC 4.0 | `step` defined as step **or transaction index**, not verified hourly; extra BPAY/EFTPOS/NPP/OSKO categories, absent CASH_IN; `isFraud` is suspicious/fraudulent under laundering typologies rather than PaySim account-draining fraud; nonnegative balance coverage not checked; 764.6 MB file | **Best open schema lead, but fail-closed until semantics resolved** |
| MoMTSim two uploaded CSVs | Independently generated mobile-money logs, hourly steps, persistent IDs, balances and labels; CC BY 4.0 publication | Existing label-blind audit: 42,574 / 44,402 negative sender balances, differences in DEPOSIT/WITHDRAWAL roles, many history-contamination flags, step zero | **No direct scoring** (see frozen compatibility audit) |
| Tide (2026) | Open-source independent temporal/graph AML generator, controlled typologies and reproducibility | Dataset oriented to transaction graph/AML labels; verified PaySim-compatible pre-transaction balances and five transaction types absent | Separate AML methodology, not frozen-scorer test |
| IBM AMLSim and standard AML synthetic datasets | Account-to-account graph and labeled suspicious activity | Often AML labels, not account-takeover transaction fraud; no verified PaySim-matching pre-event balance/type contract | No direct scoring |
| BankSim | Separate synthetic transaction-fraud model evaluated with frozen BankSim-native chronological protocol | Missing PaySim pre-sender balance and categories; within-BankSim final holdout already consumed | **Separate methodology study complete** |
| CPE232 Kaggle competition / Hugging Face PaySim mirrors / Kaggle `Fraud.csv` | Near-identical field names, same five type values, 744-step PaySim description | Clearly PaySim-format redistributions or plausible renamed versions; row/provenance independence unproven | **Reject as external independence unless independent origin demonstrated** |
| ABDcodes76 `Financial-Fraud-Detection` 60k generated transactions | Published independent Python code, PaySim-like five transaction types, sender balance, labeled fraud | Source `dataset/generate_dataset.py` randomly picks fraud examples in TRANSFER/CASH_OUT, invents random sender IDs per row, makes fraud largely an account-draining shortcut and does not maintain durable account-history balances; no evidence of realistic history | **Not suitable for credible external historical generalization** |
| NeurIPS Bank Account Fraud (BAF) and IEEE-CIS | Established labeled fraud benchmarks | Account-opening/application or anonymized payment schema; no full chronological sender/recipient histories and initial balances | Separate benchmark only |

## Prioritized next action (before opening any new labels)

1. **Access inquiry, not model evaluation:** Check with IBM whether a
   non-commercial/student-evaluation sample from the *full* Core Banking
   dataset includes `From_Initial_Balance`, account identities, ordered
   timestamps and transaction-specific fraud flags, and whether use in the
   Datathon is permitted. Public schema alone does not prove sample access.
2. **Open public candidate audit, only if needed:** Inspect AMLNet v2
   *schema and unlabeled source metadata only*. Verify whether `step`
   encodes hour vs transaction index, permitted transaction-type domain,
   negative balances, actor roles, and whether `isFraud` means a payment
   fraud outcome distinct from suspicious money laundering. Do not look
   at fraud values or publish calibration/score results.
3. **Strict stop condition:** If source semantics, coverage, independent
   provenance or license cannot be established, leave Phase 6N independent
   **PaySim-scorer** validation OPEN and explicitly say it was not feasible
   with publicly available verified data. No invented fields, row dropping,
   label substitution, rescaling, score threshold changes or refitting.

## Primary links

- IBM official schema: https://github.com/IBM/IBM-Synthetic-Data-Sets/blob/main/schemas/Core%20Banking%20and%20Money%20Laundering/v1.1.0/bank_xfers.md
- IBM product: https://www.ibm.com/products/synthetic-data-sets
- AMLNet v2: https://zenodo.org/records/21237971
- MoMTSim publication: https://pmc.ncbi.nlm.nih.gov/articles/PMC12036017/
- Tide generator: https://github.com/mntijn/Tide
- IBM AMLSim: https://github.com/IBM/AMLSim
- PaySim original: https://www.kaggle.com/datasets/ealaxi/paysim1
- Example PaySim mirror: https://huggingface.co/datasets/LordNR/AMLGraphX-Paysim
- Example independent-looking but deficient generator:
  https://github.com/ABDcodes76/Financial-Fraud-Detection/blob/main/dataset/generate_dataset.py
- BAF official study: https://proceedings.neurips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html

## Preservation

Original full-context PaySim model/feature schema/review policy are unchanged;
BLOCK stays disabled. BankSim final holdout and report remain frozen, and
MoMTSim fraud labels remain sealed. LinuxONE, hardware benchmarking and
final submission packaging remain deliberately paused.
