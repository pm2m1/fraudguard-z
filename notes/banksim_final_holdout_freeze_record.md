# Phase 6N-B — BankSim final holdout freeze record

**Record date:** 2026-10-17  
**Status:** ONE-TIME BANKSIM HOLDOUT CONSUMED — NO RETUNING OR FRESH-TEST CLAIMS  
**Evidence basis:** operator-supplied local terminal output (the assistant has not obtained the local report bytes).

## Scope and immutable references

This is a **separately trained BankSim synthetic fraud classifier**, tested on an
ordered within-BankSim holdout (steps 162–179). It does **not** validate the
frozen PaySim-trained FraudGuard Z model, or show production fraud-detection
accuracy. No PaySim model/policy updates; no automated BLOCK.

Local files — preserve their bytes and maintain a private backup:

| Purpose | Local path | Recorded SHA-256 |
| --- | --- | --- |
| One-time aggregate evaluation | `reports/banksim_native_final_test_v1.json` | `ce5c8738a9448c4e40848ae1f26a184354a2e36a80f40427f9a83373da01a6de` |
| Irreversible attempt marker | `reports/banksim_native_final_test_v1.json.attempt.json` | `de04e8b4d52a949147a943ddf6d266963a1ed60bea018ef3f8761f69d940b1ae` |

The two fingerprints allow later byte-for-byte comparison; they are not
independent attestation of the underlying statistical result.

## Frozen inputs and policy

- BankSim raw source digest: `e37006f76d993bfaec3a02d717b4f0bdc1ebfa5d36449e91fb7c07e225278377`.
- Frozen feature schema digest: `562041c4ae75d8993a48bc431ec0809530bec4da5c1e0c4ee28927afd5ad77c7`.
- Selected model: behavioral logistic regression, 24 BankSim-native features.
- Model digest: `6ca486570adc58ded65ab916e01b68f802604ac0e8cd706aa9c4bd24778b80f1`.
- Calibrator digest: `74fe5994b3e9d09ca5e1cdf7865cc3731077300b7655217fbbeb7dd004388e8b`.
- Threshold fixed during calibration: `0.41746818058367946`.
- REVIEW: fixed threshold with arrival-order prefix 1% capacity. BLOCK: disabled.
- Final test: step range 162–179, 66,972 transactions. This holdout has **already been evaluated once**, and is no longer untouched.
- Prior local regression run: 211/211 passing before the one-time evaluation.

## Results of the single final evaluation

| Measure | Recorded value |
| --- | ---: |
| Total final transactions | 66,972 |
| Fraudulent transactions | 720 |
| Legitimate transactions | 66,252 |
| REVIEW actions | 566 |
| True fraud reviews | 484 |
| Legitimate reviews (false positives) | 82 |
| Missed fraud (ALLOW) | 236 |
| Legitimate ALLOW | 66,170 |
| REVIEW precision | 0.8551236749116607 |
| Fraud review recall | 0.6722222222222223 |
| False-positive rate | 0.0012376984845740506 |
| Average precision (PR-AUC) | 0.8636475166133669 |
| Fraction reviewed | 566 / 66,972 (~0.845%) |
| BLOCK actions | 0 |

### Material limitations

- A constrained review workload misses 236 of 720 simulated frauds.
- Category-level recall varies: `es_fashion` 0/10,
  `es_barsandrestaurants` 1/18, `es_tech` 3/22,
  `es_hyper` 4/28 and `es_wellnessandbeauty` 22/84.
- `es_transportation` contains 56,859 transactions with no fraud labels in
  this final period; some other categories are heavily fraud-enriched.
  Class/category design in synthetic BankSim limits external interpretation.
- A zero observed false-BLOCK count is entirely due to BLOCK being disabled
  and cannot establish safe blocking or a real false-block rate.
- Final labels have been observed. Do not change model, features, calibrator
  or threshold using this result and claim the same BankSim final segment as
  a fresh holdout. A replacement run would be a post-hoc analysis, labeled as such.
- BankSim label and feature semantics differ from PaySim; methodology transfer
  on BankSim does **not** measure out-of-domain generalization of the original
  PaySim classifier. Suitable independently labeled PaySim-compatible data
  remain unavailable/uncertified.
- Public licensing/provenance approval for further dissemination has not
  been established in this record.

## Safe future verification

From the project root, **without reopening labels or running evaluation**:

```sh
shasum -a 256 \
  reports/banksim_native_final_test_v1.json \
  reports/banksim_native_final_test_v1.json.attempt.json
```

Compare both values with this document. Preserve the frozen local model,
calibrator and data snapshots. Do not check raw customer/merchant IDs, BankSim
source CSVs or untrusted joblib binaries into Git.

**Next research dependency:** identify a legitimately independent, licensed,
compatible dataset for the **original PaySim scorer**. That independent
validation remains an open gate. IBM LinuxONE deployment/performance work
and final submission packaging are deliberately paused.
