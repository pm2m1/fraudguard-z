# Phase 6K-6N (pre-LinuxONE): four bounded completion gates

Baseline: `feature/full-context-failure-acceptance` at commit
`0bdac5d9ebb1a81608b52944ec760c97408de116`.
All changes are isolated on `feature/prelinuxone-four-gates`.
Do not merge/promote without a fresh local test run.

## 6K — three-action **shadow** intervention

- Test `ALLOW/REVIEW/BLOCK` on scores in a **separate in-memory,
  no-side-effects** evaluator. Explicitly label BLOCK as
  `WOULD_BLOCK_SIMULATION_ONLY`; never return executable payment instructions.
- Hard-code no production flag. Do not change the frozen v2 threshold,
  observed PaySim outcomes, `FullContextJournal` DB `CHECK`, or the
  journal's fail-closed guard preventing real BLOCK.
- For a prospective BLOCK safety claim, require separately approved
  model/calibration and independent fixed-policy evidence. A 95% *upper*
  false-block-rate confidence bound must be smaller than a
  predeclared business limit. Require documented block cost and review
  capacity. A zero-block policy is a valid safe outcome.
- Test mutual exclusivity, ordering of review/block thresholds, invalid
  scores, budget exhaustion, no automatic block and no state mutation.

## 6L — faithful full-context logistic attribution

- Use the verified, frozen StandardScaler + LogisticRegression model.
  Calculate individual contributions `coef_i * standardized_feature_i`
  to the **pre-calibration decision margin**, and verify
  `intercept + sum(contrib) == model.decision_function(X)`.
- Expose through a new authenticated **read-only** endpoint for committed
  records only; reconstruct calibrated risk and compare to stored score.
- No causal interpretation, feature importance for populations, raw
  account IDs, model probability contribution, or SHAP claims.
- The existing observational evidence remains in a separate API object.
  If no attributor is supplied, return 501 (not an invented explanation).
- Verify exact selected 16-feature names/order and frozen model checksum.

## 6M — local privacy/auth closure (not public service approval)

- Restrict Host to loopback names using Flask 3.1 TRUSTED_HOSTS.
- Retain Basic auth required, CSRF Origin restrictions, no-store, strict
  browser CSP, security headers and HMAC account pseudonymization.
- Test spoofed Host, wrong credentials, cross-origin writes, static assets,
  no raw account disclosures and failure on changed privacy key.
- Do not claim data encryption, RBAC, secret rotation, retention, rate
  limiting, TLS or external-user authorization. These remain later
  deployment gates, not part of localhost completion.

## 6N — independent validation protocol and executable evaluator

- Freeze **before** selecting an independent dataset: label semantics,
  exact source provenance/license, chronological ordering, full 21-column
  feature feasibility, data access boundaries, overlap/dedup detection,
  missing-feature handling, scoring checksum, an operating policy held
  fixed before test labels are unsealed, and type-specific precision,
  recall, FPR, PR-AUC, review/false-block costs with uncertainty bounds.
- A fixture-only unit test must prove evaluator arithmetic. The fixture
  is NOT independent detection evidence.
- A real independent score/report must be read-only and non-overwriting;
  refuse absent/unclear provenance, modified model artifacts or leakage.
  Do not reuse previously inspected PaySim partitions or final holdout as
  an "independent" test; do not silently use datasets lacking required
  account/transaction semantics.
- Mark full independent model-quality validation **pending** until a
  legitimate, separately generated/licensed source and labeled evidence
  genuinely satisfying this protocol is available.

## Explicitly out of scope until user approves

IBM LinuxONE provisioning or code transfer, s390x benchmarking, deployment,
submission/pitch/video packaging and public-facing services.
