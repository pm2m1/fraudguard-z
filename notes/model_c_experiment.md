# FraudGuard Z Model C experimental evaluation

This experiment is separate from the deployed Model B.

## Hypothesis

On the chronological validation period, sender/recipient behavior and
amount-versus-balance features may rank CASH_OUT frauds better than the
existing recipient-count baseline. A nonlinear histogram gradient-boosting
model tests this hypothesis.

## Frozen comparison design

- Train: PaySim steps 1–238 (3,177,938 transactions).
- Validation: steps 239–306 (1,085,341 transactions).
- Do not read steps 307 or later for this experiment.
- Compare baseline Model B and experimental Model C on **the same validation
  rows** using PR-AUC and retrospective **global** top-1% review ranking.
- The global selection is then broken down by TRANSFER and CASH_OUT.
  It is not a deployable streaming review policy.
- For a meaningful candidate improvement, seek at least **20 more frauds
  detected overall**, including **15 more CASH_OUT frauds**, at the same
  overall 1% review count, while losing no more than **10 TRANSFER detections**.
  These are exploratory acceptance criteria, not statistical significance.
- Do not claim fresh out-of-sample generalization from this run. The original
  final holdout has already been inspected in earlier work.

## Implementation constraints

- The first eight features of Model C must reproduce Model B exactly.
- Current events are added to history only after features are extracted.
- Fraud labels and post-event balances are never used as features.
- Model C's `predict_proba` outputs are **uncalibrated model scores** because
  the classifier uses class balancing. Do not display them as real fraud
  probabilities or use them with Model B's frozen threshold.
- Model C is NOT wired into the production SQLite engine, policy, or server.
- No change to Model A/B artifacts, their checksums, the frozen review
  threshold, or final-test results.

## Run on the Mac

```bash
cd ~/IBM-Z-Datathon-2026
source .venv/bin/activate
python -m pytest tests -q
python -m src.train_model_c
cat reports/model_c_validation.json
```

Training uses the complete 3.18-million-row train period. Allow several
minutes and ensure sufficient free memory. It saves
`models/model_c_experimental.joblib` (ignored by Git) and
`reports/model_c_validation.json`.

If Model C wins, the next step is to separately implement the **identical**
feature contract in a commit-first, crash-safe state engine, then calibrate
and evaluate a new provisional policy without retroactively changing the
existing final-test claim. Deployment requires further engineering.
