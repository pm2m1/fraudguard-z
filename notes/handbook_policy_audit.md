# Operational fraud-review audit (research only)

This audit extends the already completed 183-day Fraud Detection Handbook synthetic benchmark. The initial full evaluation found:

- Total 1,754,155 transactions, 183 complete days.
- Final 354,643 transactions and 3,177 frauds.
- Both models used the same HGB family selected on validation.
- Retrospective test top 1%: baseline 781 TP / 2,766 FP; history 850 TP / 2,697 FP.
- Test PR-AUC 0.245405 baseline vs 0.262909 history.

**Caution:** the final test partition was already inspected. This audit is a supplementary operational diagnostic, not a new independent holdout, and does not justify tuning more models on this test.

## What this audit adds

- Reproduce previously committed training hyperparameters and check that new test PR-AUC / true positives match the original report. Fail closed if not.
- Choose a static decision cutoff for EACH model using only the middle 20% validation period, targeted to approximately 1% of validation transactions.
- Apply those frozen cutoffs to the later test period. Report actual test review rate, precision, recall, missed frauds, and workload drift.
- Measure daily review counts, true positives, and total fraud cases.
- Estimate paired confidence intervals for differences (history minus baseline) in recall, review rate, and precision with reproducible 7-day moving-block bootstrap.
- Leave production Model B, policy, and trained artifacts unchanged.

Important: two separately calibrated review cutoffs can lead to different review volumes on the test period. The fixed-threshold operational comparison is NOT necessarily same-budget. The original global-top-1% ranking is equal-budget but retrospective and NOT real-time.

## Commands

Stay on the existing research branch if needed. The audit branch starts from it:

\`\`\`bash
cd ~/IBM-Z-Datathon-2026
source .venv/bin/activate
git status --short
git fetch origin
git switch --track origin/research/behavioral-policy-audit
python -m pytest tests -q
\`\`\`

Keep the prior report at \`reports/handbook_behavioral_research.json\`. It was generated in the previous experiment; do not change its numbers or rerun any model search.

\`\`\`bash
python -m src.handbook_policy_audit \
  --input-dir data/external/handbook/data \
  --baseline-report reports/handbook_behavioral_research.json \
  --report reports/handbook_policy_audit.json
\`\`\`

The full-day feature extraction and HGB retraining will take some time, but avoid manual evaluation shortcuts. Check both model reproduction guards, fixed validation thresholds and test review-volume differences.

No IBM LinuxONE access needed for this research. No .pkl data files belong in Git. The results report can be committed after audit and test review, without raw transaction-level details.

## Remaining methodological work

- A genuinely new/untouched dataset, and model-selection protocol fixed BEFORE reading its labels, is needed for stronger evidence. The already inspected Handbook test is not fresh.
- The different simulator does not establish actual deployed PaySim model transfer.
- The baseline/history comparison is feature-engineering research, not evidence of a newly invented fraud-detection algorithm.
- Work remains on CASH_OUT recall on PaySim, online capacity management, calibration, uncertainty in independent environments, and operational security.
- Production access stays local until serving, authorization and TLS have been audited.
