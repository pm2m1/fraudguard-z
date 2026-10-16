# BankSim independently sourced chronological evaluation — preregistration

**Status:** Protocol fixed before running any BankSim final-period evaluation on this branch. This is a **separate synthetic payment simulator**, not production banking data, and not a test of PaySim CASH_OUT. BankSim fraud labels relate to purchases with customers, merchants and merchant categories.

## Dataset and permitted use
- Original BankSim synthetic payment dataset: https://www.kaggle.com/datasets/ntnu-testimon/banksim1
- Original authors: López-Rojas and Axelsson (2014), *BankSim: A Bank Payments Simulator for Fraud Detection Research*.
- Expected file: \`data/external/banksim/bs140513_032310.csv\` (download separately from the authorized source; do not commit/distribute).
- Reported original distribution: 594,643 transactions, 7,200 fraudulent transactions, around 180 simulated days; DO NOT treat these as verified counts until the authorized dataset is checked.
- Redistribution and commercial-use rights are not assumed. Reported dataset license is CC BY-NC-SA 4.0; verify terms at the source before competition reuse.
- The original Handbook benchmark and PaySim data/results are **not** pooled with BankSim.

## Non-negotiable protocol
1. Download BankSim independently. No fraud labels or test outcomes from this dataset may be used to choose features, hyperparameters, cutoffs, or review-policy allocations.
2. Validate unique required columns: \`step, customer, merchant, category, amount, fraud\`. Ignore age, gender, ZIP codes and identity strings as direct predictors; customer/merchant ID values are used solely to compute past-only activity history.
3. Sort by integer \`step\` ascending, breaking equal-step ties by original file row index. Compute features before observing each transaction. For each score, use only present fields and previously observed **unlabeled** transactions. No fraud labels used in feature construction.
4. Use chronological partitions by unique observed step values: earliest 60% steps for training; next 20% steps for score-threshold tuning; latest 20% steps for **one-time final evaluation**. Partition boundaries are determined from timestamps only (not labels). A fresh simulator *source* is not the same as an untouched real-world cohort.
5. Freeze exactly four model baselines with \`random_state=42\`, no adaptive training:
   - LR on log amount + train-fitted category one-hot.
   - HGB on exactly the same static inputs.
   - HGB on static inputs + past-only 7-step customer count/amount, merchant count/amount, and customer-merchant pair count (5 history features).
   - HGB amount-only control, to expose categorical simulation shortcuts.
   LR = StandardScaler + LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=300, random_state=42). HGB = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, max_bins=128, learning_rate=0.07, l2_regularization=10, class_weight='balanced', early_stopping=False, random_state=42).
6. Model scores are **uncalibrated**; do not call them predicted fraud probabilities. For each model, derive one **unlabeled** score cutoff from the validation partition's top 1% by score (ties at the cutoff included subject to capacity). Do not use validation or final labels to select cutoffs.
7. In the final period, make decisions sequentially, with frozen models/cutoffs and **a hard review cap of floor(1% of every prefix of each non-overlapping block of 10,000 transactions)**. Unused capacity is not retroactively filled. Same policy/cap for all models. Do not use global final-period retrospective top-k for a policy claim.
8. Simulate a **7-step fraud-label reporting delay**. Decisions never consult labels. Each label becomes available for offline outcome accounting only after \`current_step >= transaction_step+7\`, with the remaining labels released after a simulated reporting watermark 7 steps past the final observation. No training or decision-policy updates after delayed labels are released.
9. Compute each model's observed reviews, TP, FP, precision, recall, PR-AUC (retrospective *ranking diagnostic only*), and per-step review summaries. Primary *predeclared descriptive* comparison: HGB history vs HGB static. Include HGB amount-only as an anti-shortcut diagnostic. There is **no** post-test model selection or promotion.
10. Use \`--lock\` to create a manifest hashing the dataset bytes, this protocol, and evaluator source *without reading labels*. Commit/push this lock before invoking \`--evaluate\`. The evaluator verifies these hashes and a committed lock, and refuses to overwrite a final report. Keep the final report and data secure.

## Interpretation
- One BankSim chronological final partition permits a **one-time observational evaluation on an independently sourced synthetic simulator** after preregistration; it does not demonstrate transfer to real banks, robustness to simulator manipulation, IBM LinuxONE performance, or superior PaySim CASH_OUT recall.
- Performance on this dataset may largely reflect known injected BankSim fraud patterns or merchant categories; category-free amount-only control and history-vs-static comparison aid interpretation.
- A positive result is a candidate research finding, not necessarily algorithmic novelty. Replication on truly separate simulation runs/sources and uncertainty analysis would be needed before strong claims.
- Do not retrain or edit the policy after seeing final outcomes and then call the same final data an untouched test.
