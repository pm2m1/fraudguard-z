# Simulator selection for FraudGuard Z forward robustness (16 October 2026)

## Decision

**Use a fresh, separately seeded run of the published Fraud Detection Handbook (FDH) simulator as the first replication *source*, NOT the previously downloaded 183-day FDH archive.** This source has explainable customer-terminal histories and three documented fraud scenarios; it is familiar from the archived Handbook research, so this represents NEW INDEPENDENTLY GENERATED RUNS of an existing simulator, not an independent third simulator or real-world validation.

Reference: https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html
Source: https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook/blob/main/Chapter_3_GettingStarted/SimulatedDataset.ipynb
License: Notebook code GNU GPL v3.0; prose CC BY-SA 4.0. Review attribution and source/distribution obligations before incorporating adapted code.

## Why not the easy-looking alternatives?

- farishaddad/fraud-synth, MIT 0.1.0: source inspection of SyntheticBankSimGenerator shows a *Bernoulli random fraud label* independent of amount/customer/merchant history, while edge_features[2] directly depends on label. Using [2] leaks label information; excluding it makes fraud detection essentially unlearnable from the remaining raw event attributes. Do NOT use this generator as evidence that historical detection improves. Could become a negative-control leakage test only, with advance declaration.
- micolocco/fraud-detection-simulator, Markov chain: behavior types and amount distributions are heavily separated; script does not expose a seed CLI and the amount distributions differ dramatically between legit/fraud. Needs substantial reproducibility and shortcut audits before credible replication.
- FDH is also imperfect: its first fraud rule marks transactions over amount 220 as fraud. This *synthetic amount shortcut* must be explicitly disclosed and isolated. The simulator itself is not proof of generalization.

## Replication methodology — to freeze before generation

1. Independently generate three new datasets with disjoint, documented seeds (preliminary candidate seeds 104729, 130363, 155921) and stable parameter settings. DO NOT execute generation until a locked run specification and generator implementation have been committed and checked.
2. Original FDH notebook hardcodes random seeds in several helper functions: customer profiles use random_state=0; transaction generation resets NumPy and Python RNG with each CUSTOMER_ID; fraud patterns draw by day seed. Merely calling the notebook three times or setting one top-level seed will NOT produce independent replicates. Implement explicit hierarchical RNG seeding for profile, transaction and fraud-scenario routines; unit-test same-seed reproducibility and different-seed divergence.
3. Require each generated run to have a unique dataset SHA-256 and event-ID namespace, and independently generated customer profiles and fraud allocations. Verify empirical diversity without looking at held-out outcomes.
4. Use original FDH simulator as separately generated run source; don't reuse saved .pkl daily files from data/external/handbook or the previously inspected 183-day holdout.
5. Predeclare three regimes: (a) default simulator with ALL documented fraud scenarios; (b) exclude amount >220 labeling rule to remove the simple amount shortcut; (c) drift regime with changes to customer spending/terminal profile distribution and fraud composition, with exact numeric parameters frozen and disclosed. All are simulator manipulations and cannot imply real-world effects.
6. Freeze the two-model comparison (static HGB vs past-only HGB), exact model hyperparameters, causal 7-day rolling windows, chronological train/tune/test partitions, score cutoffs from tuning scores only, a streaming hard 1% review cap, 7-day delayed-label reporting, error/tie/retry handling and reporting metrics. The same capacities apply to both models; report reviews used and true/false positives.
7. Define primary analysis as difference in true positives under the same 1% maximum review cap plus **actual reviews used** for each model. Secondary: PR-AUC ranking diagnostic, paired per-run descriptive differences, false alerts, and uncertainty across independent run seeds. Avoid calibrating/picking models or selecting seeds after final results.
8. If a run has no positives in a required segment, treat it as invalid according to a pre-specified *label-blind* data-generation check or report the condition; do not reroll the seed after reading evaluation labels.
9. Do not conflate generated-run replication with external simulator transfer. The one-time BankSim and Handbook archival results are unchanged. Real bank robustness and IBM LinuxONE are not established.

## Next engineering action

Implement a separate, testable generator adapter with explicit hierarchical RNG and dry-run metadata mode; pin upstream commit, license and parameters. **Do not execute new runs until the code and protocol are reviewed and committed.** Review whether use of source code is permitted in the intended datathon submission. Preserve all prior research files.
