# FDH-inspired seeded generator — pre-generation checkpoint

Status: **IMPLEMENTED, NOT APPROVED FOR FULL EXPERIMENT GENERATION**.

The generator is an independently written, simplified implementation inspired by the publicly documented Fraud Detection Handbook scenarios. It is **not numerically equivalent** to the upstream simulator and is not proof of external model generalization.

Upstream reference: Fraud-Detection-Handbook/fraud-detection-handbook, notebook Chapter_3_GettingStarted/SimulatedDataset.ipynb, pinned commit \`eabce290e52013e1672963d3728e3d63cd8b0705\`, blob \`3af320f0098f23c11e7fb7ff8262d9cca59daffe\`. Original upstream code is GPL-3.0; retain attribution and obtain a license compatibility review before redistribution of any derivative.

## Frozen candidate generation settings (not yet experiment-lock committed)
- Seeds: 104729, 130363, 155921. Independent NumPy SeedSequence streams for customer positions/profiles, terminal positions, individual customer activity, daily compromised sets and daily scenario draws.
- 600 customers, 3,000 terminals, 90 daily steps, distance radius 12, per-customer mean daily rate uniform [0.3, 2.2], customer mean transaction amount uniform [5, 100]. Each amount is drawn from a normal distribution with standard deviation one-half the customer mean and is clamped through a fallback for negative samples.
- Three regimes: default (all three scenarios), no_amount_shortcut (remove direct amount > 220 label), final_period_shift (multiply amounts by 1.35 for days >= 72; compromise new terminals/day from 2 to 3, customers/day from 3 to 5).
- Scenario 1: amount > 220 means fraud; Scenario 2: 2 newly compromised terminals/day, each active 28 days; Scenario 3: 3 newly compromised customers/day, active 14 days, independently select roughly one third of their events and multiply transaction amount by 5. Scenario 3 overrides other scenario labels; scenario 2 overrides scenario 1. The generator is a simplified procedural variant and does not replicate the exact Handbook function implementations.
- Original training, tuning, and final cutoffs are NOT yet specified for these runs. No full-size data has been generated or evaluated.

## Review checklist before unlocking generation
1. On your machine, pull branch and run all tests; report errors.
2. Inspect generated toy examples only. Source code deliberately exposes only a \`--spec\` command-line interface; it does not create full datasets.
3. Review risks: fraud-label leakage through generator rules, amount shortcuts (especially scenario 3 amount multiplication), independently repeated entity identities across seeds, fixed small customer/terminal populations, temporal compromises creating future label patterns, and drift only on the final period.
4. Write and test a distinct chronological benchmark evaluator with frozen train/tune/test steps, per-transaction past-only features, lagged labels, and fixed review-budget policy. We must audit source code for leakage and use unit fixtures before generating full runs.
5. Commit a manifest with SHA-256 of generator and evaluator code, seed/regime/config values, and run protocol. Enable a separate \`--generate-locked\` command only after the lock has been committed and verified.
6. Keep synthetic input files out of version control and all prior BankSim, Handbook and PaySim artifacts unmodified. Do not use BandSim or PaySim holdouts to tune this research.

Note: results across these three seeds remain draws from ONE related synthetic family. They are not independent simulator families or validation on real bank data.
