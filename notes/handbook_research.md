# Behavioral research on a second synthetic simulator

**Research branch only:** this code does not modify frozen PaySim Model B, its review policy, calibration, or final test reports.

## Research question

Does 24-hour past-only customer/merchant activity improve fraud ranking over transaction-only amount/time features when model family, training window, and 1% retrospective review budget are held fixed?

The second simulator is the [Fraud Detection Handbook synthetic dataset](https://github.com/Fraud-Detection-Handbook/simulated-data-raw), which differs from PaySim. **This is not real bank data and does not establish direct portability of PaySim-trained Model B weights.**

Inputs: TRANSACTION_ID, TX_TIME_SECONDS, TX_TIME_DAYS, CUSTOMER_ID, TERMINAL_ID, TX_AMOUNT, and TX_FRAUD. The first six are metadata/current-time inputs; the label is *only* used to fit models and evaluate them. No future fraud labels, post-transaction amounts, recipient-balance equality checks, or group-level fraud rates are used as features. Chronology follows TX_TIME_SECONDS, then TRANSACTION_ID.

## Setup

From FraudGuard Z repository root:

\`\`\`bash
source .venv/bin/activate
git fetch origin
git switch --track origin/research/behavioral-cross-simulator
python -m pytest tests -q
\`\`\`

Obtain the dataset into an ignored local folder, outside Git. The Handbook distributes per-day .pkl files in its \`data/\` directory:

\`\`\`bash
mkdir -p data/external
git clone --depth 1 https://github.com/Fraud-Detection-Handbook/simulated-data-raw.git data/external/handbook
ls data/external/handbook/data/*.pkl | wc -l
\`\`\`

Confirm downloaded files originate from the intended source before loading; **Python pickle files execute code when loaded, so do not use untrusted pickle files**. Never commit the external dataset, secret keys, local databases, or private model binaries. If .gitignore does not cover \`data/external/\`, update it before staging files.

Fast developer smoke test only:

\`\`\`bash
python -m src.handbook_research --input-dir data/external/handbook/data \
  --max-days 12 --report reports/handbook_smoke.json
\`\`\`

**Do not interpret or publish the smoke-test score as independent final performance.** For a real evaluation, use the entire dataset and do not use \`--max-days\`:

\`\`\`bash
python -m src.handbook_research --input-dir data/external/handbook/data \
  --report reports/handbook_behavioral_research.json
\`\`\`

## Evaluation protocol

- Train: first 60% of complete days.
- Validation: next 20%; choose model family by validation PR-AUC and recall at retrospective top-1%.
- Test: final 20%; report *both* transaction-only and historical features for the chosen family.
- Candidates: balanced logistic baseline/history and balanced histogram boosting baseline/history, all using deterministic settings.
- Features: log(amount), transaction hour sin/cos and day index flag as baseline; plus customer/terminal counts, 24h means, relative customer amount, graph unique-neighbor counts and repeat pair counts.
- At inference time, every row sees only previously observed transactions, including previous rows at the same timestamp (ordered by ID).
- Evaluation's top-1% is a *retrospective global* allocation and must not be represented as an online policy.
- The historical and baseline models are trained on the second simulator from scratch; no claim that trained weights transfer from PaySim.
- Record day counts, exact dataset version, software versions, seeds, and negative results, rather than optimizing the held-out test.

## Important limitations / next gates

The Handbook generator may have algorithmic fraud patterns too, so a large improvement is not sufficient on its own to demonstrate general real-world generalization. A genuine extra research contribution must be compared to published behavioral baselines. Future follow-ups should implement calibrated rolling review budgets, robust bootstrap uncertainty, temporal drift/stability, and dataset-specific threat-model analysis. Never promote the research model into the production scorer on this evidence alone.

IBM LinuxONE is deferred explicitly at user's request.
