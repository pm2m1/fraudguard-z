# FDH-inspired amended replication — archived-only diagnostic memo

**Status:** Read-only post hoc analysis of archived results. No model fitting, dataset regeneration, new holdout evaluation, cutoff adjustment, or production promotion.

**Provenance:** original experiment commit `d233ce7` failed on its first touched run because of unsigned NumPy subtraction; original marker preserved. Amended lock commit `b13f569`; complete archived nine-result commit `e646041`, branch `research/fdh-replication-amended-01`. All amended results are **exploratory**, not untouched confirmation. This note is in an isolated branch and makes no edits to source models or frozen production Model B.

## Outcomes from archived JSON only

Sources: `reports/fdh_amended_01_runs/seed_<seed>__<regime>.json` for seeds 104729, 130363, 155921 and regimes default, no_amount_shortcut, final_period_shift; `reports/fdh_amended_01_summary.json`.

| Metric | Static HGB | History HGB |
| --- | ---: | ---: |
| Total final-period transactions, nine runs | 117,561 | 117,561 |
| Total labeled frauds, nine runs | 5,823 | 5,823 |
| Actual reviews | 1,146 | 1,136 |
| Detected frauds / TPs | 1,099 | 1,082 |
| False alerts / FPs | 47 | 54 |
| Sum of per-run maximum reviews by the per-block 1% prefix rule | 1,173 | 1,173 |
| Aggregate fraction of that maximum utilized | 97.70% | 96.85% |
| Runs with larger PR-AUC ranking diagnostic | 0 | 9 |
| Runs with more detected frauds | 5 | 0 |

These totals repeat the *same seed* across three paired regimes and do not constitute nine independent datasets or observations for confidence intervals. The total maximum of 1,173 follows from 100 reviews in each complete 10,000-event block plus floor(1% of the remainder) in each run (127/133/131 maxima for the respective seed). This is an arithmetic bound derived from the protocol; **it is not an event-level audit that the prefix rule was applied correctly**.

Paired discordances across the nine experiments: 77 detected frauds were selected by history only versus 94 by static only, net -17 detections. History alone selected 88 reviews versus static alone 98, net -10 reviews. Individual model false alert totals: history 54, static 47.

## Fraud scenario coverage

| Mutually exclusive recorded terminal scenario (after overwrite precedence) | Labeled frauds | Static detections | History detections |
| --- | ---: | ---: | ---: |
| Scenario 1 — threshold amount | 530 | 141 | 131 |
| Scenario 2 — compromised terminal | 2,212 | 4 | 4 |
| Scenario 3 — compromised customer / amount multiplier | 3,081 | 954 | 947 |
| **Totals** | **5,823** | **1,099** | **1,082** |

Each model caught only 4/2,212 = approximately 0.18% of explicitly labeled scenario-2 fraud events, even though scenario 2 accounts for 2,212/5,823 = approximately 38.0% of fraud labels in these reports. These are recorded mutually exclusive scenario labels after scenario-3 overwrites, **not independent fraud causes**.

## Interpretation and bounded hypotheses

1. **Ranking/decision disconnect:** history AP exceeds static AP in all nine final runs (per-run increases range approximately 0.0031 to 0.0344), but there are five negative and four tied detection deltas at the prespecified threshold and cap. AP is computed over *all final scores*, while actual reviews are causal cutoff-and-budget decisions. Neither a better AP nor a saturated review budget implies more true positives at the deployed operating point.
2. **Review-cap utilization is high but not perfect:** 1,146/1,173 vs 1,136/1,173. The largest single-run shortfall is history, seed 104729, default: 114 of maximum 127 possible reviews, versus static 123. The original failure's negative direction is retained. This can explain **part** of the -9 difference in that run, but the seven equal-review comparisons include three history losses and four ties; lower review volume alone does not explain the entire pattern.
3. **Terminal compromise blind spot:** history features use seven-day counts, amounts and pair reuse, with *no past confirmed fraud-label history* or supervised terminal-risk signal. Scenario 2 marks compromise events without an inherent high-amount trigger, so historical activity counts need not distinguish it. This is an explanatory hypothesis, not a proven cause.
4. **Scenario-3 shortcut remains:** multiplying the compromised customer's amount by five produces high-value frauds. The no_amount_shortcut ablation only removes scenario 1's direct threshold, not scenario 3's multiplication, and therefore does not eliminate amount-label shortcut structure.
5. **Cross-simulator contrast is suggestive, not proof:** archived BankSim history gave +90 TPs with -60 reviews relative to static, while the separately defined FDH-inspired amended benchmark here gave -17 TPs with -10 reviews. Different data distributions, feature schemas, sample sizes and policies can account for these discrepancies. No causal generalization claim or model promotion is warranted.

## What the archive cannot establish

- It does not save event-level scores, per-event decisions, event-level identifiers or block-prefix traces. Therefore we cannot reconstruct top-k precision curves, simulate exact matched-review thresholds, audit each prefix-cap decision from outputs, locate false positives at event level, or run paired cluster bootstrap uncertainty.
- Aggregate reports cannot distinguish whether model scores, threshold crossing, or stream placement caused any particular missed transaction. Examining cutoff values (often near one) is diagnostic only: **scores are uncalibrated**.
- AP differences are not directly comparable across regimes with different prevalence. The 3 seeds within each regime are not real-world validation.

## Recommended next scientific decision

**Do not retune or redo these nine runs.** Archive this conclusion as a genuine negative operating-point result. If work continues, prepare a separate future experiment *before generating its data* with (a) a terminal-compromise-specific predeclared metric and stress test, (b) lagged fraud-feedback features available only after their original seven-day label delay, (c) static-vs-history and equal-capacity comparisons that report *actual* reviews, (d) an event-level pseudonymized and access-controlled evaluation trace with block-prefix counters where licensing permits, and (e) genuinely new seeds/generator parameters frozen before observation. The new experiment would be exploratory unless a genuinely untouched holdout and protocol are preregistered.

Keep original run markers and all amended archive JSON immutable. IBM LinuxONE deployment remains deferred.
