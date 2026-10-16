# Full-context candidate comparison — precommitted development protocol

Status: EXPLORATORY DEVELOPMENT ONLY. This protocol is authored before executing
the full-context candidate training workflow. It is NOT a new untouched holdout.

## Fixed inputs and partitions

- Read only the completed `data/processed/full_context_dev_v1` materialization.
- Require exact `reports/split_manifest.json` hash and `full_context.py`
  schema hash recorded when features were materialized.
- 5,285,888 events; rows [0, 3,177,938) train,
  [3,177,938, 4,263,279) validation,
  [4,263,279, 4,759,371) calibration,
  [4,759,371, 5,285,888) policy-tuning.
- This stage is explicitly **train and validation only**. It must not read,
  fit on, or report labels/scores for the calibration and policy partitions.
- PaySim step >=373 remains off limits for this candidate exercise. The
  previously used original final test is not restored to untouched status.

## Candidate models and comparison

Evaluate six frozen candidates on the **same** training/validation records:

| Family | Feature columns | Width |
| --- | --- | --- |
| Transaction | T1–T5, numeric columns 0..8 | 9 |
| Transaction + behavior | T1–T5, B1–B7, columns 0..15 | 16 |
| Full context | T1–T5, B1–B7, N1–N5, columns 0..20 | 21 |

For each feature family:
- `StandardScaler` fitted on train only + balanced `LogisticRegression`:
  `C=1.0`, `max_iter=300`, `solver=lbfgs`, `random_state=2026`.
- `HistGradientBoostingClassifier`: `learning_rate=0.07`,
  `max_iter=100`, `max_leaf_nodes=15`, `max_bins=128`,
  `l2_regularization=10.0`, `class_weight=balanced`,
  `early_stopping=False`, `random_state=2026`.

All model `predict_proba` values remain **uncalibrated** and may not be
used as deployed fraud probabilities or with the original Model B threshold.

## Reported exploratory diagnostics

- Validation PR-AUC / average precision.
- Retrospective global top-1% reviews, TP, FP, precision and recall.
- TP/FP and capture by TRANSFER and CASH_OUT at the **global** top-1% cutoff.
- Validation fraud prevalence and observed type counts.
- Per-candidate fit duration, model artifact checksum, feature family/schema.
- No review-cap streaming performance claim: hindsight top-k is not an
  actionable streaming threshold. No zero-false-block claim.

## Interpretation and promotion gate

Comparison is exploratory and **does not automatically select or deploy** the
highest PR-AUC candidate. Report if nonlinear/full-context features leverage
the known PaySim amount/balance simulator shortcut. A separate, prospective
robustness experiment and feature/simulator-shortcut sensitivity assessment
are necessary before choosing an operational candidate.

After inspecting development validation:
1. Freeze one or more candidates and document *why*, including limitations.
2. Only then fit probability calibration on its dedicated earlier-frozen
   calibration partition.
3. Only then derive ALLOW/REVIEW/BLOCK thresholds on policy partition using
   explicit review capacity and false-block constraints. Zero BLOCK is valid.
4. Never re-label this exploratory exercise as an independent final test.

The new training script must write **only** into a fresh ignored
`data/processed/full_context_candidates_v1` directory, with no overwrite,
never touch production model files or existing research archives.
