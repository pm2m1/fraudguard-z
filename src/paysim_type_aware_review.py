"""Exploratory online review-budget allocation on PaySim validation data.

Only original train (steps 1..238) and validation (239..306) are read.
The HGB classifier uses SEVEN balance-free features: amount, five type
indicators, and prior recipient count. The validation segment is split in
original row order: first half sets unlabeled score cutoffs; second half
evaluates strictly online decisions with <=1% review capacity per 10k
transactions (including every prefix within each block).

IMPORTANT: this entire validation set was previously inspected and the model
family was chosen with that knowledge. Results are developmental diagnostics,
not an independent test and not grounds for production promotion. No
production model, review threshold, or final-test artifact is modified.
"""
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from .features import TYPE_TO_INDEX
from .train_model_c import load_train_validation

REPORTS = Path("reports")
TRANSFER = TYPE_TO_INDEX["TRANSFER"]
CASH_OUT = TYPE_TO_INDEX["CASH_OUT"]
BUDGET = 0.01
BLOCK_SIZE = 10_000
PREDICTOR_COLUMNS = (0, 1, 2, 3, 4, 5, 7)
TYPE_SHARES = (0.25, 0.50, 0.75)


def cutoff_at_quota(scores, quota):
    """Find unlabeled cutoff for up to quota high-score rows.

    Ties can generate more candidates than quota; streaming counters
    enforce the actual cap. -inf allows every row, inf allows none.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite one-dimensional vector")
    if not isinstance(quota, (int, np.integer)) or quota < 0:
        raise ValueError("Quota must be a nonnegative integer")
    if quota == 0:
        return math.inf
    if quota >= len(scores):
        return -math.inf
    return float(np.partition(scores, len(scores) - quota)[len(scores) - quota])


def policy_thresholds(tune_scores, tune_types, *, transfer_share=None,
                      restrict_eligible=False, budget=BUDGET):
    """Set thresholds on earlier, unlabeled chronological score records."""
    scores = np.asarray(tune_scores, dtype=float)
    kinds = np.asarray(tune_types)
    if scores.shape != kinds.shape or scores.ndim != 1 or len(scores) == 0:
        raise ValueError("Invalid tuning arrays")
    if not np.isfinite(scores).all() or not 0 < budget <= 1:
        raise ValueError("Invalid score or budget")
    total = int(math.floor(len(scores) * budget))
    eligible = (kinds == TRANSFER) | (kinds == CASH_OUT)

    if transfer_share is None:
        candidates = scores[eligible] if restrict_eligible else scores
        return {"GLOBAL": cutoff_at_quota(candidates, total)}

    if not 0 <= transfer_share <= 1:
        raise ValueError("Invalid transfer allocation")
    transfer_quota = int(round(total * transfer_share))
    cashout_quota = total - transfer_quota
    return {
        "TRANSFER": cutoff_at_quota(scores[kinds == TRANSFER], transfer_quota),
        "CASH_OUT": cutoff_at_quota(scores[kinds == CASH_OUT], cashout_quota),
    }


def select_online(scores, kinds, thresholds, *, transfer_share=None,
                  restrict_eligible=False, budget=BUDGET, block_size=BLOCK_SIZE):
    """Causal review decisions with an exact <=budget cap at each block prefix.

    Inputs are consumed left to right; the decision for event i uses only
    that event's kind/score, frozen cutoffs and earlier decisions. No
    retrospectively chosen top-k indices or label access.
    """
    scores = np.asarray(scores, dtype=float)
    kinds = np.asarray(kinds)
    if (
        scores.ndim != 1 or scores.shape != kinds.shape
        or not np.isfinite(scores).all()
        or not isinstance(block_size, int) or block_size < 1
        or not 0 < budget <= 1
    ):
        raise ValueError("Invalid streaming inputs")

    allocated = np.zeros(len(scores), dtype=bool)
    total_used = 0
    transfer_used = 0
    cashout_used = 0

    for i, (score, txn_type) in enumerate(zip(scores, kinds)):
        position = i % block_size + 1
        if position == 1:
            total_used = 0
            transfer_used = 0
            cashout_used = 0
        capacity = int(math.floor(position * budget + 1e-10))
        if total_used >= capacity:
            continue

        if transfer_share is None:
            if restrict_eligible and txn_type not in (TRANSFER, CASH_OUT):
                continue
            if score < thresholds["GLOBAL"]:
                continue
        else:
            transfer_limit = int(round(capacity * transfer_share))
            cashout_limit = capacity - transfer_limit
            if txn_type == TRANSFER:
                if transfer_used >= transfer_limit or score < thresholds["TRANSFER"]:
                    continue
            elif txn_type == CASH_OUT:
                if cashout_used >= cashout_limit or score < thresholds["CASH_OUT"]:
                    continue
            else:
                continue

        allocated[i] = True
        total_used += 1
        if txn_type == TRANSFER:
            transfer_used += 1
        elif txn_type == CASH_OUT:
            cashout_used += 1

    return allocated


def describe(labels, kinds, scores, selected):
    labels = np.asarray(labels)
    kinds = np.asarray(kinds)
    scores = np.asarray(scores)
    selected = np.asarray(selected, dtype=bool)
    if (
        labels.ndim != 1 or
        labels.shape != kinds.shape or
        labels.shape != scores.shape or
        labels.shape != selected.shape or
        not np.isin(labels, (0, 1)).all()
    ):
        raise ValueError("Mismatched evaluation data")
    n = len(labels)
    frauds = int(labels.sum())
    reviews = int(selected.sum())
    tp = int(labels[selected].sum())

    output = {
        "transactions": n,
        "frauds": frauds,
        "reviews": reviews,
        "review_rate": reviews / n if n else 0.0,
        "true_positives": tp,
        "false_positives": reviews - tp,
        "precision": tp / reviews if reviews else None,
        "recall": tp / frauds if frauds else None,
        "pr_auc_ranking_only": (
            float(average_precision_score(labels, scores))
            if 0 < frauds < n else None
        ),
        "by_type": {},
    }
    for name, code in (("TRANSFER", TRANSFER), ("CASH_OUT", CASH_OUT)):
        mask = kinds == code
        y = labels[mask]
        selection = selected[mask]
        hits = int(y[selection].sum())
        count = int(selection.sum())
        denominator = int(y.sum())
        output["by_type"][name] = {
            "transactions": int(mask.sum()),
            "frauds": denominator,
            "reviews": count,
            "true_positives": hits,
            "false_positives": count - hits,
            "precision": hits / count if count else None,
            "recall": hits / denominator if denominator else None,
        }
    return output


def threshold_tie_audit(tuning_scores, threshold):
    """Quantify exact score ties at a fixed, unlabeled cutoff."""
    scores = np.asarray(tuning_scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Invalid tuning score vector")
    if threshold == math.inf:
        return {"above": 0, "equal": 0, "at_or_above": 0}
    if threshold == -math.inf:
        return {"above": len(scores), "equal": 0, "at_or_above": len(scores)}
    above = int(np.count_nonzero(scores > threshold))
    equal = int(np.count_nonzero(scores == threshold))
    return {"above": above, "equal": equal, "at_or_above": above + equal}


def paired_block_stability(labels, reference, candidate, *,
                           block_size=BLOCK_SIZE, draws=3000, seed=42):
    """Descriptive paired moving-period variation under fixed policies.

    Non-overlapping chronological blocks are resampled jointly. These are
    exploratory intervals, not inference on an untouched test. The policy
    candidate was chosen after examining the same evaluation outcomes.
    """
    y = np.asarray(labels)
    ref = np.asarray(reference, dtype=bool)
    alt = np.asarray(candidate, dtype=bool)
    if (y.ndim != 1 or not len(y) or y.shape != ref.shape
            or y.shape != alt.shape or not np.isin(y, (0, 1)).all()
            or block_size < 1 or draws < 100):
        raise ValueError("Invalid paired stability arguments")
    tp_blocks, review_blocks = [], []
    for begin in range(0, len(y), block_size):
        end = min(begin + block_size, len(y))
        tp_blocks.append(int(np.sum(y[begin:end] * alt[begin:end])) -
                         int(np.sum(y[begin:end] * ref[begin:end])))
        review_blocks.append(int(alt[begin:end].sum()) -
                             int(ref[begin:end].sum()))
    tp_blocks = np.array(tp_blocks, dtype=int)
    review_blocks = np.array(review_blocks, dtype=int)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(tp_blocks), size=(draws, len(tp_blocks)))
    tp_samples = tp_blocks[samples].sum(axis=1)
    review_samples = review_blocks[samples].sum(axis=1)
    return {
        "reference": "global_all",
        "candidate": "type_aware_75T_25C",
        "blocks": len(tp_blocks),
        "block_size": block_size,
        "observed_extra_fraud_detections": int(tp_blocks.sum()),
        "observed_review_count_difference": int(review_blocks.sum()),
        "additional_tp_95pct_descriptive_interval": [
            float(x) for x in np.quantile(tp_samples, [.025, .975])
        ],
        "additional_reviews_95pct_descriptive_interval": [
            float(x) for x in np.quantile(review_samples, [.025, .975])
        ],
        "bootstrap_fraction_positive_detection_difference": float(
            np.mean(tp_samples > 0)
        ),
        "per_block_extra_tp": tp_blocks.tolist(),
        "warning": (
            "Paired block resampling on an already examined validation suffix. "
            "This is a descriptive stability check, NOT confirmatory inference "
            "or proof of future effectiveness. Candidate policy selected "
            "after examining these outcomes."
        ),
    }


def run_experiment(manifest):
    if manifest["split_end_steps"][:2] != [238, 306]:
        raise RuntimeError("Unexpected PaySim development split")
    print("Extracting past-only features through validation step 306...", flush=True)
    X, labels, types, train_end = load_train_validation(manifest)
    y_val = labels[train_end:]
    kinds_val = types[train_end:]
    middle = len(y_val) // 2
    if middle < BLOCK_SIZE:
        raise RuntimeError("Validation is too small for bounded streaming evaluation")

    print("Training balance-free HGB on original train partition...", flush=True)
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=15,
        max_bins=128,
        learning_rate=0.07,
        l2_regularization=10.0,
        class_weight="balanced",
        early_stopping=False,
        random_state=42,
    )
    model.fit(X[:train_end, PREDICTOR_COLUMNS], labels[:train_end])
    print("Scoring original validation partition...", flush=True)
    scores = model.predict_proba(X[train_end:, PREDICTOR_COLUMNS])[:, 1]
    tune_scores, eval_scores = scores[:middle], scores[middle:]
    tune_types, eval_types = kinds_val[:middle], kinds_val[middle:]
    eval_labels = y_val[middle:]

    configs = [
        ("global_all", None, False),
        ("global_eligible_only", None, True),
        *((f"type_aware_{int(share*100)}T_{int((1-share)*100)}C", share, False)
          for share in TYPE_SHARES),
    ]
    results = {}
    selections = {}
    for name, share, restrict in configs:
        cutoffs = policy_thresholds(
            tune_scores, tune_types, transfer_share=share,
            restrict_eligible=restrict,
        )
        chosen = select_online(
            eval_scores, eval_types, cutoffs, transfer_share=share,
            restrict_eligible=restrict,
        )
        metrics = describe(eval_labels, eval_types, eval_scores, chosen)
        selections[name] = chosen
        if share is None:
            tune_slice = tune_scores[
                ((tune_types == TRANSFER) | (tune_types == CASH_OUT))
                if restrict else np.ones(len(tune_scores), dtype=bool)
            ]
            tie_audit = {"GLOBAL": threshold_tie_audit(tune_slice, cutoffs["GLOBAL"])}
        else:
            tie_audit = {
                "TRANSFER": threshold_tie_audit(tune_scores[tune_types == TRANSFER], cutoffs["TRANSFER"]),
                "CASH_OUT": threshold_tie_audit(tune_scores[tune_types == CASH_OUT], cutoffs["CASH_OUT"]),
            }
        results[name] = {
            "score_ties_in_tuning_at_cutoffs": tie_audit,
            "tuning_thresholds": {
                kind: (None if not math.isfinite(cutoff) else cutoff)
                for kind, cutoff in cutoffs.items()
            },
            "metrics": metrics,
        }
        print(
            f"{name}: reviews={metrics['reviews']:,}, "
            f"TP={metrics['true_positives']}, FP={metrics['false_positives']}, "
            f"review_rate={metrics['review_rate']:.3%}",
            flush=True,
        )

    return {
        "status": "EXPLORATORY_DEVELOPMENT_ONLY",
        "dataset": "PaySim synthetic dataset",
        "model": "HistGradientBoostingClassifier balance-free seven predictors",
        "predictor_columns": list(PREDICTOR_COLUMNS),
        "original_train_steps": [1, 238],
        "original_validation_steps": [239, 306],
        "tuning_validation_prefix_rows": int(middle),
        "evaluation_validation_suffix_rows": int(len(eval_labels)),
        "split_method": "chronological original validation row midpoint (may bisect a step)",
        "budget_fraction": BUDGET,
        "budget_block_size": BLOCK_SIZE,
        "policy": "fixed unlabeled score cutoffs from earlier validation rows; causal block-prefix hard caps; unused quota not reallocated",
        "warning": (
            "Original validation and final holdout were previously inspected. "
            "Model family and feature sets have already been developed using "
            "validation feedback. Comparison is exploratory, NOT independent "
            "evidence or a deployed policy. Rates can differ below the same "
            "maximum 1% review budget; compare detections AND workloads. "
            "Scores are uncalibrated, not fraud probabilities."
        ),
        "results": results,
        "posthoc_stability_75T25C_vs_global": paired_block_stability(
            eval_labels, selections["global_all"], selections["type_aware_75T_25C"],
        ),
    }


def main():
    manifest = json.loads((REPORTS / "split_manifest.json").read_text())
    result = run_experiment(manifest)
    out = REPORTS / "paysim_type_aware_review_experiment.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {out}")
    print("Original production models and policies unchanged.")


if __name__ == "__main__":
    main()
