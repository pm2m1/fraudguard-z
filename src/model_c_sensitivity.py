"""Stress-test experimental Model C without assigning counterfactual fraud labels.

The data loader only reads steps 1-306, and only validation features are
perturbed. This is sensitivity analysis, NOT an accuracy or robustness
certification on independent real-world fraud data.
"""
import json
from math import isfinite
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from .model_c import MODEL_C_FEATURE_NAMES
from .model_c_ablations import validation_rule
from .train_model_c import load_train_validation

ROOT = Path(".")
MANIFEST = ROOT / "reports/split_manifest.json"
DATA = ROOT / "data/raw/PS_20174392719_1491204439457_log.csv"
MODEL = ROOT / "models/model_c_experimental.joblib"
REPORT = ROOT / "reports/model_c_sensitivity_validation.json"

# In the 17-column Model C schema:
# 0 amount, 6 sender balance, 8 ratio, 9 nearly drained, 10 exceeds balance.
CURRENT_FINANCIAL_COLUMNS = (0, 6, 8, 9, 10)


def alter_current_financial_features(X, *, amount_factor=1.0,
                                     balance_factor=1.0):
    """Transform one event at a time, keeping all past-only features fixed.

    Historical features are frozen deliberately. This does not simulate a
    causally consistent alternative stream with changed future balances.
    """
    if (np.ndim(X) != 2 or X.shape[1] != len(MODEL_C_FEATURE_NAMES)
            or not np.all(np.isfinite(X))):
        raise ValueError("Expected finite 17-column feature matrix")
    if (not isfinite(amount_factor) or not isfinite(balance_factor)
            or amount_factor <= 0 or balance_factor <= 0):
        raise ValueError("Factors must be finite and positive")

    changed = np.array(X, copy=True, dtype=np.float32)
    amount = np.expm1(X[:, 0].astype(np.float64)) * amount_factor
    balance = np.expm1(X[:, 6].astype(np.float64)) * balance_factor
    if not np.all(np.isfinite(amount)) or not np.all(np.isfinite(balance)):
        raise ValueError("Financial perturbation overflow")
    changed[:, 0] = np.log1p(amount)
    changed[:, 6] = np.log1p(balance)
    changed[:, 8] = np.log1p(np.minimum(amount / (balance + 1.0), 1e12))
    changed[:, 9] = (
        (balance > 0) & (0.95 * balance <= amount)
        & (amount <= 1.01 * balance)
    ).astype(np.float32)
    changed[:, 10] = (amount > balance).astype(np.float32)
    if not np.all(np.isfinite(changed)):
        raise ValueError("Perturbation produced invalid features")
    return changed


def select_top_k(scores, k):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("Scores must be one-dimensional and finite")
    if not 1 <= k <= len(scores):
        raise ValueError("Invalid review count")
    order = np.lexsort((np.arange(len(scores)), -scores))
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:k]] = True
    return selected


def summarize_sensitivity(original_scores, changed_scores, selected_before,
                          selected_after, original_rule_matches):
    original_scores = np.asarray(original_scores, dtype=float)
    changed_scores = np.asarray(changed_scores, dtype=float)
    before = np.asarray(selected_before, dtype=bool)
    after = np.asarray(selected_after, dtype=bool)
    rule = np.asarray(original_rule_matches, dtype=bool)
    if (original_scores.ndim != 1 or any(v.shape != original_scores.shape
                for v in (changed_scores, before, after, rule))):
        raise ValueError("Misaligned sensitivity arrays")

    deltas = np.abs(changed_scores - original_scores)
    original_rule_reviews = int(np.sum(before & rule))
    remaining_rule_reviews = int(np.sum(before & after & rule))
    return {
        "mean_absolute_score_change": float(np.mean(deltas)),
        "p95_absolute_score_change": float(np.percentile(deltas, 95)),
        "top_1pct_overlap_count": int(np.sum(before & after)),
        "top_1pct_overlap_fraction": float(
            np.sum(before & after) / max(1, np.sum(before))
        ),
        "original_exact_balance_rule_matches": int(np.sum(rule)),
        "original_rule_matches_in_top_1pct": original_rule_reviews,
        "original_rule_matches_remaining_in_top_1pct": remaining_rule_reviews,
        "fraction_original_rule_reviews_retained": (
            remaining_rule_reviews / original_rule_reviews
            if original_rule_reviews else None
        ),
    }


def main():
    if not MODEL.exists():
        raise FileNotFoundError(
            f"{MODEL} is missing. First run python -m src.train_model_c."
        )
    manifest = json.loads(MANIFEST.read_text())
    if manifest["split_end_steps"][:2] != [238, 306]:
        raise RuntimeError("Unexpected split manifest")
    print("Rebuilding features through validation step 306...", flush=True)
    X, y, _, cutoff = load_train_validation(manifest, DATA)
    features = X[cutoff:]
    labels = y[cutoff:]
    del X, y
    rule_matches = validation_rule(DATA, manifest)
    if len(rule_matches) != len(features):
        raise RuntimeError("Validation rule/feature alignment failed")
    model = joblib.load(MODEL)
    if model.n_features_in_ != 17:
        raise RuntimeError("Model C artifact must have 17 input features")
    scores = model.predict_proba(features)[:, 1]
    pr_auc = float(average_precision_score(labels, scores))
    baseline_report = json.loads(
        (ROOT / "reports/model_c_validation.json").read_text()
    )
    recorded_pr_auc = baseline_report["model_c"]["overall"]["pr_auc"]
    if abs(pr_auc - recorded_pr_auc) > 1e-6:
        raise RuntimeError("Model artifact does not reproduce validation PR-AUC")

    k = int(np.ceil(len(scores) * 0.01))
    selected = select_top_k(scores, k)
    experiments = {
        "amount_decreased_10pct": (0.90, 1.0),
        "sender_balance_increased_10pct": (1.0, 1.10),
        "amount_and_balance_decreased_10pct": (0.90, 0.90),
    }
    report = {
        "status": "EXPERIMENTAL VALIDATION SENSITIVITY ONLY",
        "partition": "Validation steps 239-306; train history reconstructed",
        "baseline_model": "Uncalibrated experimental Model C",
        "baseline_pr_auc_reproduced": pr_auc,
        "validation_rows": len(scores),
        "original_review_budget_count": k,
        "caveats": [
            "This is not an independent test or a claim of real-world performance.",
            "Perturbations alter current event financial data with past-only history fixed; future account balances are not resimulated.",
            "Original fraud labels cannot be assumed correct for altered transactions. Accordingly no counterfactual precision/recall is computed.",
            "All top-1% queues are retrospective rankings, not streaming policies.",
            "Original holdout steps 373+ are not read. No model or policy artifacts are modified.",
        ],
        "experiments": {},
    }

    for name, (amount_factor, balance_factor) in experiments.items():
        print(f"Scoring sensitivity case: {name}", flush=True)
        altered = alter_current_financial_features(
            features, amount_factor=amount_factor,
            balance_factor=balance_factor,
        )
        changed_scores = model.predict_proba(altered)[:, 1]
        changed_top = select_top_k(changed_scores, k)
        report["experiments"][name] = {
            "amount_factor": amount_factor,
            "balance_factor": balance_factor,
            **summarize_sensitivity(
                scores, changed_scores, selected, changed_top, rule_matches
            ),
        }
        del altered, changed_scores, changed_top

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    for name, result in report["experiments"].items():
        print(f"{name}: top-1% overlap={result['top_1pct_overlap_fraction']:.2%}, "
              f"original rule-positive alerts retained="
              f"{result['fraction_original_rule_reviews_retained']:.2%}")
    print(f"Saved {REPORT}")
    print("Production model and policy untouched.")


if __name__ == "__main__":
    main()
