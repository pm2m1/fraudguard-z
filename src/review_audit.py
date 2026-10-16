"""Audit frozen fraud-review behavior on the reserved policy period.

Diagnostic only: this code never updates models, calibrators, or the review
threshold. Retrospective top-k selection uses the full policy-period ranking;
it cannot be implemented as a real-time policy without future information.
"""

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from .calibrate import load_calibration_and_policy_rows
from .features import TYPE_TO_INDEX


REPORTS = Path("reports")
MODELS = Path("models")
BUDGET_FRACTIONS = (0.0025, 0.005, 0.01, 0.02)


def summarize_selection(labels, selected):
    labels = np.asarray(labels)
    selected = np.asarray(selected, dtype=bool)

    if labels.shape != selected.shape or labels.ndim != 1:
        raise ValueError("Labels and selection must be one-dimensional and aligned")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels must be binary")

    total = len(labels)
    frauds = int(labels.sum())
    reviews = int(selected.sum())
    true_positives = int(labels[selected].sum())
    false_positives = reviews - true_positives

    return {
        "transactions": total,
        "frauds": frauds,
        "reviews": reviews,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": frauds - true_positives,
        "review_rate": reviews / total if total else 0.0,
        "precision": true_positives / reviews if reviews else 0.0,
        "recall": true_positives / frauds if frauds else None,
    }


def ranked_indices(scores):
    """Sort descending by score, resolving exact ties by source order."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("Scores must be a finite one-dimensional array")
    return np.lexsort((np.arange(len(scores)), -scores))


def top_k_selection(sorted_indices, length, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("Review budget must be between 0 and 1")
    k = int(np.ceil(length * fraction))
    selected = np.zeros(length, dtype=bool)
    selected[sorted_indices[:k]] = True
    return selected


def group_breakdown(labels, selected, types):
    output = {}
    for name in ("TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"):
        mask = np.asarray(types) == TYPE_TO_INDEX[name]
        output[name] = summarize_selection(labels[mask], selected[mask])
    return output


def score_model(model_path, calibrator_path, features):
    model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path)
    margins = model.decision_function(features).reshape(-1, 1)
    scores = np.asarray(calibrator.predict_proba(margins)[:, 1])
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Calibrated probabilities are invalid")
    return scores


def main():
    with (REPORTS / "split_manifest.json").open() as file:
        manifest = json.load(file)

    policy_path = REPORTS / "review_policy.json"
    policy_bytes = policy_path.read_bytes()
    policy = json.loads(policy_bytes)
    if policy["selected_model"] != "B" or policy["block_enabled"]:
        raise RuntimeError("Expected frozen Model B review-only policy")

    print("Loading policy-period features and labels. This may take a minute...")
    _, _, X, labels, types = load_calibration_and_policy_rows(manifest)

    expected = manifest["partitions"]["policy"]
    if len(labels) != expected["rows"] or int(labels.sum()) != expected["frauds"]:
        raise RuntimeError("Policy dataset does not match the manifest")

    scores = {}
    for name, width in (("A", 7), ("B", 8)):
        scores[name] = score_model(
            MODELS / f"model_{name.lower()}.joblib",
            MODELS / f"calibrator_{name.lower()}.joblib",
            X[:, :width],
        )

    frozen_selected = scores["B"] >= float(policy["review_threshold"])
    frozen = summarize_selection(labels, frozen_selected)
    if frozen["reviews"] != policy["review_count"] or frozen["true_positives"] != policy["frauds_in_review"]:
        raise RuntimeError("Frozen policy no longer reproduces its recorded outcomes")

    budget_rows = []
    for name in ("A", "B"):
        order = ranked_indices(scores[name])
        for fraction in BUDGET_FRACTIONS:
            selected = top_k_selection(order, len(labels), fraction)
            metrics = summarize_selection(labels, selected)
            budget_rows.append({"model": name, "fraction": fraction, **metrics})

    report = {
        "partition": "policy (steps 334-372)",
        "purpose": "Diagnostic review-workload audit, not an untouched test",
        "warning": (
            "Top-k results require access to all policy-period scores and "
            "are retrospective diagnostics, not executable streaming policies. "
            "The final holdout was previously inspected and is not rerun."
        ),
        "frozen_policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "frozen_threshold": float(policy["review_threshold"]),
        "frozen_policy_overall": frozen,
        "frozen_policy_by_type": group_breakdown(labels, frozen_selected, types),
        "retrospective_top_k": budget_rows,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS / "review_audit_policy.json"
    csv_path = REPORTS / "review_audit_policy.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(budget_rows[0]))
        writer.writeheader()
        writer.writerows(budget_rows)

    print("\nFrozen Model B policy, reserved policy period")
    print(f"Transactions: {frozen['transactions']:,}")
    print(f"Reviews: {frozen['reviews']:,} ({frozen['review_rate']:.2%})")
    print(f"Detected frauds: {frozen['true_positives']:,}")
    print(f"False alerts: {frozen['false_positives']:,}")
    print(f"Precision: {frozen['precision']:.2%}; Recall: {frozen['recall']:.2%}")
    print("\nRetrospective ranking at fixed review budgets")
    print(f"{'Model':<8} {'Budget':>8} {'Reviewed':>10} {'Frauds':>8} {'Precision':>11} {'Recall':>9}")
    for r in budget_rows:
        print(f"{r['model']:<8} {r['fraction']:>7.2%} {r['reviews']:>10,} {r['true_positives']:>8,} {r['precision']:>10.2%} {r['recall']:>8.2%}")
    print(f"\nSaved {json_path} and {csv_path}")
    print("No model artifacts, policies, or final-test results were modified.")


if __name__ == "__main__":
    main()
