"""Compare validation ranking quality of Models A and B."""

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from .features import TYPE_TO_INDEX


REPORT_DIR = Path("reports")
REVIEW_FRACTION = 0.01


def evaluate_group(y, scores, model_name, group_name):
    n = len(y)
    positives = int(y.sum())
    negatives = n - positives

    if n == 0:
        return None

    # Select the highest-scoring 1% of transactions.
    k = max(1, int(np.ceil(n * REVIEW_FRACTION)))

    top_indices = np.argpartition(scores, -k)[-k:]
    true_positives = int(y[top_indices].sum())
    false_positives = k - true_positives

    result = {
        "model": model_name,
        "group": group_name,
        "rows": n,
        "frauds": positives,
        "pr_auc": (
            float(average_precision_score(y, scores))
            if positives and negatives else None
        ),
        "roc_auc": (
            float(roc_auc_score(y, scores))
            if positives and negatives else None
        ),
        "review_count": k,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "precision_top_1pct": true_positives / k,
        "recall_top_1pct": (
            true_positives / positives
            if positives else None
        ),
        "fpr_top_1pct": (
            false_positives / negatives
            if negatives else None
        ),
    }

    return result


def show_number(value):
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def main():
    path = REPORT_DIR / "validation_predictions.npz"

    if not path.exists():
        raise FileNotFoundError(
            "Run python -m src.train first"
        )

    with np.load(path, allow_pickle=False) as data:
        y = data["labels"].astype(np.uint8)
        groups = data["transaction_types"]
        scores_a = data["scores_a"]
        scores_b = data["scores_b"]

    assert len(y) == len(groups) == len(scores_a) == len(scores_b)

    masks = {
        "ALL": np.ones(len(y), dtype=bool),
        "TRANSFER": groups == TYPE_TO_INDEX["TRANSFER"],
        "CASH_OUT": groups == TYPE_TO_INDEX["CASH_OUT"],
    }

    results = []

    for group_name, mask in masks.items():
        for model_name, scores in (
            ("A", scores_a),
            ("B", scores_b),
        ):
            result = evaluate_group(
                y[mask],
                scores[mask],
                model_name,
                group_name,
            )

            if result is not None:
                results.append(result)

    output = REPORT_DIR / "model_comparison.csv"

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0]),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n=== FRAUDGUARD Z VALIDATION RESULTS ===")
    print(
        f"{'Group':<12} {'Model':<7} "
        f"{'PR-AUC':>12} {'P@1%':>12} {'R@1%':>12}"
    )

    for row in results:
        print(
            f"{row['group']:<12} "
            f"{row['model']:<7} "
            f"{show_number(row['pr_auc']):>12} "
            f"{show_number(row['precision_top_1pct']):>12} "
            f"{show_number(row['recall_top_1pct']):>12}"
        )

    print(f"\nSaved {output}")

    overall = {
        (row["group"], row["model"]): row
        for row in results
    }

    a = overall[("ALL", "A")]["pr_auc"]
    b = overall[("ALL", "B")]["pr_auc"]

    if a is not None and b is not None:
        print(f"Validation PR-AUC delta (B - A): {b-a:+.6f}")

    print(
        "These are validation ranking metrics, "
        "not calibrated fraud probabilities."
    )


if __name__ == "__main__":
    main()
