import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .features import (
    MODEL_B_FEATURE_NAMES,
    RecipientHistory,
    TYPE_TO_INDEX,
    make_feature_row,
)


DATA = Path("data/raw/PS_20174392719_1491204439457_log.csv")
REPORTS = Path("reports")
MODELS = Path("models")

EXPECTED_TEST_ROWS = 1_076_732


def load_test_data(start_step):
    X = np.empty((EXPECTED_TEST_ROWS, 8), dtype=np.float32)
    y = np.empty(EXPECTED_TEST_ROWS, dtype=np.uint8)
    types = np.empty(EXPECTED_TEST_ROWS, dtype=np.uint8)

    history = RecipientHistory(window_steps=6)
    count = 0

    with DATA.open(newline="") as file:
        for row in csv.DictReader(file):
            step = int(row["step"])
            recipient = row["nameDest"]

            if step < start_step:
                history.observe(recipient, step)
                continue

            if count >= EXPECTED_TEST_ROWS:
                raise ValueError("Unexpected additional test rows")

            X[count] = make_feature_row(row, history)

            label = row["isFraud"]
            if label not in ("0", "1"):
                raise ValueError("Invalid fraud label")

            y[count] = int(label)
            types[count] = TYPE_TO_INDEX[row["type"]]

            history.observe(recipient, step)
            count += 1

            if count % 250_000 == 0:
                print(f"Test rows processed: {count:,}")

    if count != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_ROWS:,}, got {count:,}"
        )

    return X, y, types


def calculate_metrics(labels, scores):
    count = len(labels)
    frauds = int(labels.sum())
    legitimate = count - frauds

    if not 0 < frauds < count:
        return None

    k = max(1, int(np.ceil(count * 0.01)))
    top = np.argpartition(scores, -k)[-k:]

    detected = int(labels[top].sum())

    return {
        "rows": count,
        "frauds": frauds,
        "pr_auc": float(
            average_precision_score(labels, scores)
        ),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "precision_top_1pct": detected / k,
        "recall_top_1pct": detected / frauds,
        "false_positives_top_1pct": k - detected,
        "fpr_top_1pct": (k - detected) / legitimate,
    }


def main():
    output = REPORTS / "final_test_results.json"

    if output.exists():
        raise FileExistsError(
            "Final test report already exists. Do not rerun "
            "or retune against the holdout."
        )

    with (REPORTS / "split_manifest.json").open() as file:
        manifest = json.load(file)

    policy_file = REPORTS / "review_policy.json"
    with policy_file.open() as file:
        policy = json.load(file)

    if manifest["schema_b"] != list(MODEL_B_FEATURE_NAMES):
        raise ValueError("Feature schema does not match the model")

    if policy["selected_model"] != "B":
        raise ValueError("Expected frozen Model B policy")

    if policy["block_enabled"]:
        raise ValueError("Automatic blocking must remain disabled")

    if policy["final_test_evaluated"]:
        raise ValueError("Final test already marked as evaluated")

    if manifest["final_holdout_start_step"] != 373:
        raise ValueError("Unexpected holdout boundary")

    print("Replaying transaction history...")
    X, y, types = load_test_data(373)

    probabilities = {}

    for name, columns in (("A", 7), ("B", 8)):
        print(f"Scoring Model {name}...")

        model = joblib.load(
            MODELS / f"model_{name.lower()}.joblib"
        )
        calibrator = joblib.load(
            MODELS / f"calibrator_{name.lower()}.joblib"
        )

        margins = model.decision_function(
            X[:, :columns]
        ).reshape(-1, 1)

        probabilities[name] = calibrator.predict_proba(
            margins
        )[:, 1]

    masks = {
        "ALL": np.ones(len(y), dtype=bool),
        "TRANSFER": types == TYPE_TO_INDEX["TRANSFER"],
        "CASH_OUT": types == TYPE_TO_INDEX["CASH_OUT"],
    }

    results = {}

    print("\n=== FINAL HOLDOUT RESULTS ===")
    print(
        f"{'Group':<12} {'Model':<7} "
        f"{'PR-AUC':>10} {'P@1%':>10} {'R@1%':>10}"
    )

    for group, mask in masks.items():
        results[group] = {}

        for name in ("A", "B"):
            metrics = calculate_metrics(
                y[mask], probabilities[name][mask]
            )
            results[group][name] = metrics

            if metrics is not None:
                print(
                    f"{group:<12} {name:<7} "
                    f"{metrics['pr_auc']:>10.6f} "
                    f"{metrics['precision_top_1pct']:>10.6f} "
                    f"{metrics['recall_top_1pct']:>10.6f}"
                )

    # Evaluate the already-frozen policy threshold.
    threshold = float(policy["review_threshold"])
    selected = probabilities["B"] >= threshold

    review_count = int(selected.sum())
    detected = int(y[selected].sum())
    total_frauds = int(y.sum())

    deployment_metrics = {
        "threshold": threshold,
        "review_count": review_count,
        "review_rate": review_count / len(y),
        "frauds_detected": detected,
        "frauds_missed": total_frauds - detected,
        "false_positives": review_count - detected,
        "precision": (
            detected / review_count if review_count else 0.0
        ),
        "recall": (
            detected / total_frauds if total_frauds else 0.0
        ),
        "automatic_blocking": False,
    }

    report = {
        "test_start_step": 373,
        "test_rows": len(y),
        "model_comparison": results,
        "frozen_policy": deployment_metrics,
        "policy_sha256": hashlib.sha256(
            policy_file.read_bytes()
        ).hexdigest(),
    }

    with output.open("w") as file:
        json.dump(report, file, indent=2)

    print("\n=== FROZEN REVIEW POLICY ON FINAL HOLDOUT ===")
    print(f"Review threshold: {threshold:.8f}")
    print(f"Transactions reviewed: {review_count:,}")
    print(f"Actual review rate: {review_count / len(y):.4%}")
    print(f"Frauds detected: {detected:,}")
    print(f"Frauds missed: {total_frauds - detected:,}")
    print(f"False positives: {review_count - detected:,}")
    print(f"Precision: {deployment_metrics['precision']:.4%}")
    print(f"Recall: {deployment_metrics['recall']:.4%}")
    print("Automatic blocking: disabled")
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
