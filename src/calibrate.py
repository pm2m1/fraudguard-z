import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from .features import RecipientHistory, TYPE_TO_INDEX, make_feature_row


DATA_FILE = Path("data/raw/PS_20174392719_1491204439457_log.csv")
REPORTS = Path("reports")
MODELS = Path("models")


def load_calibration_and_policy_rows(manifest):
    validation_end = manifest["split_end_steps"][1]
    calibration_end = manifest["split_end_steps"][2]
    development_end = manifest["dev_max_step"]

    n_cal = manifest["partitions"]["calibration"]["rows"]
    n_policy = manifest["partitions"]["policy"]["rows"]

    x_cal = np.empty((n_cal, 8), dtype=np.float32)
    y_cal = np.empty(n_cal, dtype=np.uint8)

    x_policy = np.empty((n_policy, 8), dtype=np.float32)
    y_policy = np.empty(n_policy, dtype=np.uint8)
    type_policy = np.empty(n_policy, dtype=np.uint8)

    history = RecipientHistory(window_steps=6)
    cal_index = 0
    policy_index = 0

    with DATA_FILE.open(newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            step = int(row["step"])
            if step > development_end:
                break

            recipient = row["nameDest"]

            if step <= validation_end:
                history.observe(recipient, step)
                continue

            features = make_feature_row(row, history)
            fraud = int(row["isFraud"])

            if fraud not in (0, 1):
                raise ValueError("Invalid fraud label")

            if step <= calibration_end:
                if cal_index >= n_cal:
                    raise ValueError("Calibration row count exceeded")

                x_cal[cal_index] = features
                y_cal[cal_index] = fraud
                cal_index += 1
            else:
                if policy_index >= n_policy:
                    raise ValueError("Policy row count exceeded")

                x_policy[policy_index] = features
                y_policy[policy_index] = fraud
                type_policy[policy_index] = TYPE_TO_INDEX[row["type"]]
                policy_index += 1

            history.observe(recipient, step)

    if cal_index != n_cal or policy_index != n_policy:
        raise ValueError("Row counts do not match training manifest")

    return x_cal, y_cal, x_policy, y_policy, type_policy


def calibrate_model(model, x_cal, y_cal, x_policy):
    # Calibrate the model's raw decision scores, not its predictions.
    margins_cal = model.decision_function(x_cal).reshape(-1, 1)

    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
    )
    calibrator.fit(margins_cal, y_cal)

    if calibrator.coef_[0, 0] <= 0:
        raise ValueError("Calibration reversed the score ranking")

    margins_policy = model.decision_function(
        x_policy
    ).reshape(-1, 1)

    probabilities = calibrator.predict_proba(
        margins_policy
    )[:, 1]

    return calibrator, probabilities


def summarize(y, probabilities, name, group):
    count = len(y)
    frauds = int(y.sum())

    k = max(1, int(np.ceil(count * 0.01)))
    top = np.argpartition(probabilities, -k)[-k:]

    detected = int(y[top].sum())

    return {
        "model": name,
        "group": group,
        "rows": count,
        "frauds": frauds,
        "pr_auc": (
            float(average_precision_score(y, probabilities))
            if 0 < frauds < count else None
        ),
        "brier": float(brier_score_loss(y, probabilities)),
        "mean_risk": float(probabilities.mean()),
        "precision_top_1pct": detected / k,
        "recall_top_1pct": detected / frauds if frauds else None,
    }


def main():
    with (REPORTS / "split_manifest.json").open() as file:
        manifest = json.load(file)

    print("Restoring recipient history and preparing reserved rows...")

    x_cal, y_cal, x_policy, y_policy, types = (
        load_calibration_and_policy_rows(manifest)
    )

    print(f"Calibration: {len(y_cal):,} rows, {int(y_cal.sum())} frauds")
    print(f"Policy: {len(y_policy):,} rows, {int(y_policy.sum())} frauds")

    probabilities = {}

    for name, columns in (("a", 7), ("b", 8)):
        print(f"Calibrating Model {name.upper()}...")

        model = joblib.load(MODELS / f"model_{name}.joblib")

        calibrator, scores = calibrate_model(
            model,
            x_cal[:, :columns],
            y_cal,
            x_policy[:, :columns],
        )

        joblib.dump(
            calibrator,
            MODELS / f"calibrator_{name}.joblib",
        )

        probabilities[name] = scores

    masks = {
        "ALL": np.ones(len(y_policy), dtype=bool),
        "TRANSFER": types == TYPE_TO_INDEX["TRANSFER"],
        "CASH_OUT": types == TYPE_TO_INDEX["CASH_OUT"],
    }

    results = []

    for group, mask in masks.items():
        for name in ("a", "b"):
            results.append(
                summarize(
                    y_policy[mask],
                    probabilities[name][mask],
                    name.upper(),
                    group,
                )
            )

    import csv as csv_writer

    output = REPORTS / "calibration_policy.csv"
    with output.open("w", newline="") as file:
        writer = csv_writer.DictWriter(
            file,
            fieldnames=list(results[0]),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n=== POLICY PERIOD RESULTS ===")
    print(
        f"{'Group':<12} {'Model':<7} "
        f"{'PR-AUC':>10} {'P@1%':>10} "
        f"{'R@1%':>10} {'Brier':>12}"
    )

    for row in results:
        values = [
            "N/A" if row[k] is None else f"{row[k]:.6f}"
            for k in (
                "pr_auc",
                "precision_top_1pct",
                "recall_top_1pct",
                "brier",
            )
        ]

        print(
            f"{row['group']:<12} {row['model']:<7} "
            f"{values[0]:>10} {values[1]:>10} "
            f"{values[2]:>10} {values[3]:>12}"
        )

    # Set a provisional REVIEW threshold for Model B.
    # Keep at most 1% of policy-period transactions.
    scores_b = probabilities["b"]
    k = max(1, int(np.ceil(len(scores_b) * 0.01)))

    threshold = float(np.partition(scores_b, -k)[-k])

    # Do not exceed the budget if predictions tie at the cutoff.
    if int(np.count_nonzero(scores_b >= threshold)) > k:
        threshold = float(np.nextafter(threshold, np.inf))

    selected = scores_b >= threshold

    policy = {
        "selected_model": "B",
        "review_threshold": threshold,
        "review_budget_fraction": 0.01,
        "review_count": int(selected.sum()),
        "frauds_in_review": int(y_policy[selected].sum()),
        "policy_rows": len(y_policy),
        "block_enabled": False,
        "calibration_method": "sigmoid_on_decision_function",
        "final_test_evaluated": False,
    }

    with (REPORTS / "review_policy.json").open("w") as file:
        json.dump(policy, file, indent=2)

    print("\nModel B provisional review threshold:", threshold)
    print("Selected for review:", policy["review_count"])
    print("Frauds in review:", policy["frauds_in_review"])
    print("Automatic blocking: disabled")
    print("Final test remains untouched.")
    print(f"Saved {output}")
    print("Saved reports/review_policy.json")


if __name__ == "__main__":
    main()
