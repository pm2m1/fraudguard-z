"""Train FraudGuard Z transaction baseline vs recipient-history model."""

import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import (
    RecipientHistory,
    make_feature_row,
    TYPE_TO_INDEX,
    BASE_FEATURE_NAMES,
    MODEL_B_FEATURE_NAMES,
)


DATA_PATH = Path(
    "data/raw/PS_20174392719_1491204439457_log.csv"
)
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

# Never inspect or train on rows after this step during development.
DEV_MAX_STEP = 372

SPLIT_FRACTIONS = (0.60, 0.80, 0.90)


def count_rows_by_step(path):
    counts = Counter()
    previous_step = -1

    with path.open(newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            step = int(row["step"])

            if step > DEV_MAX_STEP:
                break

            if step < previous_step:
                raise ValueError("CSV is not chronologically ordered")

            previous_step = step
            counts[step] += 1

    return counts


def select_split_boundaries(counts):
    steps = sorted(counts)

    if len(steps) < 4:
        raise ValueError("Need at least four nonempty steps")

    cumulative = np.cumsum(
        [counts[step] for step in steps]
    )
    total = int(cumulative[-1])

    indices = []
    previous = -1

    for i, fraction in enumerate(SPLIT_FRACTIONS):
        proposed = int(
            np.searchsorted(cumulative, total * fraction)
        )

        # Reserve at least one step for each later partition.
        max_index = len(steps) - (4 - i)
        index = max(previous + 1, proposed)

        if index > max_index:
            raise ValueError("Cannot construct chronological splits")

        indices.append(index)
        previous = index

    split_steps = [steps[i] for i in indices]
    split_ends = [int(cumulative[i]) for i in indices]

    return split_steps, split_ends, total


def load_development_features(path, row_count):
    X = np.empty((row_count, 8), dtype=np.float32)
    y = np.empty(row_count, dtype=np.uint8)
    groups = np.empty(row_count, dtype=np.uint8)

    history = RecipientHistory(window_steps=6)
    processed = 0

    with path.open(newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            step = int(row["step"])

            if step > DEV_MAX_STEP:
                break

            if processed >= row_count:
                raise ValueError("More rows than expected")

            # Compute features before observing this row.
            X[processed] = make_feature_row(row, history)

            label = row["isFraud"]
            if label not in ("0", "1"):
                raise ValueError(f"Invalid fraud label: {label}")

            y[processed] = int(label)
            groups[processed] = TYPE_TO_INDEX[row["type"]]

            # Historical observation happens after extraction.
            history.observe(row["nameDest"], step)

            processed += 1

            if processed % 500_000 == 0:
                print(f"Feature extraction: {processed:,} rows")

    if processed != row_count:
        raise ValueError(
            f"Expected {row_count} rows; found {processed}"
        )

    return X, y, groups


def fit_model(X_train, y_train):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=300,
            random_state=42,
        ),
    )

    model.fit(X_train, y_train)
    return model


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    print("Scanning development-period step counts...")
    counts = count_rows_by_step(DATA_PATH)

    split_steps, split_ends, total = (
        select_split_boundaries(counts)
    )

    train_end, val_end, cal_end = split_ends

    print("Development rows:", f"{total:,}")
    print("Train end step:", split_steps[0])
    print("Validation end step:", split_steps[1])
    print("Calibration end step:", split_steps[2])
    print("Policy tuning end step:", DEV_MAX_STEP)
    print("Final holdout begins at step:", DEV_MAX_STEP + 1)

    print("\nGenerating historical features...")
    X, y, groups = load_development_features(
        DATA_PATH, total
    )

    partitions = {
        "train": (0, train_end),
        "validation": (train_end, val_end),
        "calibration": (val_end, cal_end),
        "policy": (cal_end, total),
    }

    for name, (start, end) in partitions.items():
        positives = int(y[start:end].sum())

        print(
            f"{name}: {end-start:,} rows, "
            f"{positives:,} frauds"
        )

        if name in ("train", "validation"):
            if positives == 0 or positives == end-start:
                raise ValueError(
                    f"{name} does not have both classes"
                )

    print("\nTraining Model A...")
    model_a = fit_model(
        X[:train_end, :7],
        y[:train_end],
    )

    scores_a = model_a.predict_proba(
        X[train_end:val_end, :7]
    )[:, 1].astype(np.float32)

    joblib.dump(model_a, MODEL_DIR / "model_a.joblib")

    print("Training Model B...")
    model_b = fit_model(
        X[:train_end, :8],
        y[:train_end],
    )

    scores_b = model_b.predict_proba(
        X[train_end:val_end, :8]
    )[:, 1].astype(np.float32)

    joblib.dump(model_b, MODEL_DIR / "model_b.joblib")

    np.savez_compressed(
        REPORT_DIR / "validation_predictions.npz",
        labels=y[train_end:val_end],
        transaction_types=groups[train_end:val_end],
        scores_a=scores_a,
        scores_b=scores_b,
    )

    manifest = {
        "schema_a": list(BASE_FEATURE_NAMES),
        "schema_b": list(MODEL_B_FEATURE_NAMES),
        "dev_max_step": DEV_MAX_STEP,
        "final_holdout_start_step": DEV_MAX_STEP + 1,
        "split_end_steps": split_steps,
        "split_end_rows": split_ends,
        "development_row_count": total,
        "partitions": {
            name: {
                "rows": end - start,
                "frauds": int(y[start:end].sum()),
            }
            for name, (start, end) in partitions.items()
        },
        "model": "StandardScaler + balanced LogisticRegression",
        "note": (
            "Validation scores are uncalibrated. "
            "Calibration/policy rows were not used for fitting."
        ),
    }

    with (REPORT_DIR / "split_manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)

    print("\nTraining complete.")
    print("Saved models/model_a.joblib")
    print("Saved models/model_b.joblib")
    print("Saved reports/validation_predictions.npz")
    print("Saved reports/split_manifest.json")
    print("Calibration/policy/final test scores not evaluated.")


if __name__ == "__main__":
    main()
