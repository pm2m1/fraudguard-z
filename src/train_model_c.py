"""Experimental Model C training; only original training/validation steps.

Usage: python -m src.train_model_c

Never reads calibration, policy, or final-test transaction rows. Does not
modify frozen A/B models, calibrators, policies, or final-test reports.
Scores are uncalibrated and MUST NOT be served as fraud probabilities.
"""

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from .features import TYPE_TO_INDEX
from .model_c import MODEL_C_FEATURE_NAMES, ModelCHistory

DATA = Path("data/raw/PS_20174392719_1491204439457_log.csv")
REPORTS = Path("reports")
MODELS = Path("models")


def load_train_validation(manifest, data_path=DATA):
    train_step, validation_step = manifest["split_end_steps"][:2]
    train_count = manifest["partitions"]["train"]["rows"]
    validation_count = manifest["partitions"]["validation"]["rows"]
    total = train_count + validation_count
    X = np.empty((total, len(MODEL_C_FEATURE_NAMES)), dtype=np.float32)
    labels = np.empty(total, dtype=np.uint8)
    kinds = np.empty(total, dtype=np.uint8)
    history = ModelCHistory()
    previous_step = -1
    count = 0

    with Path(data_path).open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            step = int(row["step"])
            if step > validation_step:
                break  # Calibration and later data are never read.
            if step < previous_step:
                raise ValueError("Dataset is not step-ordered")
            previous_step = step
            if count >= total:
                raise ValueError("Unexpected additional development rows")
            fraud = row["isFraud"]
            if fraud not in ("0", "1"):
                raise ValueError("Invalid label")
            X[count] = history.features(row)
            labels[count] = int(fraud)
            kinds[count] = TYPE_TO_INDEX[row["type"]]
            history.observe(row)
            count += 1
            if count % 500_000 == 0:
                print(f"Processed {count:,} rows", flush=True)
    if count != total:
        raise ValueError(f"Expected {total:,} rows, got {count:,}")
    if int(labels[:train_count].sum()) != manifest["partitions"]["train"]["frauds"]:
        raise ValueError("Training labels do not match manifest")
    if int(labels[train_count:].sum()) != manifest["partitions"]["validation"]["frauds"]:
        raise ValueError("Validation labels do not match manifest")
    return X, labels, kinds, train_count


def review_metrics(labels, scores, fraction=0.01):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim != 1 or labels.shape != scores.shape or len(labels) == 0:
        raise ValueError("Labels and scores must be aligned nonempty vectors")
    if not np.all(np.isfinite(scores)) or not np.all(np.isin(labels, (0, 1))):
        raise ValueError("Invalid predictions or labels")
    k = max(1, int(np.ceil(fraction * len(labels))))
    # Sort ties by original row index, reproducibly.
    order = np.lexsort((np.arange(len(scores)), -scores))
    selected = np.zeros(len(labels), dtype=bool)
    selected[order[:k]] = True
    tp = int(labels[selected].sum())
    frauds = int(labels.sum())
    return selected, {
        "rows": int(len(labels)),
        "frauds": frauds,
        "reviews": k,
        "true_positives": tp,
        "false_positives": k - tp,
        "precision": tp / k,
        "recall": tp / frauds if frauds else None,
        "pr_auc": float(average_precision_score(labels, scores)) if 0 < frauds < len(labels) else None,
    }


def explain_ranking(labels, types, scores):
    selected, all_metrics = review_metrics(labels, scores)
    report = {"overall": all_metrics, "by_type_at_global_1pct": {}}
    for name in ("TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"):
        mask = types == TYPE_TO_INDEX[name]
        subgroup = labels[mask]
        group_selected = selected[mask]
        frauds = int(subgroup.sum())
        reviews = int(group_selected.sum())
        tp = int(subgroup[group_selected].sum())
        report["by_type_at_global_1pct"][name] = {
            "rows": int(mask.sum()),
            "frauds": frauds,
            "reviews": reviews,
            "true_positives": tp,
            "false_positives": reviews - tp,
            "recall": tp / frauds if frauds else None,
            "precision": tp / reviews if reviews else None,
        }
    return report


def main():
    manifest = json.loads((REPORTS / "split_manifest.json").read_text())
    if manifest["split_end_steps"][:2] != [238, 306]:
        raise RuntimeError("Unexpected training/validation split")

    print("Preparing strictly past-only Model C features through step 306...")
    X, y, types, train_count = load_train_validation(manifest)
    train_X, validation_X = X[:train_count], X[train_count:]
    train_y, validation_y = y[:train_count], y[train_count:]
    validation_types = types[train_count:]

    print("Training histogram gradient boosting (experimental)...", flush=True)
    candidate = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=15,
        max_bins=128,
        learning_rate=0.07,
        l2_regularization=10.0,
        class_weight="balanced",
        early_stopping=False,
        random_state=42,
    )
    candidate.fit(train_X, train_y)
    candidate_scores = candidate.predict_proba(validation_X)[:, 1]

    baseline = joblib.load(MODELS / "model_b.joblib")
    baseline_scores = baseline.decision_function(validation_X[:, :8])

    baseline_report = explain_ranking(validation_y, validation_types, baseline_scores)
    candidate_report = explain_ranking(validation_y, validation_types, candidate_scores)
    comparison = {
        "status": "EXPERIMENTAL - not deployed or calibrated",
        "partitions": "Train steps 1-238; validation steps 239-306 only",
        "warning": (
            "Original final holdout already inspected. These validation results "
            "are development diagnostics, not a fresh independent test. "
            "Retrospective global top-1% ranking is not a streaming policy."
        ),
        "model_c_schema": list(MODEL_C_FEATURE_NAMES),
        "model_b": baseline_report,
        "model_c": candidate_report,
    }
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    out = REPORTS / "model_c_validation.json"
    out.write_text(json.dumps(comparison, indent=2) + "\n")
    joblib.dump(candidate, MODELS / "model_c_experimental.joblib")

    for name, report in (("B", baseline_report), ("C", candidate_report)):
        summary = report["overall"]
        print(f"Model {name}: PR-AUC={summary['pr_auc']:.6f}, "
              f"frauds@1%={summary['true_positives']}, "
              f"recall@1%={summary['recall']:.2%}")
        for type_name in ("TRANSFER", "CASH_OUT"):
            sub = report["by_type_at_global_1pct"][type_name]
            print(f"  {type_name}: {sub['true_positives']}/{sub['frauds']} frauds, "
                  f"{sub['false_positives']} false alerts")
    print(f"Saved {out}")
    print("Model C is an experiment; production Model B remains unchanged.")


if __name__ == "__main__":
    main()
