"""Controlled Model C ablations on original training and validation only.

Never reads beyond step 306; does not alter production artifacts.
Predicted probabilities are uncalibrated ranking scores.
"""
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .model_c import MODEL_C_FEATURE_NAMES
from .model_c_shortcut_audit import exact_positive_balance_match
from .train_model_c import load_train_validation, explain_ranking

ROOT = Path(".")
DATA = ROOT / "data/raw/PS_20174392719_1491204439457_log.csv"
MANIFEST = ROOT / "reports/split_manifest.json"
OUT = ROOT / "reports/model_c_ablations_validation.json"

# First eight columns reproduce Model B; 8-10 encode current amount/balance
# relationships; 11-16 are strictly past-only sender/recipient behavior.
VARIANTS = {
    "C0_non_linear_original": tuple(range(8)),
    "C1_original_plus_history": tuple(range(8)) + tuple(range(11, 17)),
    "C2_type_plus_history_only": tuple(range(1, 6)) + tuple(range(11, 17)),
    "C3_full": tuple(range(17)),
}

def validate_variants():
    all_features = set(range(len(MODEL_C_FEATURE_NAMES)))
    for name, columns in VARIANTS.items():
        if not columns or len(set(columns)) != len(columns) or not set(columns) <= all_features:
            raise ValueError(f"Invalid feature selection: {name}")
    if set(VARIANTS["C1_original_plus_history"]) & {8, 9, 10}:
        raise ValueError("History variant includes explicit shortcut features")
    if set(VARIANTS["C2_type_plus_history_only"]) & {0, 6, 8, 9, 10}:
        raise ValueError("History-only variant includes current amount/balance")

def validation_rule(path, manifest):
    """Exact sender-balance rule restricted to validation rows."""
    train_step, val_step = manifest["split_end_steps"][:2]
    values = []
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            step = int(row["step"])
            if step > val_step:
                break
            if step > train_step:
                match = (
                    row["type"] in ("TRANSFER", "CASH_OUT")
                    and exact_positive_balance_match(
                        row["amount"], row["oldbalanceOrg"]
                    )
                )
                values.append(bool(match))
    expected = manifest["partitions"]["validation"]["rows"]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} validation rule scores, got {len(values)}")
    return np.asarray(values, dtype=bool)

def rule_report(labels, types, flags):
    labels = np.asarray(labels, dtype=np.uint8)
    flags = np.asarray(flags, dtype=bool)
    if labels.shape != flags.shape:
        raise ValueError("Rule/label length mismatch")
    tp = int(np.sum(labels[flags]))
    reviewed = int(np.sum(flags))
    total_frauds = int(np.sum(labels))
    return {
        "reviewed": reviewed,
        "frauds_detected": tp,
        "false_alerts": reviewed - tp,
        "precision": tp / reviewed if reviewed else None,
        "recall": tp / total_frauds if total_frauds else None,
        "note": "Fixed deterministic heuristic, not a top-1% selection."
    }

def fit_candidate(x_train, y_train, columns):
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
    model.fit(x_train[:, columns], y_train)
    return model

def main():
    validate_variants()
    manifest = json.loads(MANIFEST.read_text())
    if manifest["split_end_steps"][:2] != [238, 306]:
        raise RuntimeError("Unexpected split boundaries")
    print("Loading past-only features through validation step 306...", flush=True)
    X, y, kinds, train_n = load_train_validation(manifest, DATA)
    x_train, x_val = X[:train_n], X[train_n:]
    y_train, y_val = y[:train_n], y[train_n:]
    val_types = kinds[train_n:]

    baseline = joblib.load(ROOT / "models/model_b.joblib")
    reference = explain_ranking(y_val, val_types,
                                baseline.decision_function(x_val[:, :8]))
    report = {
        "status": "DEVELOPMENT-ONLY; uncalibrated; not deployed",
        "splits": "Train 1-238; validation 239-306; no later rows read",
        "warning": ("Repeated analysis on the original validation set is exploratory. "
                    "The earlier final holdout has already been inspected. "
                    "Top-1% is retrospective, not a streaming policy."),
        "reference_model_b": reference,
        "feature_names": list(MODEL_C_FEATURE_NAMES),
        "variants": {},
    }
    for name, columns in VARIANTS.items():
        print(f"Fitting {name} ({len(columns)} features)...", flush=True)
        model = fit_candidate(x_train, y_train, columns)
        scores = model.predict_proba(x_val[:, columns])[:, 1]
        report["variants"][name] = {
            "features": [MODEL_C_FEATURE_NAMES[i] for i in columns],
            **explain_ranking(y_val, val_types, scores),
        }
        summary = report["variants"][name]["overall"]
        print(f"  PR-AUC {summary['pr_auc']:.6f}; frauds at 1% "
              f"{summary['true_positives']}/{summary['frauds']}")

    flags = validation_rule(DATA, manifest)
    report["exact_balance_rule"] = rule_report(y_val, val_types, flags)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {OUT}")
    print("Production model, policy, and original holdout unchanged.")

if __name__ == "__main__":
    main()
