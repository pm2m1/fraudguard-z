"""Isolated, conservative PaySim policy fit. BLOCK is always disabled.

Only the frozen policy partition is read; output is simulation-only and
not independent validation or a production payment decision system.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

from .full_context_calibration import INDICES, MODEL_NAME, verify_model
from .full_context_candidates import validate_inputs
from .full_context_materialize import file_sha256
from .full_context_journal import FEATURE_SCHEMA_SHA256
from .features import TRANSACTION_TYPES

SOURCE = Path("data/processed/full_context_dev_v1")
SPLIT = Path("reports/split_manifest.json")
CANDIDATES = Path("data/processed/full_context_no_current_shortcut_v1")
CALIBRATION = Path("data/processed/full_context_calibrated_logistic_v1")
OUTPUT = Path("data/processed/full_context_policy_logistic_v1")
REVIEW_RATE = 0.01


def frozen_threshold(scores, capacity):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid policy scores")
    if not 0 <= capacity <= len(scores):
        raise ValueError("Invalid review capacity")
    if capacity == 0:
        return None
    threshold = float(np.partition(scores, -capacity)[-capacity])
    if int(np.count_nonzero(scores >= threshold)) > capacity:
        threshold = float(np.nextafter(threshold, np.inf))
    return threshold


def decisions_with_prefix_cap(scores, threshold, rate=REVIEW_RATE):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all() or scores.ndim != 1:
        raise ValueError("Invalid scores")
    if not 0 < rate < 1:
        raise ValueError("Invalid review rate")
    reviewed = np.zeros(len(scores), dtype=bool)
    used = 0
    if threshold is None:
        return reviewed
    if not np.isfinite(threshold):
        raise ValueError("Invalid threshold")
    for i, score in enumerate(scores, start=1):
        if score >= threshold and used < int(np.floor(rate*i)):
            reviewed[i-1] = True
            used += 1
    return reviewed


def summarize(y, reviewed):
    n = int(len(y))
    tp = int(np.count_nonzero(y[reviewed]))
    count = int(np.count_nonzero(reviewed))
    positives = int(np.count_nonzero(y))
    return {
        "rows": n, "frauds": positives, "reviews": count,
        "true_positives": tp, "false_positives": count-tp,
        "precision": tp/count if count else None,
        "recall": tp/positives if positives else None,
    }


def run(source=SOURCE, split=SPLIT, candidates=CANDIDATES,
        calibration=CALIBRATION, output=OUTPUT):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Policy output exists; refusing overwrite")
    model_path, model_sha = verify_model(candidates)
    calibration = Path(calibration)
    info = json.loads((calibration / "calibration.json").read_text())
    if (info.get("status") != "PROVISIONAL_CALIBRATION_FIT_ONLY"
            or info.get("model_name") != MODEL_NAME
            or info.get("model_sha256") != model_sha
            or info.get("feature_indices") != list(INDICES)
            or info.get("feature_names") is None
            or info.get("block_enabled") is not False
            or info.get("split_manifest_sha256") != file_sha256(split)):
        raise RuntimeError("Provisional calibrator provenance mismatch")
    cal_path = calibration / "calibrator.joblib"
    if cal_path.is_symlink() or file_sha256(cal_path) != info.get("calibrator_sha256"):
        raise RuntimeError("Calibrator checksum mismatch")

    manifest, _, X, y = validate_inputs(source, split)
    start = manifest["split_end_rows"][2]
    end = manifest["development_row_count"]
    expected = manifest["partitions"]["policy"]
    if end-start != expected["rows"]:
        raise RuntimeError("Policy split mismatch")
    # No access to training, validation or calibration labels here.
    labels = np.asarray(y[start:end], dtype=np.uint8)
    if int(np.count_nonzero(labels)) != expected["frauds"]:
        raise RuntimeError("Policy fraud count mismatch")
    model = joblib.load(model_path)
    calibrator = joblib.load(cal_path)
    if (getattr(model, "n_features_in_", None) != len(INDICES)
            or getattr(calibrator, "n_features_in_", None) != 1
            or float(calibrator.coef_[0, 0]) <= 0):
        raise RuntimeError("Model/calibrator interface mismatch")
    feature_slice = X[start:end]
    features = np.asarray(feature_slice[:, INDICES], dtype=np.float32)
    margins = np.asarray(model.decision_function(features), dtype=np.float64)
    del features
    if margins.shape != labels.shape or not np.isfinite(margins).all():
        raise RuntimeError("Invalid scoring margins")
    risks = np.asarray(
        calibrator.predict_proba(margins.reshape(-1, 1))[:, 1],
        dtype=np.float64
    )
    if (risks.shape != labels.shape or not np.isfinite(risks).all()
            or np.any(risks < 0) or np.any(risks > 1)):
        raise RuntimeError("Invalid calibrated scores")
    capacity = int(np.floor(REVIEW_RATE*len(risks)))
    threshold = frozen_threshold(risks, capacity)
    eligible = risks >= threshold if threshold is not None else np.zeros(len(risks), bool)
    review = decisions_with_prefix_cap(risks, threshold)
    if review.sum() > capacity or np.any(review & ~eligible):
        raise RuntimeError("Review budget or threshold violated")
    types = np.argmax(feature_slice[:, 1:6], axis=1)
    if np.any(np.count_nonzero(feature_slice[:, 1:6], axis=1) != 1):
        raise RuntimeError("Malformed transaction type one-hot")
    by_type = {
        kind: summarize(labels[types == i], review[types == i])
        for i, kind in enumerate(TRANSACTION_TYPES)
    }
    result = {
        "status": "EXPLORATORY_POLICY_TUNING_SIMULATION_ONLY",
        "selected_model": MODEL_NAME,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "feature_indices": list(INDICES),
        "model_sha256": model_sha,
        "calibrator_sha256": info["calibrator_sha256"],
        "split_manifest_sha256": file_sha256(split),
        "policy_start_row": start, "policy_end_row_exclusive": end,
        "review_budget_fraction": REVIEW_RATE,
        "review_capacity": capacity,
        "review_threshold": threshold,
        "eligible_by_threshold": int(eligible.sum()),
        "prefix_cap": "floor(0.01 * cumulative_event_count)",
        "actual_policy_outcomes": summarize(labels, review),
        "by_transaction_type": by_type,
        "policy_period_ap": float(average_precision_score(labels, risks)),
        "policy_period_brier": float(brier_score_loss(labels, risks)),
        "policy_period_mean_risk": float(risks.mean()),
        "policy_period_fraud_prevalence": float(labels.mean()),
        "block_enabled": False, "block_threshold": None,
        "blocks": 0, "false_blocks": 0,
        "block_gate": "BLOCK_DISABLED_UNVALIDATED",
        "warning": "Threshold selected on this same policy partition, not independent performance. No real payment blocking. Prior final holdout not inspected.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fgz_policy_", dir=output.parent) as td:
        stage = Path(td)
        (stage / "policy.json").write_text(json.dumps(result, indent=2)+"\n")
        os.replace(stage, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args.source, args.split, args.candidates, args.calibration, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
