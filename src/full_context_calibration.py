"""Isolated calibration of the frozen shortcut-reduced logistic model.

No model refit, no policy/holdout access. Locally generated trusted joblib only.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from .full_context import FULL_FEATURE_NAMES
from .full_context_candidates import validate_inputs
from .full_context_materialize import file_sha256
from .full_context_shortcut_sensitivity import GROUPS

MODEL_NAME = "logistic_type_history_network"
INDICES = GROUPS["type_history_network"]
SOURCE = Path("data/processed/full_context_dev_v1")
SPLIT = Path("reports/split_manifest.json")
CANDIDATES = Path("data/processed/full_context_no_current_shortcut_v1")
OUTPUT = Path("data/processed/full_context_calibrated_logistic_v1")


def verify_model(directory):
    directory = Path(directory)
    info = json.loads((directory / "comparison.json").read_text())
    if info.get("status") != "EXPLORATORY_SHORTCUT_SENSITIVITY_NOT_FOR_PROMOTION":
        raise RuntimeError("Unexpected candidate experiment")
    matches = [item for item in info.get("candidates", [])
               if item.get("candidate") == MODEL_NAME]
    if len(matches) != 1 or matches[0].get("feature_names") != [
        FULL_FEATURE_NAMES[i] for i in INDICES
    ]:
        raise RuntimeError("Frozen model/feature contract mismatch")
    path = directory / f"{MODEL_NAME}.joblib"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Model file missing or symlinked")
    digest = file_sha256(path)
    if digest != matches[0].get("model_sha256"):
        raise RuntimeError("Frozen candidate checksum mismatch")
    return path, digest


def fit_sigmoid(margins, labels):
    margins = np.asarray(margins, dtype=np.float64).reshape(-1, 1)
    labels = np.asarray(labels, dtype=np.uint8)
    if (len(labels) != len(margins) or len(labels) < 4 or
            np.unique(labels).size != 2 or not np.isfinite(margins).all()):
        raise ValueError("Invalid calibration labels or margins")
    cal = LogisticRegression(solver="lbfgs", max_iter=500, random_state=2026)
    cal.fit(margins, labels)
    if (not np.isfinite(cal.coef_).all() or
            not np.isfinite(cal.intercept_).all() or cal.coef_[0, 0] <= 0):
        raise RuntimeError("Invalid/reversed calibration ranking")
    return cal


def run(source=SOURCE, split=SPLIT, candidates=CANDIDATES, output=OUTPUT):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Calibration output exists; refusing overwrite")
    model_path, model_digest = verify_model(candidates)
    manifest, _, X, y = validate_inputs(source, split)
    start, end = manifest["split_end_rows"][1:3]
    expected = manifest["partitions"]["calibration"]
    labels = np.asarray(y[start:end], dtype=np.uint8)
    if len(labels) != expected["rows"] or int(labels.sum()) != expected["frauds"]:
        raise RuntimeError("Calibration partition mismatch")
    # Verified before loading: joblib is an executable serialization format.
    model = joblib.load(model_path)
    if (getattr(model, "n_features_in_", None) != len(INDICES)
            or not hasattr(model, "decision_function")):
        raise RuntimeError("Frozen model interface mismatch")
    feature_matrix = np.asarray(X[start:end, INDICES], dtype=np.float32)
    margins = np.asarray(model.decision_function(feature_matrix), dtype=np.float64)
    del feature_matrix
    if margins.shape != (len(labels),) or not np.isfinite(margins).all():
        raise RuntimeError("Nonfinite or malformed margins")
    cal = fit_sigmoid(margins, labels)
    prob = cal.predict_proba(margins.reshape(-1, 1))[:, 1]
    if not np.isfinite(prob).all() or np.any(prob <= 0) or np.any(prob >= 1):
        raise RuntimeError("Invalid calibration outputs")
    report = {
        "status": "PROVISIONAL_CALIBRATION_FIT_ONLY",
        "model_name": MODEL_NAME,
        "model_sha256": model_digest,
        "feature_indices": list(INDICES),
        "feature_names": [FULL_FEATURE_NAMES[i] for i in INDICES],
        "split_manifest_sha256": file_sha256(split),
        "calibration_row_start": start,
        "calibration_row_end_exclusive": end,
        "calibration_rows": len(labels),
        "calibration_frauds": int(labels.sum()),
        "calibration_slope": float(cal.coef_[0, 0]),
        "calibration_intercept": float(cal.intercept_[0]),
        "fit_period_brier": float(brier_score_loss(labels, prob)),
        "fit_period_logloss": float(log_loss(labels, prob)),
        "fit_period_ap": float(average_precision_score(labels, prob)),
        "fit_period_mean_probability": float(prob.mean()),
        "fit_period_fraud_prevalence": float(labels.mean()),
        "block_enabled": False,
        "warning": "Fit-period metrics are NOT independent validation. PaySim only; no policy or final holdout scored.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fgz_cal_", dir=output.parent) as work:
        staging = Path(work)
        joblib.dump(cal, staging / "calibrator.joblib")
        report["calibrator_sha256"] = file_sha256(staging / "calibrator.joblib")
        (staging / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
        os.replace(staging, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(args.source, args.split, args.candidates, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
