"""Frozen BankSim POLICY diagnostic. Never accesses final-test labels."""
import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np

from .banksim_native_fit import FIT_OUTPUT, checksum, split_boundaries
from .banksim_native_prepare import CATEGORIES, OUTPUT, SCHEMA_SHA256
from .independent_evaluation_metrics import score_report

REPORT = Path("reports/banksim_native_policy_diagnostic_v1.json")

# Recorded before examining policy labels.
PIN = {
    "model_sha256": "6ca486570adc58ded65ab916e01b68f802604ac0e8cd706aa9c4bd24778b80f1",
    "calibrator_sha256": "74fe5994b3e9d09ca5e1cdf7865cc3731077300b7655217fbbeb7dd004388e8b",
    "prepared_metadata_sha256": "43933a94f661b614c02f5ee1e140a7655898f4ac97b01e3777ffbe451369513d",
    "prepared_features_sha256": "edd0983d1399beadf2a9aa28286fe3c66ac982b28f7a6e8713947d942d097be3",
    "prepared_development_labels_sha256": "ca047ecbaf04766fdee19d7f8eb1c84be6016917c769f5f45781544e19e6d9b6",
}
THRESHOLD = 0.41746818058367946


def prefix_reviews(scores, cutoff=THRESHOLD):
    """Arrival-order floor(1% of processed transactions) review limit."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Invalid score vector")
    decisions = np.full(len(scores), "ALLOW", dtype="<U6")
    reviewed = 0
    for i, risk in enumerate(scores):
        if risk >= cutoff and reviewed < (i + 1) // 100:
            decisions[i] = "REVIEW"
            reviewed += 1
    return decisions


def diagnostic(prepared=OUTPUT, fit_dir=FIT_OUTPUT, report=REPORT):
    prepared, fit_dir, report = map(Path, (prepared, fit_dir, report))
    if report.exists() or report.is_symlink():
        raise FileExistsError("Refusing to overwrite existing diagnostic report")
    for name, expected in PIN.items():
        path = (fit_dir if name in ("model_sha256", "calibrator_sha256")
                else prepared) / {
                    "model_sha256": "model.joblib",
                    "calibrator_sha256": "calibrator.joblib",
                    "prepared_metadata_sha256": "materialization.json",
                    "prepared_features_sha256": "features.npy",
                    "prepared_development_labels_sha256": "development_labels.npy",
                }[name]
        if path.is_symlink() or checksum(path) != expected:
            raise RuntimeError("Frozen artifact integrity mismatch: " + name)
    meta = json.loads((prepared / "materialization.json").read_text())
    bounds = split_boundaries(meta)
    frozen = json.loads((fit_dir / "fit.json").read_text())
    for k, v in PIN.items():
        if frozen.get(k) != v:
            raise RuntimeError("Fit manifest mismatch: " + k)
    if (frozen.get("selected") != "behavioral"
            or frozen.get("selected_columns") != 24
            or frozen.get("feature_schema_sha256") != SCHEMA_SHA256
            or frozen.get("review_threshold") != THRESHOLD
            or frozen.get("block_enabled") is not False
            or frozen.get("policy_labels_accessed") is not False
            or frozen.get("final_labels_accessed") is not False):
        raise RuntimeError("Unapproved fit/policy configuration")
    features = np.load(prepared / "features.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(prepared / "development_labels.npy", mmap_mode="r", allow_pickle=False)
    if (features.shape != (594643, 24) or features.dtype != np.float32
            or labels.shape != (bounds[-2],) or labels.dtype != np.uint8):
        raise RuntimeError("Unexpected prepared arrays")
    # Trusted local joblib ONLY, verified above. Slice excludes final-test.
    start, end = bounds[3], bounds[4]
    X = np.asarray(features[start:end], dtype=np.float32)
    y = np.asarray(labels[start:end], dtype=np.uint8)
    if not np.isfinite(X).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("Invalid policy diagnostics data")
    model = joblib.load(fit_dir / "model.joblib")
    calibrator = joblib.load(fit_dir / "calibrator.joblib")
    margins = np.asarray(model.decision_function(X), dtype=np.float64).reshape(-1, 1)
    risk = calibrator.predict_proba(margins)[:, 1]
    if not np.isfinite(risk).all() or np.any((risk < 0) | (risk > 1)):
        raise ValueError("Invalid calibrated policy risk")
    decisions = prefix_reviews(risk)
    category = np.asarray(CATEGORIES, dtype=str)[np.argmax(X[:, 1:16], axis=1)]
    if not np.all(np.sum(X[:, 1:16], axis=1) == 1):
        raise RuntimeError("Invalid one-hot categories")
    metrics = score_report(y, risk, decisions, category)
    if metrics["would_block"] != 0 or metrics["reviewed"] > len(y) // 100:
        raise RuntimeError("Safety/capacity violation")
    output = {
        "status": "BANKSIM_POLICY_DIAGNOSTIC_FROZEN_NO_FINAL_LABELS",
        "policy_steps": [144, 161],
        "frozen_sha256": PIN,
        "model": "behavioral",
        "feature_count": 24,
        "review_threshold": THRESHOLD,
        "review_capacity": "prefix_floor_1_percent",
        "final_labels_accessed": False,
        "paySim_model_changed": False,
        "metrics": metrics,
        "limitations": "Within-BankSim policy diagnostic only, not PaySim external validation.",
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation ensures a preexisting result cannot be overwritten.
    with report.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=OUTPUT)
    parser.add_argument("--fit", type=Path, default=FIT_OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    print(json.dumps(diagnostic(args.prepared, args.fit, args.report), indent=2))


if __name__ == "__main__":
    main()
