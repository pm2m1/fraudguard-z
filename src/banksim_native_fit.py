"""Fit a separate BankSim model; do not read policy/final labels.

Uses the frozen chronological partitions and model choices in
notes/banksim_native_benchmark_protocol.md. Local artifacts are trusted-only
joblib files; never load them from untrusted sources.
"""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .banksim_native_prepare import (
    FEATURE_NAMES, OUTPUT, PARTITIONS, SCHEMA_SHA256, SOURCE_SHA256,
)

FIT_OUTPUT = Path("data/processed/banksim_native_fit_v1")
EXPECTED_ROWS = 594643


def checksum(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def split_boundaries(meta):
    if meta["feature_schema_sha256"] != SCHEMA_SHA256:
        raise RuntimeError("Unexpected feature schema")
    if meta["feature_names"] != list(FEATURE_NAMES):
        raise RuntimeError("Unexpected feature ordering")
    if meta["source_sha256"] != SOURCE_SHA256:
        raise RuntimeError("Unexpected BankSim source")
    if meta["rows"] != EXPECTED_ROWS or meta["feature_shape"] != [EXPECTED_ROWS, 24]:
        raise RuntimeError("Unexpected BankSim row/feature count")
    counts = meta["partition_rows"]
    if set(counts) != set(PARTITIONS) or any(
        not isinstance(counts[k], int) or counts[k] <= 0 for k in PARTITIONS
    ) or sum(counts.values()) != EXPECTED_ROWS:
        raise RuntimeError("Invalid fixed partition sizes")
    boundaries = np.cumsum([0] + [counts[k] for k in PARTITIONS]).tolist()
    if meta["development_labels_rows"] != boundaries[-2]:
        raise RuntimeError("Development labels boundary mismatch")
    if meta.get("final_labels_accessed") is not False:
        raise RuntimeError("Prepared artifact breached label isolation")
    return boundaries


def fit(prepared=OUTPUT, output=FIT_OUTPUT):
    prepared, output = Path(prepared), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refusing to overwrite frozen BankSim fit")
    with (prepared / "materialization.json").open(encoding="utf-8") as handle:
        meta = json.load(handle)
    boundaries = split_boundaries(meta)
    features = np.load(prepared / "features.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(prepared / "development_labels.npy", allow_pickle=False)
    if features.shape != (EXPECTED_ROWS, 24) or features.dtype != np.float32:
        raise RuntimeError("Unexpected feature matrix")
    if labels.shape != (boundaries[-2],) or labels.dtype != np.uint8:
        raise RuntimeError("Unexpected development label vector")
    # Only training, validation and calibration slices are accessed.
    end_train, end_validation, end_calibration = boundaries[1:4]
    if not np.isfinite(features[:end_calibration]).all():
        raise ValueError("Nonfinite development features")
    if not np.isin(labels[:end_calibration], (0, 1)).all():
        raise ValueError("Invalid development labels")
    if any(len(np.unique(y)) != 2 for y in (
        labels[:end_train], labels[end_train:end_validation],
        labels[end_validation:end_calibration],
    )):
        raise ValueError("Each fitting stage needs both fraud classes")

    candidates = {}
    for name, cols in (("transaction_only", 16), ("behavioral", 24)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", solver="lbfgs",
                max_iter=300, C=1.0, random_state=2026,
            ),
        )
        model.fit(features[:end_train, :cols], labels[:end_train])
        decision = model.decision_function(
            features[end_train:end_validation, :cols]
        )
        ap = float(average_precision_score(
            labels[end_train:end_validation], decision,
        ))
        candidates[name] = (model, cols, ap)

    # Exact ties choose transaction-only, as frozen in the protocol.
    selected = ("behavioral" if candidates["behavioral"][2] >
                candidates["transaction_only"][2] else "transaction_only")
    model, cols, selected_ap = candidates[selected]
    cal_margin = np.asarray(model.decision_function(
        features[end_validation:end_calibration, :cols]
    ), dtype=np.float64).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=1e6, solver="lbfgs", max_iter=300, random_state=2026,
    )
    calibrator.fit(cal_margin, labels[end_validation:end_calibration])
    slope = float(calibrator.coef_[0, 0])
    if not np.isfinite(slope) or slope <= 0:
        raise RuntimeError("Non-monotone calibration; stop")
    probabilities = calibrator.predict_proba(cal_margin)[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Invalid calibrated probabilities")
    # >= threshold may nominate >1% on exact ties; the predeclared
    # arrival-order prefix capacity guard is applied only at evaluation.
    review_threshold = float(
        np.quantile(probabilities, 0.99, method="higher")
    )
    report = {
        "status": "BANKSIM_FIT_FROZEN_NOT_EVALUATED",
        "prepared_source_sha256": meta["source_sha256"],
        "feature_schema_sha256": SCHEMA_SHA256,
        "prepared_metadata_sha256": checksum(prepared / "materialization.json"),
        "prepared_features_sha256": checksum(prepared / "features.npy"),
        "prepared_development_labels_sha256":
            checksum(prepared / "development_labels.npy"),
        "partitions": {k: meta["partition_rows"][k] for k in PARTITIONS},
        "validation_average_precision": {
            name: float(values[2]) for name, values in candidates.items()
        },
        "selected": selected,
        "selected_columns": cols,
        "calibration_slope": slope,
        "calibration_intercept": float(calibrator.intercept_[0]),
        "review_threshold": review_threshold,
        "calibration_target_review_fraction": 0.01,
        "policy_labels_accessed": False,
        "final_labels_accessed": False,
        "block_enabled": False,
        "warning": "BankSim-native benchmark only; frozen PaySim model unchanged.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".banksim_fit_", dir=output.parent) as tmp:
        stage = Path(tmp)
        joblib.dump(model, stage / "model.joblib")
        joblib.dump(calibrator, stage / "calibrator.joblib")
        report["model_sha256"] = checksum(stage / "model.joblib")
        report["calibrator_sha256"] = checksum(stage / "calibrator.joblib")
        (stage / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
        os.replace(stage, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, default=FIT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(fit(args.prepared, args.output), indent=2))


if __name__ == "__main__":
    main()
