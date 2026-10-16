"""Post-hoc v2 REVIEW sensitivity: calibration-derived cutoff, inclusive ties.

Standalone exploratory replay: no original Model B changes, no BLOCK,
no untouched-holdout claims. Never overwrite the archived v1 results.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np

from .full_context_calibration import INDICES, MODEL_NAME, verify_model
from .full_context_candidates import validate_inputs
from .full_context_materialize import file_sha256
from .full_context_policy import (
    REVIEW_RATE, summarize, decisions_with_prefix_cap, CANDIDATES,
    CALIBRATION, SOURCE, SPLIT
)
from .features import TRANSACTION_TYPES

OUTPUT = Path("data/processed/full_context_policy_logistic_v2")


def inclusive_threshold(cal_scores, rate=REVIEW_RATE):
    scores = np.asarray(cal_scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid calibration score array")
    if not 0 < rate < 1:
        raise ValueError("Invalid capacity fraction")
    k = int(np.floor(rate * len(scores)))
    return float(np.partition(scores, -k)[-k]) if k else None


def run(source=SOURCE, split=SPLIT, candidates=CANDIDATES,
        calibration=CALIBRATION, output=OUTPUT):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("v2 output exists; refusing overwrite")
    model_path, digest = verify_model(candidates)
    calibration = Path(calibration)
    info = json.loads((calibration/"calibration.json").read_text())
    cp = calibration/"calibrator.joblib"
    if (info.get("status") != "PROVISIONAL_CALIBRATION_FIT_ONLY"
            or info.get("model_name") != MODEL_NAME
            or info.get("model_sha256") != digest
            or info.get("feature_indices") != list(INDICES)
            or info.get("split_manifest_sha256") != file_sha256(split)
            or info.get("block_enabled") is not False
            or cp.is_symlink()
            or file_sha256(cp) != info.get("calibrator_sha256")):
        raise RuntimeError("Calibration provenance mismatch")
    manifest, _, X, y = validate_inputs(source, split)
    cal_start, policy_start = manifest["split_end_rows"][1:3]
    end = manifest["development_row_count"]
    if (info.get("calibration_row_start") != cal_start
            or info.get("calibration_row_end_exclusive") != policy_start):
        raise RuntimeError("Calibration split mismatch")
    model = joblib.load(model_path)
    calibrator = joblib.load(cp)
    if (model.n_features_in_ != len(INDICES)
            or calibrator.n_features_in_ != 1
            or calibrator.coef_[0, 0] <= 0):
        raise RuntimeError("Model or sigmoid mismatch")

    def scores(start, stop):
        features = np.asarray(X[start:stop, INDICES], dtype=np.float32)
        margins = np.asarray(model.decision_function(features), dtype=np.float64)
        result = np.asarray(
            calibrator.predict_proba(margins.reshape(-1, 1))[:, 1],
            dtype=np.float64
        )
        if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
            raise RuntimeError("Invalid calibrated predictions")
        return result

    # Determine threshold without reading calibration labels or any policy
    # rows. Policy scores and labels are used later only for a clearly
    # retrospective, already-inspected diagnostic.
    cal_scores = scores(cal_start, policy_start)
    threshold = inclusive_threshold(cal_scores)
    tie_count = int(np.count_nonzero(cal_scores == threshold)) if threshold is not None else 0
    del cal_scores
    policy_scores = scores(policy_start, end)
    labels = np.asarray(y[policy_start:end], dtype=np.uint8)
    if int(labels.sum()) != manifest["partitions"]["policy"]["frauds"]:
        raise RuntimeError("Policy label counts mismatch")
    chosen = decisions_with_prefix_cap(policy_scores, threshold)
    types = np.asarray(X[policy_start:end, 1:6])
    if np.any(np.count_nonzero(types, axis=1) != 1):
        raise RuntimeError("Invalid transaction types")
    type_ids = np.argmax(types, axis=1)
    result = {
        "status": "POST_HOC_REVIEW_POLICY_V2_SIMULATION_ONLY",
        "model": MODEL_NAME,
        "model_sha256": digest,
        "calibrator_sha256": info["calibrator_sha256"],
        "split_manifest_sha256": file_sha256(split),
        "calibration_threshold": threshold,
        "calibration_boundary_ties": tie_count,
        "tie_rule": "inclusive cutoff; arrival-order prefix cap",
        "review_budget_fraction": REVIEW_RATE,
        "maximum_reviews": int(np.floor(REVIEW_RATE*len(policy_scores))),
        "policy_rows": int(len(policy_scores)),
        "policy_frauds": int(labels.sum()),
        "policy_eligible": int(np.count_nonzero(policy_scores >= threshold)) if threshold is not None else 0,
        "outcomes": summarize(labels, chosen),
        "by_type": {
            kind: summarize(labels[type_ids == i], chosen[type_ids == i])
            for i, kind in enumerate(TRANSACTION_TYPES)
        },
        "block_enabled": False,
        "block_threshold": None,
        "blocks": 0,
        "warning": "Post-hoc diagnostic on previously inspected policy data. Not a fresh test or deployed fraud policy; automatic blocking remains disabled.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fgz_review_v2_", dir=output.parent) as td:
        stage = Path(td)
        (stage/"policy.json").write_text(json.dumps(result, indent=2)+"\n")
        os.replace(stage, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(output=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
