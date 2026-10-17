"""Read-only diagnosis of the already-inspected PaySim v2 CASH_OUT policy blind spot.

This is retrospective explanation of existing engineering results, NOT a new
validation, policy-selection or model-training experiment. Never touch final
holdout rows or alter threshold, review quota or frozen artifacts.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from .features import TRANSACTION_TYPES
from .full_context_calibration import INDICES, MODEL_NAME, verify_model
from .full_context_candidates import validate_inputs
from .full_context_materialize import file_sha256
from .full_context_policy import (
    CANDIDATES, CALIBRATION, REVIEW_RATE, SOURCE, SPLIT,
    decisions_with_prefix_cap, summarize
)
from .full_context_policy_v2 import OUTPUT as POLICY_DIR, inclusive_threshold

OUTPUT = Path("data/processed/full_context_cashout_diagnostic_v1.json")


def distribution(values):
    """Only aggregates; never return row indices, ids or event-level scores."""
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("Expected 1-D values")
    if not len(a):
        return None
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite risk score")
    return {
        "min": float(a.min()),
        "p50": float(np.quantile(a, .50)),
        "p90": float(np.quantile(a, .90)),
        "p99": float(np.quantile(a, .99)),
        "p999": float(np.quantile(a, .999)),
        "max": float(a.max()),
    }


def type_diagnostic(labels, scores, reviewed, threshold):
    y = np.asarray(labels)
    p = np.asarray(scores, dtype=np.float64)
    chosen = np.asarray(reviewed, dtype=bool)
    if (y.ndim != 1 or p.ndim != 1 or chosen.ndim != 1
            or len(y) != len(p) or len(p) != len(chosen)
            or not np.isin(y, (0, 1)).all()
            or not np.isfinite(p).all()
            or not (0 <= threshold <= 1)):
        raise ValueError("Malformed diagnostic inputs")
    eligible = p >= threshold
    fraud = y == 1
    return {
        **summarize(y, chosen),
        "eligible": int(np.count_nonzero(eligible)),
        "eligible_frauds": int(np.count_nonzero(eligible & fraud)),
        "eligible_not_reviewed": int(np.count_nonzero(eligible & ~chosen)),
        "reviewed_below_threshold": int(np.count_nonzero(chosen & ~eligible)),
        "exact_cutoff_ties": int(np.count_nonzero(p == threshold)),
        "score_distribution_all": distribution(p),
        "score_distribution_fraud": distribution(p[fraud]),
        "score_distribution_nonfraud": distribution(p[~fraud]),
        "score_max_minus_threshold": float(p.max() - threshold) if len(p) else None,
        "ap_ranking_diagnostic": (
            float(average_precision_score(y, p))
            if len(y) and np.any(fraud) and np.any(~fraud) else None
        ),
    }


def cashout_observation(info):
    if info["eligible"] == 0:
        return (
            "ZERO_CASHOUT_SCORE_REACHED_GLOBAL_THRESHOLD; "
            "review-cap competition cannot alone explain zero CASH_OUT reviews"
        )
    if info["reviews"] == 0:
        return (
            "SOME_CASHOUT_SCORES_REACHED_THRESHOLD_BUT_NONE_REVIEWED; "
            "prefix-cap and chronological order require investigation"
        )
    return "SOME_CASHOUT_TRANSACTIONS_WERE_REVIEWED"


def run(source=SOURCE, split=SPLIT, candidates=CANDIDATES,
        calibration=CALIBRATION, policy=POLICY_DIR / "policy.json",
        output=OUTPUT):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Read-only diagnostic refuses to overwrite reports")
    model_path, model_sha = verify_model(candidates)
    calibration = Path(calibration)
    meta = json.loads((calibration / "calibration.json").read_text())
    cal_file = calibration / "calibrator.joblib"
    policy = Path(policy)
    frozen = json.loads(policy.read_text())
    split_sha = file_sha256(split)
    if (meta.get("status") != "PROVISIONAL_CALIBRATION_FIT_ONLY"
            or meta.get("model_name") != MODEL_NAME
            or meta.get("model_sha256") != model_sha
            or meta.get("feature_indices") != list(INDICES)
            or meta.get("split_manifest_sha256") != split_sha
            or meta.get("block_enabled") is not False
            or cal_file.is_symlink() or not cal_file.is_file()
            or file_sha256(cal_file) != meta.get("calibrator_sha256")
            or frozen.get("status") != "POST_HOC_REVIEW_POLICY_V2_SIMULATION_ONLY"
            or frozen.get("model") != MODEL_NAME
            or frozen.get("model_sha256") != model_sha
            or frozen.get("calibrator_sha256") != meta["calibrator_sha256"]
            or frozen.get("split_manifest_sha256") != split_sha
            or frozen.get("review_budget_fraction") != REVIEW_RATE
            or frozen.get("block_enabled") is not False
            or frozen.get("blocks") != 0
            or frozen.get("tie_rule") != "inclusive cutoff; arrival-order prefix cap"):
        raise RuntimeError("Frozen model/calibrator/v2 policy provenance mismatch")

    manifest, _, X, y = validate_inputs(source, split)
    cal_begin, pol_begin = manifest["split_end_rows"][1:3]
    end = manifest["development_row_count"]
    if (meta.get("calibration_row_start") != cal_begin
            or meta.get("calibration_row_end_exclusive") != pol_begin
            or pol_begin - cal_begin != manifest["partitions"]["calibration"]["rows"]
            or end - pol_begin != manifest["partitions"]["policy"]["rows"]):
        raise RuntimeError("Calibration/policy boundaries mismatch")

    model = joblib.load(model_path)  # checksum-verified, project-generated file
    calibrator = joblib.load(cal_file)
    if (getattr(model, "n_features_in_", None) != len(INDICES)
            or getattr(calibrator, "n_features_in_", None) != 1
            or float(calibrator.coef_[0, 0]) <= 0):
        raise RuntimeError("Scoring interface or monotone calibrator mismatch")

    def score(first, last):
        # Explicitly bound below development end: no untouched final holdout.
        if not cal_begin <= first < last <= end:
            raise RuntimeError("Diagnostic attempted access outside allowed periods")
        values = np.asarray(X[first:last, :][:, INDICES], dtype=np.float32)
        margins = np.asarray(model.decision_function(values), dtype=np.float64)
        result = np.asarray(calibrator.predict_proba(margins.reshape(-1, 1))[:, 1],
                            dtype=np.float64)
        if len(result) != last - first or not np.isfinite(result).all():
            raise RuntimeError("Invalid calibrated score array")
        return result

    calibration_scores = score(cal_begin, pol_begin)
    cutoff = inclusive_threshold(calibration_scores)
    frozen_cutoff = frozen.get("calibration_threshold")
    if (cutoff is None or type(frozen_cutoff) not in (float, int)
            or not np.isclose(cutoff, frozen_cutoff, rtol=0, atol=1e-15)):
        raise RuntimeError("Existing v2 cutoff not reproducible")
    policy_scores = score(pol_begin, end)
    policy_labels = np.asarray(y[pol_begin:end], dtype=np.uint8)
    if (int(np.count_nonzero(policy_labels)) !=
            manifest["partitions"]["policy"]["frauds"]):
        raise RuntimeError("Policy label totals mismatch")
    chosen = decisions_with_prefix_cap(policy_scores, cutoff)

    onehots = np.asarray(X[pol_begin:end, 1:6], dtype=np.float32)
    if not np.all(np.count_nonzero(onehots, axis=1) == 1):
        raise RuntimeError("Malformed policy transaction types")
    type_ids = np.argmax(onehots, axis=1)
    by_type = {
        kind: type_diagnostic(policy_labels[type_ids == i],
                              policy_scores[type_ids == i],
                              chosen[type_ids == i], cutoff)
        for i, kind in enumerate(TRANSACTION_TYPES)
    }
    overall = summarize(policy_labels, chosen)
    # Most important safety guarantee: this isn't an alternate policy.
    if (overall != frozen.get("outcomes")
            or int(np.count_nonzero(policy_scores >= cutoff)) !=
            frozen.get("policy_eligible")
            or sum(t["eligible"] for t in by_type.values()) !=
            frozen.get("policy_eligible")
            or any({k: details[k] for k in
                    ("rows", "frauds", "reviews", "true_positives",
                     "false_positives", "precision", "recall")}
                   != frozen["by_type"][kind]
                   for kind, details in by_type.items())):
        raise RuntimeError("Existing frozen v2 outcomes not reproduced")

    result = {
        "status": "READ_ONLY_POST_HOC_CASHOUT_DIAGNOSIS",
        "evidence_class": "ALREADY_INSPECTED_PAYSIM_POLICY_ONLY",
        "source_model": MODEL_NAME,
        "model_sha256": model_sha,
        "calibrator_sha256": meta["calibrator_sha256"],
        "policy_report_sha256": file_sha256(policy),
        "split_manifest_sha256": split_sha,
        "cutoff": float(cutoff),
        "cutoff_calibration_exact_ties": int(np.count_nonzero(
            calibration_scores == cutoff
        )),
        "cutoff_is_global": True,
        "no_type_exclusion": True,
        "global_review_budget": REVIEW_RATE,
        "policy_rows": int(len(policy_scores)),
        "policy_reviews": overall["reviews"],
        "policy_frauds": overall["frauds"],
        "policy_eligible": frozen["policy_eligible"],
        "policy_outcomes_reproduced": True,
        "by_type": by_type,
        "cashout_observation": cashout_observation(by_type["CASH_OUT"]),
        "warning": (
            "Uses already-inspected policy labels, never the untouched final holdout. "
            "Descriptive score/eligibility diagnosis; not attribution, causal proof, "
            "new validation or a proposed cutoff. Calibrated scores are not validated "
            "real-world fraud probabilities. Model/policy not changed; BLOCK disabled."
        )
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(output=args.output)
    print("Status:", result["status"])
    print("Prior policy outcomes reproduced:", result["policy_outcomes_reproduced"])
    print("CASH_OUT:", json.dumps(result["by_type"]["CASH_OUT"], indent=2))
    print("TRANSFER:", json.dumps(result["by_type"]["TRANSFER"], indent=2))
    print("Observation:", result["cashout_observation"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
