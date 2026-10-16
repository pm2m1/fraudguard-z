"""Read-only audit of the already-published, one-time BankSim report.

Reads ONLY archived JSON: never opens the raw BankSim CSV, loads a model,
computes predictions, reruns final evaluation or changes a policy.

Run: python -m src.banksim_report_audit
Frozen source: commit 44971f2 (the first published final-result archive).
"""
import hashlib
import json
import math
import subprocess
from pathlib import Path

ARCHIVE_COMMIT = "44971f2"
REPORT = Path("reports/banksim_final_evaluation.json")
LOCK = Path("reports/banksim_protocol_lock.json")
STARTED = Path("reports/banksim_final_evaluation_started.json")
OUTPUT = Path("reports/banksim_integrity_audit.json")
MODELS = {
    "lr_static": 16,
    "hgb_static": 16,
    "hgb_history": 21,
    "hgb_amount_only": 1,
}
FINAL_STEPS = tuple(range(144, 180))


def _archived_bytes(path):
    try:
        return subprocess.run(
            ["git", "show", f"{ARCHIVE_COMMIT}:{path}"],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Frozen BankSim final-result commit is unavailable") from exc


def read_archived():
    """Ensure working copies match the original immutable archive commit."""
    payload = {}
    hashes = {}
    for name, path in (("report", REPORT), ("lock", LOCK), ("started", STARTED)):
        raw = _archived_bytes(path)
        if path.read_bytes() != raw:
            raise RuntimeError(f"Working copy of {path} differs from original archive")
        payload[name] = json.loads(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return payload, hashes


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def close_enough(a, b, name):
    require(isinstance(b, (int, float)) and not isinstance(b, bool),
            f"{name}: not numeric")
    require(math.isclose(a, float(b), rel_tol=1e-10, abs_tol=1e-12),
            f"{name}: inconsistent metric")


def audit_report(report, lock, started):
    """Check all observable accounting identities; no raw event-level evidence.

    Hard per-event prefix caps, fraud timing, paired detector outcomes, and
    per-step fraud performance CANNOT be certified from this aggregate JSON.
    """
    require(report.get("status") == "ONE_TIME_INDEPENDENT_SIMULATOR_FINAL",
            "Unexpected report status")
    require(lock.get("lock_status") == "PRE_EVALUATION_NO_FRAUD_LABELS_READ",
            "Unexpected protocol lock status")
    require(lock.get("final_data_evaluated") is False,
            "Original lock was not created pre-evaluation")
    require(report.get("dataset_sha256") == lock.get("dataset_sha256") ==
            started.get("dataset_sha256"), "Dataset SHA-256 mismatch")
    require(report.get("evaluator_script_sha256") ==
            lock.get("evaluator_script_sha256"), "Evaluator checksum mismatch")
    require(report.get("train_steps") == [0, 107] and
            report.get("tuning_steps") == [108, 143] and
            report.get("final_steps") == [144, 179],
            "Unexpected split")
    require(report.get("training_category_count") == 15,
            "Unexpected training feature schema")
    require(report.get("fraud_label_reporting_delay_steps") == 7,
            "Unexpected label-delay configuration")
    block = report.get("block_size")
    budget = report.get("max_review_fraction")
    require(type(block) is int and block == 10000 and
            type(budget) in (int, float) and budget == .01,
            "Unexpected review-cap configuration")
    source = report.get("models", {})
    require(set(source) == set(MODELS), "Missing or unexpected model")
    ref_rows = None
    ref_frauds = None
    model_audits = {}
    for name, expected_width in MODELS.items():
        item = source[name]
        require(item.get("number_of_predictors") == expected_width,
                f"{name}: changed predictor count")
        cutoff = item.get("unlabeled_tuning_score_cutoff")
        require(type(cutoff) in (int, float) and math.isfinite(cutoff),
                f"{name}: cutoff invalid")
        m = item["final"]
        rows = m["transactions"]
        frauds = m["frauds"]
        reviewed = m["reviews"]
        tp = m["true_positives"]
        fp = m["false_positives"]
        fn = m["false_negatives"]
        require(all(type(v) is int and v >= 0 for v in
                    (rows, frauds, reviewed, tp, fp, fn)), f"{name}: bad counts")
        require(rows > 0 and 0 < frauds <= rows, f"{name}: bad denominators")
        if ref_rows is None:
            ref_rows = rows
            ref_frauds = frauds
        require((rows, frauds) == (ref_rows, ref_frauds),
                f"{name}: models evaluated on different populations")
        require(rows == 133231 and frauds == 1440,
                f"{name}: final partition numbers changed")
        require(tp + fp == reviewed and tp + fn == frauds,
                f"{name}: review confusion-matrix inconsistency")
        close_enough(tp / reviewed if reviewed else 0,
                     m["precision"] if reviewed else (m["precision"] or 0),
                     f"{name}: precision")
        close_enough(tp / frauds, m["recall"], f"{name}: recall")
        ap = m["pr_auc_ranking_diagnostic"]
        require(type(ap) in (float, int) and 0 <= ap <= 1,
                f"{name}: PR-AUC out of range")
        counts = m["review_counts_by_step"]
        require(set(counts) == {str(step) for step in FINAL_STEPS},
                f"{name}: incomplete per-step accounting")
        require(all(type(v) is int and v >= 0 for v in counts.values()),
                f"{name}: invalid step count")
        require(sum(counts.values()) == reviewed,
                f"{name}: per-step reviews do not sum to total")

        # Aggregate upper bound is necessary, NOT a proof of prefix-level caps.
        full_blocks, remainder = divmod(rows, block)
        max_reviewable = (full_blocks * math.floor(block * budget) +
                          math.floor(remainder * budget))
        require(reviewed <= max_reviewable,
                f"{name}: aggregate cap exceeded")
        delayed = m["delayed_label_accounting"]
        require(delayed["delay_steps"] == 7 and
                delayed["all_labels_released"] is True,
                f"{name}: delayed release invalid")
        during = delayed["labels_released_during_stream"]
        after = delayed["labels_released_after_final_watermark"]
        pending = delayed["max_pending_labels"]
        require(all(type(v) is int and v >= 0 for v in (during, after, pending)),
                f"{name}: invalid delayed counts")
        require(during + after == rows and
                after <= pending <= rows and
                delayed["true_positives_after_reporting_delay"] == tp,
                f"{name}: delayed accounting mismatch")
        vals = list(counts.values())
        model_audits[name] = {
            "transactions": rows,
            "frauds": frauds,
            "reviews": reviewed,
            "max_reviews_per_aggregate_block_accounting": max_reviewable,
            "review_utilization_of_aggregate_cap": reviewed / max_reviewable,
            "per_step_review_min": min(vals),
            "per_step_review_max": max(vals),
            "per_step_review_mean": reviewed / len(vals),
            "per_step_review_count": len(vals),
            "delayed_labels_accounted_for": during + after,
            "labels_released_at_final_watermark": after,
            "accounting_checks_pass": True,
        }
    static = source["hgb_static"]["final"]
    history = source["hgb_history"]["final"]
    claimed = report["primary_comparison_hgb_history_minus_hgb_static"]
    require(history["true_positives"] - static["true_positives"] ==
            claimed["true_positive_difference"], "TP delta mismatch")
    require(history["reviews"] - static["reviews"] ==
            claimed["reviews_difference"], "Review delta mismatch")
    close_enough(history["pr_auc_ranking_diagnostic"] -
                 static["pr_auc_ranking_diagnostic"],
                 claimed["pr_auc_difference"], "PR-AUC delta")
    h = source["hgb_history"]["final"]["review_counts_by_step"]
    s = source["hgb_static"]["final"]["review_counts_by_step"]
    step_diff = {step: h[str(step)] - s[str(step)] for step in FINAL_STEPS}
    require(sum(step_diff.values()) == claimed["reviews_difference"],
            "Per-step review differences mismatch")
    return {
        "status": "PASS_AGGREGATE_ARCHIVED_JSON_INTEGRITY",
        "source_archive_commit": ARCHIVE_COMMIT,
        "primary_observed": {
            "additional_frauds_detected": claimed["true_positive_difference"],
            "fewer_false_alerts": static["false_positives"] -
                                 history["false_positives"],
            "change_in_reviews": claimed["reviews_difference"],
            "pr_auc_difference": claimed["pr_auc_difference"],
        },
        "model_checks": model_audits,
        "per_step_review_difference_history_minus_static": {
            "more_history_reviews_steps": sum(value > 0 for value in step_diff.values()),
            "same_reviews_steps": sum(value == 0 for value in step_diff.values()),
            "fewer_history_reviews_steps": sum(value < 0 for value in step_diff.values()),
            "minimum_difference": min(step_diff.values()),
            "maximum_difference": max(step_diff.values()),
            "total_difference": sum(step_diff.values()),
            "by_step": {str(step): value for step, value in step_diff.items()},
        },
        "unverifiable_from_archived_aggregate_only": [
            "Transaction-level paired true-positive overlap and paired confidence interval",
            "Per-step fraud recall or time stability of extra true positives",
            "Every-prefix review-cap compliance (aggregate count is only necessary)",
            "Exact label-release timing for individual transactions",
            "True out-of-distribution generalization to an unseen simulator run",
            "Model calibration, or performance on real-world fraud",
        ],
        "interpretation": (
            "A read-only consistency audit of a previously completed one-time "
            "synthetic final evaluation. Does not re-evaluate or change it; "
            "positive model differences are not independent replications."
        ),
    }


def main():
    data, hashes = read_archived()
    result = audit_report(data["report"], data["lock"], data["started"])
    result["archived_json_sha256"] = hashes
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output = json.dumps(result, indent=2) + "\n"
    if OUTPUT.exists() and OUTPUT.read_text() != output:
        raise RuntimeError("Existing audit output differs; refusing overwrite")
    if not OUTPUT.exists():
        OUTPUT.write_text(output)
    print("PASS: Archived aggregate report and delayed accounting agree")
    print(json.dumps(result["primary_observed"], indent=2))
    print(json.dumps(result["per_step_review_difference_history_minus_static"],
                     indent=2))
    print("Saved: " + str(OUTPUT))
    print("No raw data, model files, or final evaluator were opened.")


if __name__ == "__main__":
    main()
