"""Read-only audit of archived BankSim one-time final evaluation.

No raw CSV access, model training, or reevaluation. Validates arithmetic
and published per-step review totals against the committed report and lock.
"""
import json
import math
from pathlib import Path

LOCK = Path("reports/banksim_protocol_lock.json")
REPORT = Path("reports/banksim_final_evaluation.json")
START = Path("reports/banksim_final_evaluation_started.json")
OUT = Path("reports/banksim_integrity_audit.json")


def audit(lock, report, started):
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    check(report.get("dataset_sha256") == lock.get("dataset_sha256") ==
          started.get("dataset_sha256"), "dataset hashes differ")
    check(report.get("evaluator_script_sha256") ==
          lock.get("evaluator_script_sha256"), "evaluator hashes differ")
    check(report.get("train_steps") == [0, 107], "training steps differ")
    check(report.get("tuning_steps") == [108, 143], "tuning steps differ")
    check(report.get("final_steps") == [144, 179], "final steps differ")
    check(report.get("block_size") == 10000, "block size differs")
    check(report.get("max_review_fraction") == 0.01, "review budget differs")
    check(report.get("fraud_label_reporting_delay_steps") == 7, "delay differs")
    summaries = {}
    for name in ("lr_static", "hgb_static", "hgb_history", "hgb_amount_only"):
        m = report["models"][name]["final"]
        n, frauds, reviews = (m["transactions"], m["frauds"], m["reviews"])
        tp, fp, fn = m["true_positives"], m["false_positives"], m["false_negatives"]
        check(n == 133231 and frauds == 1440, name + ": cohort size")
        check(tp + fp == reviews, name + ": review decomposition")
        check(tp + fn == frauds, name + ": fraud decomposition")
        check(0 <= reviews <= math.floor(n * 0.01), name + ": aggregate budget")
        check(math.isclose(m["precision"], tp / reviews, abs_tol=1e-12),
              name + ": precision")
        check(math.isclose(m["recall"], tp / frauds, abs_tol=1e-12),
              name + ": recall")
        counts = m["review_counts_by_step"]
        check(set(counts) == {str(s) for s in range(144, 180)},
              name + ": missing day")
        check(sum(counts.values()) == reviews, name + ": step review sum")
        check(all(isinstance(v, int) and v >= 0 for v in counts.values()),
              name + ": negative/noninteger step review")
        d = m["delayed_label_accounting"]
        check(d["delay_steps"] == 7, name + ": delayed label interval")
        check(d["all_labels_released"] is True, name + ": labels not released")
        check(d["labels_released_during_stream"] +
              d["labels_released_after_final_watermark"] == n,
              name + ": delayed label counts")
        check(d["true_positives_after_reporting_delay"] == tp,
              name + ": delayed TP mismatch")
        check(d["max_pending_labels"] >= d["labels_released_after_final_watermark"],
              name + ": pending label accounting")
        summaries[name] = {
            "transactions": n, "frauds": frauds, "reviews": reviews,
            "tp": tp, "fp": fp, "fn": fn, "review_fraction": reviews / n,
            "min_daily_reviews": min(counts.values()),
            "max_daily_reviews": max(counts.values()),
            "mean_daily_reviews": reviews / 36,
        }
    base, hist = summaries["hgb_static"], summaries["hgb_history"]
    comparison = report["primary_comparison_hgb_history_minus_hgb_static"]
    check(comparison["true_positive_difference"] == hist["tp"] - base["tp"],
          "primary TP difference mismatch")
    check(comparison["reviews_difference"] == hist["reviews"] - base["reviews"],
          "primary review difference mismatch")
    check(math.isclose(
        comparison["pr_auc_difference"],
        report["models"]["hgb_history"]["final"]["pr_auc_ranking_diagnostic"] -
        report["models"]["hgb_static"]["final"]["pr_auc_ranking_diagnostic"],
        abs_tol=1e-12), "primary PR-AUC difference mismatch")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks_scope": "Archived aggregate arithmetic, per-step review totals, "
                        "reported delayed-label accounting and pinned hashes. "
                        "Cannot verify event-level decisions, per-block prefix caps, "
                        "paired outcome uncertainty or runtime leakage without "
                        "reprocessing individual evaluation records.",
        "models": summaries,
        "primary_difference": {
            "extra_tp": hist["tp"] - base["tp"],
            "fewer_false_positives": base["fp"] - hist["fp"],
            "fewer_reviews": base["reviews"] - hist["reviews"],
        },
    }


def main():
    if OUT.exists():
        raise RuntimeError("Existing audit output; refusing overwrite")
    inputs = [json.loads(p.read_text()) for p in (LOCK, REPORT, START)]
    result = audit(*inputs)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit("Archived evidence integrity audit FAILED")


if __name__ == "__main__":
    main()
