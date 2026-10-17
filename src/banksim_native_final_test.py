"""One-shot BankSim-native final holdout evaluation, kept separate from PaySim.

Default invocation verifies and scores final features WITHOUT reading any
holdout labels. Reading the held-out labels requires --unseal-final-labels.
An exclusive attempt marker is written BEFORE reading those labels, and
the report is never overwritten. All reported IDs are aggregate categories.
"""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from .banksim_native_final_gate import THRESHOLD, verify
from .banksim_native_fit import FIT_OUTPUT, checksum
from .banksim_native_policy_diagnostic import prefix_reviews
from .banksim_native_prepare import CATEGORIES, OUTPUT, SOURCE, SOURCE_SHA256
from .banksim_structural_audit import REQUIRED
from .independent_evaluation_metrics import score_report

REPORT = Path("reports/banksim_native_final_test_v1.json")
FINAL_START_STEP = 162
FINAL_END_STEP = 179
BEFORE_FINAL_ROWS = 527671
FINAL_ROWS = 66972
TOTAL_ROWS = BEFORE_FINAL_ROWS + FINAL_ROWS


def read_final_labels(source, *, expected_sha256=SOURCE_SHA256,
                      expected_before=BEFORE_FINAL_ROWS,
                      expected_final=FINAL_ROWS,
                      first_step=FINAL_START_STEP,
                      last_step=FINAL_END_STEP):
    """Explicit unseal: read ONLY final-period label values from vetted CSV."""
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise RuntimeError("Missing regular holdout source")
    if checksum(source) != expected_sha256:
        raise RuntimeError("Frozen source checksum mismatch before label access")
    labels = []
    earlier_rows = 0
    previous_step = -1
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if (not headers or len(headers) != len(set(headers))
                or set(REQUIRED).difference(headers)):
            raise RuntimeError("Unexpected BankSim source header")
        for row in reader:
            raw_step = row.get("step")
            try:
                step = int(raw_step)
                if (raw_step.strip() != str(step) or step < previous_step
                        or not 0 <= step <= last_step):
                    raise ValueError("Invalid source chronology")
            except (TypeError, ValueError, AttributeError) as exc:
                raise RuntimeError("Source chronology mismatch") from exc
            previous_step = step
            if step < first_step:
                earlier_rows += 1
                # Do not access row["fraud"] for the development partitions.
            else:
                label = row["fraud"]
                if label not in ("0", "1"):
                    raise RuntimeError("Invalid final-test fraud label")
                labels.append(int(label))
    if earlier_rows != expected_before or len(labels) != expected_final:
        raise RuntimeError("Final-test row-boundary mismatch")
    return np.asarray(labels, dtype=np.uint8)


def score_unlabeled_final(prepared, fit_dir, diagnostic):
    """Hash-gated inference on unlabeled holdout features; no label reads."""
    gate = verify(prepared, fit_dir, diagnostic)
    if gate["status"] != "BANKSIM_FINAL_INTEGRITY_GATE_PASSED":
        raise RuntimeError("Final-integrity gate refused execution")
    fit_dir, prepared = Path(fit_dir), Path(prepared)
    # joblib is trusted ONLY after verified byte-for-byte by verify().
    model = joblib.load(fit_dir / "model.joblib")
    calibrator = joblib.load(fit_dir / "calibrator.joblib")
    matrix = np.load(prepared / "features.npy", mmap_mode="r",
                     allow_pickle=False)
    if matrix.shape != (TOTAL_ROWS, 24) or matrix.dtype != np.float32:
        raise RuntimeError("Unexpected holdout feature shape")
    X = np.asarray(matrix[BEFORE_FINAL_ROWS:], dtype=np.float32)
    if X.shape != (FINAL_ROWS, 24) or not np.isfinite(X).all():
        raise RuntimeError("Nonfinite or incomplete holdout features")
    onehot = X[:, 1:16]
    if not np.all(np.sum(onehot, axis=1) == 1):
        raise RuntimeError("Invalid held-out BankSim category indicators")
    categories = np.asarray(CATEGORIES)[np.argmax(onehot, axis=1)]
    margins = np.asarray(model.decision_function(X),
                         dtype=np.float64).reshape(-1, 1)
    risks = np.asarray(calibrator.predict_proba(margins)[:, 1],
                       dtype=np.float64)
    if (risks.shape != (FINAL_ROWS,) or not np.isfinite(risks).all()
            or np.any((risks < 0) | (risks > 1))):
        raise RuntimeError("Invalid holdout calibrated risks")
    decisions = prefix_reviews(risks, cutoff=THRESHOLD)
    if (int(np.count_nonzero(decisions == "REVIEW")) > FINAL_ROWS // 100
            or not np.isin(decisions, ("ALLOW", "REVIEW")).all()):
        raise RuntimeError("Frozen review capacity or BLOCK-disabled violation")
    return risks, decisions, categories, gate


def evaluate(*, prepared=OUTPUT, fit_dir=FIT_OUTPUT,
             diagnostic=Path("reports/banksim_native_policy_diagnostic_v1.json"),
             source=SOURCE, report=REPORT, unseal=False):
    report = Path(report)
    marker = report.with_name(report.name + ".attempt.json")
    if report.exists() or report.is_symlink() or marker.exists() or marker.is_symlink():
        raise FileExistsError("Final evaluation already attempted; never overwrite or retry")
    risks, decisions, categories, gate = score_unlabeled_final(
        prepared, fit_dir, diagnostic
    )
    source = Path(source)
    if source.is_symlink() or not source.is_file() or checksum(source) != SOURCE_SHA256:
        raise RuntimeError("Unapproved final source file")
    if not unseal:
        return {
            "status": "BANKSIM_FINAL_CHECK_ONLY_HOLDOUT_LABELS_SEALED",
            "holdout_rows": FINAL_ROWS,
            "frozen_review_threshold": THRESHOLD,
            "model_sha256": gate["model_sha256"],
            "source_sha256": SOURCE_SHA256,
            "holdout_scores_computed": True,
            "holdout_labels_accessed": False,
            "final_evaluation_executed": False,
        }

    report.parent.mkdir(parents=True, exist_ok=True)
    # This marker is intentionally permanent: on an interrupted run the
    # final holdout may already have been seen and must not be claimed fresh.
    with marker.open("x", encoding="utf-8") as handle:
        json.dump({
            "status": "BANKSIM_FINAL_LABEL_ACCESS_BEGUN",
            "source_sha256": SOURCE_SHA256,
            "model_sha256": gate["model_sha256"],
            "final_rows_expected": FINAL_ROWS,
            "do_not_repeat_or_retrain": True,
        }, handle, indent=2)
        handle.write("\n")

    labels = read_final_labels(source)
    if labels.shape != (FINAL_ROWS,) or len(np.unique(labels)) != 2:
        raise RuntimeError("Final-test labels invalid; keep attempt marker")
    metrics = score_report(labels, risks, decisions, categories)
    reviewed = int(np.count_nonzero(decisions == "REVIEW"))
    if (metrics["rows"] != FINAL_ROWS or metrics["reviewed"] != reviewed
            or metrics["would_block"] != 0):
        raise RuntimeError("Final evaluation accounting mismatch")
    output = {
        "status": "BANKSIM_FINAL_TEST_ONE_TIME_COMPLETE",
        "evaluation_dataset": "BANKSIM_SYNTHETIC_SEPARATE_MODEL",
        "step_range": [FINAL_START_STEP, FINAL_END_STEP],
        "holdout_rows": FINAL_ROWS,
        "frozen_integrity_gate": gate,
        "frozen_review_threshold": THRESHOLD,
        "review_capacity": "prefix_floor_1_percent",
        "reviewed": reviewed,
        "final_labels_accessed": True,
        "block_enabled": False,
        "metrics": metrics,
        "limitations": [
            "This evaluates a separately trained BankSim model on a BankSim chronological holdout.",
            "It does not validate or modify the original frozen PaySim model.",
            "BankSim is synthetic; metrics are not evidence of real-world detection.",
            "BLOCK was disabled: zero false blocks does not establish blocking safety.",
            "Holdout is no longer untouched after this one-time evaluation.",
            "No retuning or replacement run based on final results is permitted.",
        ],
    }
    # Exclusive report write (as well as the permanent attempt marker).
    with report.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=OUTPUT)
    parser.add_argument("--fit", type=Path, default=FIT_OUTPUT)
    parser.add_argument("--diagnostic", type=Path,
                        default=Path("reports/banksim_native_policy_diagnostic_v1.json"))
    parser.add_argument("--csv", type=Path, default=SOURCE)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--unseal-final-labels", action="store_true",
                        help="One-time irreversibly open BankSim holdout labels")
    args = parser.parse_args()
    result = evaluate(prepared=args.prepared, fit_dir=args.fit,
                      diagnostic=args.diagnostic, source=args.csv,
                      report=args.report, unseal=args.unseal_final_labels)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
