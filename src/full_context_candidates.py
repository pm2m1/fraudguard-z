"""Exploratory full-context candidate comparisons; no calibration or policy access.

Output is a fresh directory under ignored data/processed. It NEVER modifies
production models or the old evaluated final test.
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .full_context import FULL_FEATURE_NAMES
from .full_context_journal import FEATURE_SCHEMA_SHA256
from .full_context_materialize import file_sha256, validate_manifest

SOURCE = Path("data/processed/full_context_dev_v1")
SPLIT = Path("reports/split_manifest.json")
OUTPUT = Path("data/processed/full_context_candidates_v1")
FEATURE_GROUPS = {"transaction": 9, "behavioral": 16, "full_context": 21}
CANDIDATES = tuple(
    (family, group) for family in ("logistic", "hgb")
    for group in FEATURE_GROUPS
)


def validate_inputs(source, split):
    source, split = Path(source), Path(split)
    base = json.loads(split.read_text())
    names, _, counts = validate_manifest(base)
    report = json.loads((source / "materialization.json").read_text())
    if (report.get("status") != "DEVELOPMENT_ONLY_FEATURE_MATERIALIZATION"
            or report.get("schema") != list(FULL_FEATURE_NAMES)
            or report.get("schema_sha256") != FEATURE_SCHEMA_SHA256
            or report.get("split_manifest_sha256") != file_sha256(split)):
        raise RuntimeError("Input materialization/schema/split mismatch")
    if report.get("rows") != base["development_row_count"]:
        raise RuntimeError("Materialized row count differs from frozen manifest")
    for name, count in zip(names, counts):
        if report["partitions"].get(name) != base["partitions"][name]:
            raise RuntimeError("Materialized partition counts differ")
        if report["partitions"][name]["rows"] != count:
            raise RuntimeError("Invalid partition rows")
    X = np.load(source / "features.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(source / "labels.npy", mmap_mode="r", allow_pickle=False)
    if (X.shape != (report["rows"], 21)
            or X.dtype != np.dtype("float32")
            or y.shape != (report["rows"],)
            or y.dtype != np.dtype("uint8")):
        raise RuntimeError("Unexpected feature or label shape/dtype")
    if sum(counts) != len(y):
        raise RuntimeError("Inconsistent row totals")
    return base, report, X, y


def construct(family):
    if family == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", solver="lbfgs",
                max_iter=300, C=1.0, random_state=2026
            )
        )
    if family == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.07, max_iter=100, max_leaf_nodes=15,
            max_bins=128, l2_regularization=10.0,
            class_weight="balanced", early_stopping=False, random_state=2026
        )
    raise ValueError("Unknown family")


def top_review_metrics(labels, scores, k_fraction=0.01):
    """Retrospective ranking diagnostic; not executable stream decisions."""
    n = len(labels)
    k = max(1, int(np.ceil(k_fraction * n)))
    order = np.lexsort((np.arange(n), -scores))
    selected = order[:k]
    tp = int(np.count_nonzero(labels[selected]))
    positives = int(np.count_nonzero(labels))
    return {
        "reviews": k, "true_positives": tp, "false_positives": k-tp,
        "precision": tp/k, "recall": tp/positives if positives else None,
    }


def fit_candidates(source=SOURCE, split=SPLIT, output=OUTPUT, *,
                   max_candidates=None):
    """Fit fixed candidates on train only, evaluate validation only.

    max_candidates is an explicit bounded engineering smoke capability;
    partial output is labeled accordingly and never eligible for selection.
    """
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Candidate directory already exists; refusing overwrite")
    base, report, X, y = validate_inputs(source, split)
    train_end, valid_end = base["split_end_rows"][:2]
    if (int(np.count_nonzero(y[:train_end])) != base["partitions"]["train"]["frauds"]
            or int(np.count_nonzero(y[train_end:valid_end]))
            != base["partitions"]["validation"]["frauds"]):
        raise RuntimeError("Training/validation fraud counts mismatch")
    # The calibration and policy slices are never indexed here.
    candidates = CANDIDATES
    if max_candidates is not None:
        if type(max_candidates) is not int or not 1 <= max_candidates <= len(candidates):
            raise ValueError("Invalid smoke candidate count")
        candidates = candidates[:max_candidates]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".fgz_candidates_", dir=output.parent))
    results = []
    try:
        y_train = np.asarray(y[:train_end])
        y_validation = np.asarray(y[train_end:valid_end])
        for family, group in candidates:
            width = FEATURE_GROUPS[group]
            key = f"{family}_{group}"
            candidate = construct(family)
            print(f"Fitting {key} ({width} features)", flush=True)
            start = time.perf_counter()
            candidate.fit(X[:train_end, :width], y_train)
            elapsed = time.perf_counter() - start
            scores = candidate.predict_proba(
                X[train_end:valid_end, :width]
            )[:, 1]
            if (not np.all(np.isfinite(scores))
                    or np.any(scores < 0) or np.any(scores > 1)):
                raise RuntimeError(f"Invalid scores for {key}")
            metrics = top_review_metrics(y_validation, scores)
            if 0 < int(np.count_nonzero(y_validation)) < len(y_validation):
                ap = float(average_precision_score(y_validation, scores))
            else:
                ap = None
            filename = f"{key}.joblib"
            joblib.dump(candidate, temporary / filename)
            results.append({
                "candidate": key, "model_family": family,
                "feature_group": group,
                "columns": list(FULL_FEATURE_NAMES[:width]),
                "fit_seconds": round(elapsed, 3),
                "validation_pr_auc": ap,
                "validation_top1pct": metrics,
                "model_sha256": file_sha256(temporary / filename),
            })
        summary = {
            "status": ("COMPLETE_EXPLORATORY_CANDIDATE_COMPARISON"
                       if len(candidates) == len(CANDIDATES)
                       else "PARTIAL_ENGINEERING_SMOKE_NOT_FOR_SELECTION"),
            "source_materialization": str(source),
            "split_manifest_sha256": file_sha256(split),
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "training_rows": train_end,
            "validation_rows": valid_end-train_end,
            "train_frauds": int(np.count_nonzero(y_train)),
            "validation_frauds": int(np.count_nonzero(y_validation)),
            "candidates": results,
            "warning": (
                "Validation is development-only, previously inspected; "
                "uncalibrated scores. Top-1% is retrospective, not streaming. "
                "Calibration, policy and final holdout were not evaluated. "
                "Do not automatically promote the highest-AP candidate."
            ),
        }
        (temporary / "comparison.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        os.replace(temporary, output)
        return summary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--smoke-candidates", type=int)
    args = parser.parse_args()
    result = fit_candidates(
        args.source, args.split, args.output,
        max_candidates=args.smoke_candidates
    )
    print("Status:", result["status"])
    for item in result["candidates"]:
        print(
            f"{item['candidate']}: validation AP={item['validation_pr_auc']}, "
            f"TP at retrospective top 1%="
            f"{item['validation_top1pct']['true_positives']}"
        )


if __name__ == "__main__":
    main()
