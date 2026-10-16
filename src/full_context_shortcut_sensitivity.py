"""Exploratory sensitivity: remove all CURRENT amount/balance features.

Train/validation only, on the existing development materialization. No
calibration, policy, holdout or existing model artifacts are read/modified.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from .full_context import FULL_FEATURE_NAMES
from .full_context_candidates import (
    construct, top_review_metrics, validate_inputs, file_sha256,
)

GROUPS = {
    "type_only": (1, 2, 3, 4, 5),
    "type_history": (1, 2, 3, 4, 5, 9, 10, 11, 13, 14, 15),
    "type_history_network": (
        1, 2, 3, 4, 5, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20
    ),
}
FORBIDDEN = {0, 6, 7, 8, 12}


def run(source, split, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output exists; refusing overwrite")
    base, metadata, X, y = validate_inputs(source, split)
    train_end, val_end = base["split_end_rows"][:2]
    train_y = np.asarray(y[:train_end])
    val_y = np.asarray(y[train_end:val_end])
    reports = []
    # Ignore any later partition deliberately; validation only.
    from tempfile import TemporaryDirectory
    import os
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".shortcut_", dir=output.parent) as work:
        wd = Path(work)
        for family in ("logistic", "hgb"):
            for name, indexes in GROUPS.items():
                assert not FORBIDDEN.intersection(indexes)
                model = construct(family)
                # Integer array indexing requires new arrays; bounded by the
                # known 48 GiB dev machine; one variant at a time.
                train_X = np.asarray(X[:train_end, :][:, indexes], dtype=np.float32)
                model.fit(train_X, train_y)
                del train_X
                val_X = np.asarray(X[train_end:val_end, :][:, indexes], dtype=np.float32)
                scores = model.predict_proba(val_X)[:, 1]
                del val_X
                if not np.all(np.isfinite(scores)):
                    raise RuntimeError("Nonfinite validation scores")
                key = f"{family}_{name}"
                path = wd / f"{key}.joblib"
                joblib.dump(model, path)
                reports.append({
                    "candidate": key,
                    "feature_names": [FULL_FEATURE_NAMES[i] for i in indexes],
                    "validation_ap": float(average_precision_score(val_y, scores)),
                    "retrospective_top1pct": top_review_metrics(val_y, scores),
                    "model_sha256": file_sha256(path),
                })
                print(f"{key}: AP={reports[-1]['validation_ap']:.6f}, "
                      f"TP@global1%={reports[-1]['retrospective_top1pct']['true_positives']}",
                      flush=True)
        result = {
            "status": "EXPLORATORY_SHORTCUT_SENSITIVITY_NOT_FOR_PROMOTION",
            "excluded_columns": [FULL_FEATURE_NAMES[i] for i in sorted(FORBIDDEN)],
            "train_rows": train_end,
            "validation_rows": val_end-train_end,
            "candidates": reports,
            "warning": (
                "Removes current amount/balance-derived features only. "
                "Historical amount structure remains. Previously inspected "
                "PaySim validation, not independent test. No streaming policy, "
                "calibration, threshold or generalization claim."
            ),
        }
        (wd / "comparison.json").write_text(json.dumps(result, indent=2)+"\n")
        os.replace(wd, output)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed/full_context_dev_v1"))
    parser.add_argument("--split", type=Path, default=Path("reports/split_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/full_context_no_current_shortcut_v1"))
    args = parser.parse_args()
    run(args.source, args.split, args.output)


if __name__ == "__main__":
    main()
