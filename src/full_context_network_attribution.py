"""Development-only N1..N5 attribution using add-one and drop-one ablations."""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from .full_context import FULL_FEATURE_NAMES
from .full_context_candidates import construct, validate_inputs, file_sha256

BASE = (1, 2, 3, 4, 5, 9, 10, 11, 13, 14, 15)
NETWORK = (16, 17, 18, 19, 20)
EXCLUDED = {0, 6, 7, 8, 12}
assert not EXCLUDED.intersection(BASE + NETWORK)


def variants():
    result = [("baseline", BASE)]
    result += [(f"add_{FULL_FEATURE_NAMES[n]}", BASE + (n,)) for n in NETWORK]
    result += [(f"drop_{FULL_FEATURE_NAMES[n]}", BASE + tuple(k for k in NETWORK if k != n)) for n in NETWORK]
    result.append(("full_network", BASE + NETWORK))
    return result


def diagnostic(y, score):
    n = len(y)
    k = max(1, int(np.ceil(n * 0.01)))
    selected = np.lexsort((np.arange(n), -score))[:k]
    positives = int(np.count_nonzero(y))
    tp = int(np.count_nonzero(y[selected]))
    return {"reviews": k, "true_positives": tp, "false_positives": k-tp,
            "precision": tp/k, "recall": tp/positives if positives else None}


def run(source, split, output, *, families=("logistic", "hgb")):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Attribution output already exists; no overwrite")
    if not families or any(f not in ("logistic", "hgb") for f in families):
        raise ValueError("Unknown model family")
    manifest, _, X, y = validate_inputs(source, split)
    train_end, val_end = manifest["split_end_rows"][:2]
    labels_train = np.asarray(y[:train_end])
    labels_val = np.asarray(y[train_end:val_end])
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix=".fgz_network_", dir=output.parent) as directory:
        stage = Path(directory)
        for family in families:
            for name, indices in variants():
                start = time.perf_counter()
                model = construct(family)
                train_X = np.asarray(X[:train_end, indices], dtype=np.float32)
                model.fit(train_X, labels_train)
                del train_X
                validation_X = np.asarray(X[train_end:val_end, indices], dtype=np.float32)
                scores = model.predict_proba(validation_X)[:, 1]
                del validation_X
                if not np.all(np.isfinite(scores)) or np.any(scores < 0) or np.any(scores > 1):
                    raise RuntimeError("Invalid uncalibrated model scores")
                path = stage / f"{family}_{name}.joblib"
                joblib.dump(model, path)
                row = {"candidate": f"{family}_{name}",
                       "features": [FULL_FEATURE_NAMES[i] for i in indices],
                       "validation_ap": float(average_precision_score(labels_val, scores)),
                       "retrospective_top1pct": diagnostic(labels_val, scores),
                       "elapsed_seconds": round(time.perf_counter()-start, 3),
                       "model_sha256": file_sha256(path)}
                results.append(row)
                print(f"{row['candidate']}: AP {row['validation_ap']:.6f}, TP@1% {row['retrospective_top1pct']['true_positives']}", flush=True)
        report = {"status": "EXPLORATORY_NETWORK_ABLATION_NOT_FOR_PROMOTION",
                  "train_rows": train_end, "validation_rows": val_end-train_end,
                  "excluded_current_feature_names": [FULL_FEATURE_NAMES[i] for i in sorted(EXCLUDED)],
                  "candidates": results,
                  "warning": "PaySim development validation already inspected. No calibration, intervention, independent holdout or streaming conclusions."}
        (stage / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
        os.replace(stage, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed/full_context_dev_v1"))
    parser.add_argument("--split", type=Path, default=Path("reports/split_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/full_context_network_attribution_v1"))
    parser.add_argument("--family", choices=("logistic", "hgb"), help="Restrict to one model family (engineering smoke)")
    args = parser.parse_args()
    run(args.source, args.split, args.output, families=(args.family,) if args.family else ("logistic", "hgb"))


if __name__ == "__main__":
    main()
