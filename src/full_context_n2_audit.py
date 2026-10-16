"""Aggregate-only audit of lifetime N2 in frozen train/validation data."""
import argparse
import json
from pathlib import Path

import numpy as np

from .full_context_candidates import validate_inputs
from .features import TRANSACTION_TYPES


BINS = ("zero", "one", "two_to_three", "four_plus")
KINDS = ("ALL", *TRANSACTION_TYPES)


def summarize_group(n2, labels, types, mask):
    results = {}
    for kind in KINDS:
        type_mask = mask if kind == "ALL" else (mask & (types == kind))
        total_fraud = int(np.count_nonzero(labels[type_mask]))
        output = {}
        for bin_index, bin_name in enumerate(BINS):
            select = type_mask & (n2 == bin_index)
            count = int(np.count_nonzero(select))
            frauds = int(np.count_nonzero(labels[select]))
            output[bin_name] = {
                "rows": count, "frauds": frauds,
                "fraud_rate": frauds/count if count else None,
                "fraction_of_group_frauds": frauds/total_fraud if total_fraud else None,
            }
        results[kind] = output
    return results


def audit(source, split, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("N2 audit output already exists; refusing overwrite")
    manifest, _, X, y = validate_inputs(source, split)
    train_end, val_end = manifest["split_end_rows"][:2]
    reports = {}
    for name, start, end in (
        ("train", 0, train_end), ("validation", train_end, val_end)
    ):
        # Zero-copy slices for only the declared train and validation periods.
        columns = X[start:end]
        n2raw = np.expm1(columns[:, 17].astype(np.float64))
        if not np.all(np.isfinite(n2raw)) or np.any(n2raw < -0.001):
            raise ValueError("Nonfinite or negative N2 count")
        counts = np.rint(n2raw).astype(np.int64)
        if np.max(np.abs(n2raw-counts)) > 0.05:
            raise ValueError("N2 column does not approximate integral past counts")
        bins = np.where(counts == 0, 0, np.where(counts == 1, 1,
                      np.where(counts <= 3, 2, 3))).astype(np.uint8)
        label_slice = y[start:end]
        type_indices = np.argmax(columns[:, 1:6], axis=1)
        one_hot = columns[:, 1:6]
        if not np.all(np.count_nonzero(one_hot, axis=1) == 1):
            raise ValueError("Invalid PaySim type one-hot matrix")
        types = np.asarray(TRANSACTION_TYPES)[type_indices]
        reports[name] = summarize_group(
            bins, label_slice, types, np.ones(len(label_slice), dtype=bool)
        )
        if name == "validation":
            quarters = {}
            edges = np.linspace(0, len(label_slice), 5, dtype=np.int64)
            for i in range(4):
                rowmask = np.zeros(len(label_slice), dtype=bool)
                rowmask[edges[i]:edges[i+1]] = True
                quarters[f"quarter_{i+1}"] = summarize_group(
                    bins, label_slice, types, rowmask
                )
            reports["validation_quarters"] = quarters
    out = {
        "status": "EXPLORATORY_N2_AGGREGATE_DIAGNOSTIC",
        "n2_definition": "distinct prior originators at current destination, full history",
        "bins": list(BINS),
        "train_rows": train_end,
        "validation_rows": val_end-train_end,
        "results": reports,
        "warning": (
            "Previously inspected simulator validation; not causal proof, "
            "not independent confirmation or a deployable policy."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create also prevents overwriting a file created concurrently.
    with output.open("x") as handle:
        json.dump(out, handle, indent=2)
        handle.write("\n")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("data/processed/full_context_dev_v1"))
    p.add_argument("--split", type=Path, default=Path("reports/split_manifest.json"))
    p.add_argument("--output", type=Path, default=Path("data/processed/full_context_n2_audit_v1.json"))
    args = p.parse_args()
    result = audit(args.source, args.split, args.output)
    for period in ("train", "validation"):
        print(period)
        for group, counts in result["results"][period]["ALL"].items():
            print(f"  {group}: {counts['frauds']}/{counts['rows']} frauds/rows")
    print("Aggregate report:", args.output)


if __name__ == "__main__":
    main()
