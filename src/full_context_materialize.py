"""Bounded, atomic materialization of the FROZEN PaySim development split.

This is feature generation only: no model fit, calibration, policy choice,
or final-holdout evaluation. It will refuse any existing destination path.
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .full_context import FULL_FEATURE_NAMES
from .full_context_dataset import (
    DEVELOPMENT_END_STEP,
    iter_feature_records,
    iter_labelled_development_csv,
)

DEFAULT_CSV = Path("data/raw/PS_20174392719_1491204439457_log.csv")
DEFAULT_MANIFEST = Path("reports/split_manifest.json")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest):
    if manifest.get("dev_max_step") != DEVELOPMENT_END_STEP:
        raise ValueError("Development end must remain step 372")
    if manifest.get("final_holdout_start_step") != 373:
        raise ValueError("Unrecognized frozen holdout boundary")
    boundaries = manifest.get("split_end_steps")
    ends = manifest.get("split_end_rows")
    expected = manifest.get("development_row_count")
    expected_partitions = manifest.get("partitions")
    if (
        not isinstance(boundaries, list)
        or len(boundaries) != 3
        or any(not isinstance(x, int) or isinstance(x, bool) for x in boundaries)
        or not (0 < boundaries[0] < boundaries[1] < boundaries[2] < DEVELOPMENT_END_STEP)
        or not isinstance(ends, list)
        or len(ends) != 3
        or any(not isinstance(x, int) or isinstance(x, bool) for x in ends)
        or not isinstance(expected, int)
        or isinstance(expected, bool)
        or not (0 < ends[0] < ends[1] < ends[2] < expected)
        or not isinstance(expected_partitions, dict)
    ):
        raise ValueError("Invalid frozen split manifest")
    rows = [ends[0], ends[1]-ends[0], ends[2]-ends[1], expected-ends[2]]
    names = ("train", "validation", "calibration", "policy")
    for name, count in zip(names, rows):
        item = expected_partitions.get(name)
        if (not isinstance(item, dict)
                or item.get("rows") != count
                or not isinstance(item.get("frauds"), int)
                or not 0 <= item["frauds"] <= count):
            raise ValueError("Manifest partition count or fraud count invalid")
    return names, (boundaries + [DEVELOPMENT_END_STEP]), rows


def materialize(csv_path, manifest_path, destination):
    csv_path = Path(csv_path)
    manifest_path = Path(manifest_path)
    destination = Path(destination)
    if not csv_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("CSV and frozen split manifest are required")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    manifest = json.loads(manifest_path.read_text())
    names, end_steps, expected_rows = validate_manifest(manifest)
    total = manifest["development_row_count"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".fgz_build_", dir=destination.parent))
    try:
        features = open_memmap(
            tmp / "features.npy", mode="w+", dtype=np.float32,
            shape=(total, len(FULL_FEATURE_NAMES)),
        )
        labels = open_memmap(
            tmp / "labels.npy", mode="w+", dtype=np.uint8, shape=(total,)
        )
        counts = [0, 0, 0, 0]
        frauds = [0, 0, 0, 0]
        first_step = None
        last_step = None
        seen = 0
        for cursor, values, label in iter_feature_records(
            iter_labelled_development_csv(csv_path)
        ):
            if seen >= total:
                raise ValueError("Development CSV has more rows than frozen manifest")
            step = cursor[0]
            if first_step is None:
                first_step = step
            last_step = step
            partition = next((i for i, end in enumerate(end_steps) if step <= end), None)
            if partition is None:
                raise ValueError("Event exceeds development boundary")
            if partition != next(
                (i for i, n in enumerate(
                    (manifest["split_end_rows"] + [total])
                ) if seen < n), None
            ):
                raise ValueError("Chronological partition row boundary mismatch")
            features[seen] = values
            labels[seen] = label
            counts[partition] += 1
            frauds[partition] += label
            seen += 1
        if seen != total or counts != expected_rows:
            raise ValueError("Development rows do not match frozen manifest")
        for name, actual in zip(names, frauds):
            if actual != manifest["partitions"][name]["frauds"]:
                raise ValueError(f"Fraud count mismatch in {name}")
        features.flush()
        labels.flush()
        del features, labels
        info = {
            "status": "DEVELOPMENT_ONLY_FEATURE_MATERIALIZATION",
            "source_csv_sha256": file_sha256(csv_path),
            "split_manifest_sha256": file_sha256(manifest_path),
            "schema": list(FULL_FEATURE_NAMES),
            "schema_sha256": hashlib.sha256(
                json.dumps(FULL_FEATURE_NAMES, separators=(",", ":")).encode()
            ).hexdigest(),
            "dtype": "float32",
            "label_dtype": "uint8",
            "rows": seen,
            "first_step": first_step,
            "last_step": last_step,
            "partitions": {
                name: {"rows": count, "frauds": fraud}
                for name, count, fraud in zip(names, counts, frauds)
            },
            "warning": (
                "Previously inspected PaySim holdout (step >= 373) not evaluated. "
                "Development matrix only; not independent model confirmation."
            ),
        }
        (tmp / "materialization.json").write_text(
            json.dumps(info, indent=2) + "\n"
        )
        os.replace(tmp, destination)
        return info
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/full_context_dev_v1")
    )
    args = parser.parse_args()
    result = materialize(args.csv, args.manifest, args.output)
    print("Saved development matrix:", args.output)
    print("Rows:", result["rows"])
    print("This has NOT trained or evaluated a model.")


if __name__ == "__main__":
    main()
