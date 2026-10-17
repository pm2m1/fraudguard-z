"""BankSim-native chronological materialization; label-blind on test rows.

Creates numeric past-only features for all rows, but reads/writes labels
ONLY for steps 0..161. The final-test labels at steps 162..179 are not
accessed. Outputs contain no account or merchant identifiers.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .banksim_structural_audit import REQUIRED, file_digest

SOURCE = Path("data/external/banksim/bs140513_032310.csv")
OUTPUT = Path("data/processed/banksim_native_prepare_v1")
SOURCE_SHA256 = "e37006f76d993bfaec3a02d717b4f0bdc1ebfa5d36449e91fb7c07e225278377"
SOURCE_ROWS = 594643
CATEGORIES = (
    "'es_barsandrestaurants'", "'es_contents'", "'es_fashion'",
    "'es_food'", "'es_health'", "'es_home'", "'es_hotelservices'",
    "'es_hyper'", "'es_leisure'", "'es_otherservices'",
    "'es_sportsandtoys'", "'es_tech'", "'es_transportation'",
    "'es_travel'", "'es_wellnessandbeauty'",
)
HISTORY_NAMES = (
    "log_customer_prior_count", "log_merchant_prior_count",
    "log_pair_prior_count", "log_customer_prior_mean_amount",
    "log_merchant_prior_mean_amount", "log_customer_recent_7_step_count",
    "log_merchant_recent_7_step_count", "log_pair_recent_7_step_count",
)
FEATURE_NAMES = ("log_amount",) + tuple("category:" + c for c in CATEGORIES) + HISTORY_NAMES
BOUNDARIES = (108, 126, 144, 162, 180)
PARTITIONS = ("train", "validation", "calibration", "policy", "final_test")
SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(FEATURE_NAMES, separators=(",", ":")).encode()
).hexdigest()


class History:
    def __init__(self):
        self.counts = [Counter(), Counter(), Counter()]
        self.sums = [defaultdict(float), defaultdict(float)]
        self.recent = [defaultdict(deque) for _ in range(3)]
        self.category_index = {c: i for i, c in enumerate(CATEGORIES)}

    def vector(self, *, step, customer, merchant, category, amount):
        if category not in self.category_index:
            raise ValueError("Unexpected BankSim category")
        keys = (customer, merchant, (customer, merchant))
        previous = [self.counts[i][key] for i, key in enumerate(keys)]
        recent_counts = []
        for i, key in enumerate(keys):
            events = self.recent[i][key]
            while events and events[0] <= step - 7:
                events.popleft()
            recent_counts.append(len(events))
        customer_mean = (self.sums[0][customer] / previous[0]
                         if previous[0] else 0.0)
        merchant_mean = (self.sums[1][merchant] / previous[1]
                         if previous[1] else 0.0)
        values = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        values[0] = math.log1p(amount)
        values[1 + self.category_index[category]] = 1.0
        values[16:] = np.log1p(np.asarray([
            *previous, customer_mean, merchant_mean, *recent_counts,
        ], dtype=np.float64))
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite feature")
        return values

    def observe(self, *, step, customer, merchant, amount):
        keys = (customer, merchant, (customer, merchant))
        for i, key in enumerate(keys):
            self.counts[i][key] += 1
            self.recent[i][key].append(step)
        self.sums[0][customer] += amount
        self.sums[1][merchant] += amount


def prepare(source=SOURCE, output=OUTPUT, *,
            expected_sha256=SOURCE_SHA256, expected_rows=SOURCE_ROWS):
    source, output = Path(source), Path(output)
    if source.is_symlink() or not source.is_file():
        raise ValueError("Missing regular BankSim input file")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refusing to overwrite materialized output")
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or not isinstance(expected_rows, int) or expected_rows <= 0):
        raise ValueError("Invalid frozen source contract")
    digest = file_digest(source)
    if digest != expected_sha256:
        raise RuntimeError("BankSim file checksum does not match frozen audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".banksim_prepare_", dir=output.parent) as tmp:
        staging = Path(tmp)
        features = open_memmap(
            staging / "features.npy", mode="w+", dtype=np.float32,
            shape=(expected_rows, len(FEATURE_NAMES)),
        )
        history = History()
        dev_labels = []
        counts = Counter()
        prior_step = -1
        rows = 0

        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if (not fields or len(fields) != len(set(fields))
                    or set(REQUIRED).difference(fields)):
                raise ValueError("Unexpected BankSim CSV header")
            for row in reader:
                if rows >= expected_rows:
                    raise RuntimeError("Unexpected excess BankSim rows")
                try:
                    step_text = row["step"]
                    step = int(step_text)
                    amount = float(row["amount"])
                    customer, merchant = row["customer"], row["merchant"]
                    category = row["category"]
                    if (str(step) != step_text.strip() or not 0 <= step < 180
                            or step < prior_step or not math.isfinite(amount)
                            or amount < 0 or not customer or not merchant):
                        raise ValueError("Bad chronology or BankSim input")
                except (TypeError, ValueError, AttributeError) as exc:
                    raise ValueError(f"Invalid structural row index {rows}") from exc

                # Feature construction never sees current fraud label.
                features[rows] = history.vector(
                    step=step, customer=customer, merchant=merchant,
                    category=category, amount=amount,
                )
                history.observe(
                    step=step, customer=customer, merchant=merchant,
                    amount=amount,
                )
                partition = next(
                    name for bound, name in zip(BOUNDARIES, PARTITIONS)
                    if step < bound
                )
                counts[partition] += 1
                if step < 162:
                    # Never access row["fraud"] for final-test steps.
                    label = row["fraud"]
                    if label not in ("0", "1"):
                        raise ValueError(f"Invalid development label at row {rows}")
                    dev_labels.append(int(label))
                previous = prior_step
                prior_step = step
                rows += 1
        if rows != expected_rows:
            raise RuntimeError("BankSim row count mismatch")
        features.flush()
        del features
        np.save(staging / "development_labels.npy",
                np.asarray(dev_labels, dtype=np.uint8), allow_pickle=False)
        expected_dev = sum(counts[k] for k in PARTITIONS[:-1])
        if len(dev_labels) != expected_dev:
            raise RuntimeError("Development label split mismatch")
        metadata = {
            "status": "BANKSIM_PREPARED_NON_TEST_LABELS_ONLY",
            "source_sha256": digest,
            "rows": rows,
            "feature_names": list(FEATURE_NAMES),
            "feature_schema_sha256": SCHEMA_SHA256,
            "feature_shape": [rows, len(FEATURE_NAMES)],
            "partition_rows": {p: counts[p] for p in PARTITIONS},
            "development_labels_rows": len(dev_labels),
            "final_labels_accessed": False,
            "banking_claims": False,
            "warning": "Separate BankSim model experiment; no frozen PaySim model validation.",
        }
        (staging / "materialization.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        # Rename the complete private directory atomically; no overwrite.
        os.replace(staging, output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = prepare(args.csv, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
