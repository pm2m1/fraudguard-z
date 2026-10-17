"""Conservative label-blind MoMTSim replay-coverage screen.

Never reads fraud labels, evaluates a frozen model or alters inputs.
Taint propagates through participant links after a row that cannot be
represented by the frozen PaySim event validator. The resulting subset is
only a deliberately conservative structural bound, NOT approved coverage.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

PINNED_SHA256 = {
    "MoMTSim_20240722202413_1000_dataset.csv":
        "99fd07c3a9d3c4bd6d3462240058ca19d0d9e9284683f78bf77542ff7fcc05e7",
    "synthetic_mobile_money_transaction_dataset.csv":
        "da951eb95735da96271740a3e66b676b342d3831ce3111cd19dbfa020d3bd0a7",
}
REQUIRED = (
    "step", "transactionType", "amount", "initiator",
    "oldBalInitiator", "recipient", "isFraud",
)
SOURCE_TYPES = {"DEBIT", "DEPOSIT", "PAYMENT", "TRANSFER", "WITHDRAWAL"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_input_valid(row):
    try:
        amount = float(row["amount"])
        balance = float(row["oldBalInitiator"])
    except (TypeError, ValueError, OverflowError):
        return False
    return (math.isfinite(amount) and math.isfinite(balance)
            and amount >= 0 and balance >= 0)


def lineage_audit(path, *, expected_sha256=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Must supply an existing regular non-symlink CSV")
    if expected_sha256 is None:
        expected_sha256 = PINNED_SHA256.get(path.name)
    if expected_sha256 is None or sha256_file(path) != expected_sha256:
        raise RuntimeError("Unapproved or changed MoMTSim source SHA-256")

    aggregate = Counter()
    by_type = {}
    tainted = set()
    last_step = -1
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames
        if (not header or len(header) != len(set(header))
                or any(name not in header for name in REQUIRED)):
            raise RuntimeError("Unexpected MoMTSim CSV schema")
        for row in reader:
            aggregate["rows"] += 1
            kind = row["transactionType"].strip()
            if kind not in SOURCE_TYPES:
                raise RuntimeError("Unapproved MoMTSim transaction type")
            counts = by_type.setdefault(kind, Counter())
            counts["rows"] += 1
            sender = row["initiator"].strip()
            receiver = row["recipient"].strip()
            if not sender or not receiver:
                raise RuntimeError("Empty MoMTSim account identity")
            raw_step = row["step"].strip()
            try:
                step = int(raw_step)
                if step < last_step or step < 0 or str(step) != raw_step:
                    raise ValueError("Step out of order")
            except (ValueError, TypeError) as exc:
                raise RuntimeError("Unexpected MoMTSim chronology") from exc
            last_step = step
            if step == 0:
                aggregate["step_zero_rows"] += 1

            direct_bad = not numeric_input_valid(row)
            prior_taint = (sender in tainted or receiver in tainted)
            if direct_bad:
                key = "direct_invalid_sender_input"
                tainted.add(sender)
                tainted.add(receiver)
            elif prior_taint:
                key = "history_affected_after_invalid_event"
                tainted.add(sender)
                tainted.add(receiver)
            else:
                key = "no_known_invalid_lineage"
            aggregate[key] += 1
            counts[key] += 1
            # Deliberately do not access row["isFraud"].

    if aggregate["rows"] != sum(aggregate[k] for k in (
        "direct_invalid_sender_input",
        "history_affected_after_invalid_event",
        "no_known_invalid_lineage",
    )):
        raise RuntimeError("Coverage accounting mismatch")
    return {
        "status": "LABEL_BLIND_CONSERVATIVE_LINEAGE_AUDIT_NOT_VALIDATION",
        "source_file": path.name,
        "source_sha256": expected_sha256,
        "rows": aggregate["rows"],
        "direct_invalid_sender_input_rows":
            aggregate["direct_invalid_sender_input"],
        "history_affected_after_invalid_event_rows":
            aggregate["history_affected_after_invalid_event"],
        "no_known_invalid_lineage_rows":
            aggregate["no_known_invalid_lineage"],
        "structural_no_known_invalid_lineage_fraction": (
            aggregate["no_known_invalid_lineage"] / aggregate["rows"]
            if aggregate["rows"] else None
        ),
        "step_zero_rows": aggregate["step_zero_rows"],
        "tainted_accounts_aggregate_count": len(tainted),
        "by_source_transaction_type": {
            k: dict(sorted(v.items())) for k, v in sorted(by_type.items())
        },
        "fraud_labels_accessed": False,
        "frozen_paysim_model_scored": False,
        "external_validation_complete": False,
        "limitations": [
            "Deliberately conservative graph-contamination heuristic, not proof of eligibility.",
            "Unapproved DEPOSIT/WITHDRAWAL mapping and participant semantics remain blockers.",
            "Step zero still requires a justified global offset before any feature replay.",
            "The PaySim model is never run; no labels or decision thresholds are accessed.",
            "No numeric repair, history mutation, row dropping, or outcome selection is performed.",
            "A high clean-lineage fraction would not establish external validity.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"datasets": [lineage_audit(p) for p in args.csv]},
                     indent=2))


if __name__ == "__main__":
    main()
