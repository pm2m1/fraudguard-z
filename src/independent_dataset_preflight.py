"""Read-only CSV header screen for independent FraudGuard Z evaluation.

Matching field names is ONLY a prerequisite. It does NOT establish semantic
compatibility, data provenance, permissions, independent labels, nonoverlap,
chronological integrity, or out-of-distribution generalization.
"""
import argparse
import csv
import json
from pathlib import Path

REQUIRED = (
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "nameDest", "isFraud",
)


def inspect_headers(fieldnames):
    if not fieldnames or any(not isinstance(f, str) or not f for f in fieldnames):
        raise ValueError("Missing or invalid CSV header")
    duplicates = sorted({f for f in fieldnames if fieldnames.count(f) > 1})
    missing = sorted(set(REQUIRED).difference(fieldnames))
    return {
        "status": ("SCHEMA_ONLY_CANDIDATE_REQUIRES_INDEPENDENT_REVIEW"
                   if not missing and not duplicates
                   else "UNSUITABLE_FOR_UNMODIFIED_FROZEN_MODEL"),
        "required": list(REQUIRED),
        "missing": missing,
        "duplicate_headers": duplicates,
        "schema_fields_present": not missing and not duplicates,
        "independent_fraud_validation": False,
        "warnings": [
            "Header matching is NOT dataset approval or a score.",
            "Verify true pretransaction balances, hourly step units, PaySim-equivalent "
            "transaction types, sender and beneficiary semantics, and binary fraud labels.",
            "Check licensing, provenance, data overlap, chronology and frozen policy "
            "before any actual evaluation.",
            "Never invent oldbalanceOrg, map SAR to fraud, or map unrelated types "
            "to PaySim classes to force model compatibility.",
        ],
    }


def inspect_csv(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("CSV must be an existing regular non-symlink file")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("Empty CSV file") from exc
    return inspect_headers(header)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True,
                        help="Local candidate CSV; reads header only")
    args = parser.parse_args()
    print(json.dumps(inspect_csv(args.csv), indent=2))


if __name__ == "__main__":
    main()
