"""Label-blind compatibility audit for the two MoMTSim CSV datasets.

This reads transaction rows to assess structure and balance semantics, but
never accesses fraud-label *values*, fits models, scores, or writes results.
Only aggregated statistics and file hashes are printed. Do not log account IDs.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED = (
    "step", "transactionType", "amount", "initiator",
    "oldBalInitiator", "newBalInitiator", "recipient",
    "oldBalRecipient", "newBalRecipient", "isFraud",
)
BALANCE_FIELDS = (
    "oldBalInitiator", "newBalInitiator",
    "oldBalRecipient", "newBalRecipient",
)
PAY_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}


def _sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _near(a, b):
    # Accounting decimals should agree to 0.02 absolute units.
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.02)


def audit(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("CSV must be a regular, non-symlink file")
    result = {
        "file": path.name,
        "sha256": _sha256(path),
        "byte_size": path.stat().st_size,
        "rows": 0,
        "step_min": None,
        "step_max": None,
        "distinct_steps": 0,
        "out_of_order_rows": 0,
        "invalid_step_rows": 0,
        "invalid_amount_or_balance_rows": 0,
        "blank_identity_rows": 0,
        "same_initiator_recipient_rows": 0,
        "transaction_type_counts": {},
        "balance_patterns_by_type": {},
        "fraud_label_values_inspected": False,
        "fraud_detection_evaluated": False,
        "paysim_frozen_model_validation": False,
    }
    types = Counter()
    patterns = defaultdict(Counter)
    steps = set()
    previous = None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)):
            raise ValueError("Missing or duplicated CSV header")
        if any(name not in headers for name in EXPECTED):
            raise ValueError("Missing MoMTSim columns")
        for row in reader:
            result["rows"] += 1
            category = (row.get("transactionType") or "").strip()
            types[category] += 1
            original = (row.get("initiator") or "").strip()
            dest = (row.get("recipient") or "").strip()
            if not original or not dest:
                result["blank_identity_rows"] += 1
            if original and dest and original == dest:
                result["same_initiator_recipient_rows"] += 1

            try:
                raw = row["step"].strip()
                step = int(raw)
                if step < 0 or raw != str(step):
                    raise ValueError("Invalid step")
            except (ValueError, TypeError, AttributeError):
                result["invalid_step_rows"] += 1
            else:
                if previous is not None and step < previous:
                    result["out_of_order_rows"] += 1
                previous = step
                steps.add(step)
                lo = result["step_min"]
                hi = result["step_max"]
                result["step_min"] = step if lo is None else min(lo, step)
                result["step_max"] = step if hi is None else max(hi, step)

            try:
                amount = float(row["amount"])
                bal = [float(row[c]) for c in BALANCE_FIELDS]
                if not all(math.isfinite(v) and v >= 0 for v in [amount] + bal):
                    raise ValueError("Invalid amount or balance")
            except (ValueError, TypeError, OverflowError):
                result["invalid_amount_or_balance_rows"] += 1
                continue
            oi, ni, od, nd = bal
            p = patterns[category]
            p["numeric_rows"] += 1
            if _near(ni - oi, 0):
                p["sender_balance_unchanged"] += 1
            if _near(ni - oi, -amount):
                p["sender_balance_decreased_by_amount"] += 1
            if _near(ni - oi, amount):
                p["sender_balance_increased_by_amount"] += 1
            if _near(nd - od, 0):
                p["recipient_balance_unchanged"] += 1
            if _near(nd - od, amount):
                p["recipient_balance_increased_by_amount"] += 1
            if _near(nd - od, -amount):
                p["recipient_balance_decreased_by_amount"] += 1

    result["distinct_steps"] = len(steps)
    result["transaction_type_counts"] = dict(sorted(types.items()))
    result["balance_patterns_by_type"] = {
        name: dict(sorted(patterns[name].items())) for name in sorted(types)
    }
    result["structural_issues_detected"] = any(result[k] for k in (
        "out_of_order_rows", "invalid_step_rows",
        "invalid_amount_or_balance_rows", "blank_identity_rows",
    )) or not result["rows"] or "" in types
    result["paySim_type_names_match_exactly"] = (
        set(types) == PAY_TYPES
    )
    result["step_zero_present"] = 0 in steps
    result["status"] = (
        "STRUCTURAL_AUDIT_REQUIRES_REVIEW" if result["structural_issues_detected"]
        else "STRUCTURAL_AUDIT_PASSED_SEMANTICS_UNVERIFIED"
    )
    result["limitations"] = [
        "Never opened fraud-label values; no accuracy or class-prevalence result.",
        "Transaction names and participant roles require domain-semantic review.",
        "PaySim FullContextHistory expects step >= 1 and five exact type names.",
        "Any offset/type mapping must be explicitly justified before scoring.",
        "Independent fraud testing requires separate labels-only evaluation gate.",
        "Source files must be pinned and the original PaySim model remain frozen.",
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"datasets": [audit(path) for path in args.csv]},
                     indent=2))


if __name__ == "__main__":
    main()
