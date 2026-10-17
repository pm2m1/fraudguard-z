"""Read-only structural audit of an external BankSim CSV.

This tool does NOT inspect fraud labels, train a model, load model artifacts,
score transactions, or establish independent validation of FraudGuard Z.
Never log customer/merchant IDs or individual transactions.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from math import isfinite
from pathlib import Path

REQUIRED = (
    "step", "customer", "age", "gender", "zipcodeOri",
    "merchant", "zipMerchant", "category", "amount", "fraud",
)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("BankSim CSV must be an existing regular non-symlink file")

    rows = 0
    categories = Counter()
    steps = set()
    invalid_step = invalid_amount = blank_identity = 0
    order_violations = 0
    first_step = last_step = previous_step = None
    min_amount = max_amount = None

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError("Missing or duplicate CSV headers")
        missing = sorted(set(REQUIRED) - set(fields))
        if missing:
            raise ValueError("Missing BankSim fields: " + ", ".join(missing))

        for row in reader:
            rows += 1
            try:
                step = int(row["step"])
                if step < 0 or str(step) != row["step"].strip():
                    raise ValueError("Malformed step")
            except (ValueError, TypeError, AttributeError):
                invalid_step += 1
            else:
                steps.add(step)
                if previous_step is not None and step < previous_step:
                    order_violations += 1
                previous_step = step
                first_step = step if first_step is None else min(first_step, step)
                last_step = step if last_step is None else max(last_step, step)
            try:
                amount = float(row["amount"])
                if not isfinite(amount) or amount < 0:
                    raise ValueError("Nonfinite or negative amount")
            except (TypeError, ValueError, OverflowError):
                invalid_amount += 1
            else:
                min_amount = amount if min_amount is None else min(min_amount, amount)
                max_amount = amount if max_amount is None else max(max_amount, amount)
            if not row.get("customer") or not row.get("merchant"):
                blank_identity += 1
            category = row.get("category") or ""
            categories[category] += 1
            # Deliberately DO NOT access row["fraud"] here.
    valid = rows > 0 and not (invalid_step or invalid_amount or blank_identity or
                              order_violations or "" in categories)
    return {
        "status": ("STRUCTURAL_AUDIT_PASSED_NOT_VALIDATION" if valid
                   else "STRUCTURAL_AUDIT_REQUIRES_REVIEW"),
        "dataset_kind": "BANKSIM_EXTERNAL_DOMAIN",
        "sha256": file_digest(path),
        "rows": rows,
        "distinct_steps": len(steps),
        "min_step": first_step,
        "max_step": last_step,
        "nonmonotonic_step_rows": order_violations,
        "invalid_step_rows": invalid_step,
        "invalid_amount_rows": invalid_amount,
        "blank_customer_or_merchant_rows": blank_identity,
        "min_amount": min_amount,
        "max_amount": max_amount,
        "category_counts": dict(sorted(categories.items())),
        "fraud_label_values_inspected": False,
        "independent_validation_of_frozen_paysim_model": False,
        "warning": (
            "BankSim categories, customer-to-merchant relationships, and step"
            " semantics cannot be silently treated as PaySim transaction"
            " types, account-to-account transfers, or hourly steps. This"
            " structural audit does not evaluate fraud-detection quality."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.csv), indent=2))


if __name__ == "__main__":
    main()
