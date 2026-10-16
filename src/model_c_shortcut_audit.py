"""Audit a suspected PaySim fraud-generation shortcut on train/validation only.

No models are trained or changed. Never opens rows later than step 306.
The exact-balance heuristic is a diagnostic, not a deployment rule.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_DATA = Path("data/raw/PS_20174392719_1491204439457_log.csv")
DEFAULT_REPORT = Path("reports/model_c_shortcut_audit.json")
COLUMNS = {"step", "type", "amount", "oldbalanceOrg", "isFraud"}


def exact_positive_balance_match(amount, oldbalance):
    amount = float(amount)
    oldbalance = float(oldbalance)
    return oldbalance > 0 and abs(amount - oldbalance) < 0.01


def new_counter():
    return {"transactions": 0, "frauds": 0, "matching_transactions": 0,
            "matching_frauds": 0, "matching_legitimate": 0}


def audit(path):
    result = {
        "description": "Exact positive sender-balance match diagnostic",
        "rule": "oldbalanceOrg > 0 and abs(amount-oldbalanceOrg) < 0.01",
        "warning": (
            "A synthetic-data shortcut is not independent fraud evidence. "
            "Only train steps 1-238 and validation steps 239-306 are inspected. "
            "The original final holdout is not read."
        ),
        "train": {"ALL": new_counter()},
        "validation": {"ALL": new_counter()},
    }
    previous_step = -1
    with Path(path).open(newline="") as file:
        reader = csv.DictReader(file)
        if not COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError("CSV is missing required columns")
        for row in reader:
            step = int(row["step"])
            if step > 306:
                break
            if step < previous_step:
                raise ValueError("Data must be step-ordered")
            previous_step = step
            label = int(row["isFraud"])
            if label not in (0, 1):
                raise ValueError("Fraud label must be 0 or 1")
            match = exact_positive_balance_match(row["amount"], row["oldbalanceOrg"])
            partition = "train" if step <= 238 else "validation"
            group = result[partition]
            kind = row["type"]
            if kind not in group:
                group[kind] = new_counter()
            for counter in (group["ALL"], group[kind]):
                counter["transactions"] += 1
                counter["frauds"] += label
                counter["matching_transactions"] += int(match)
                counter["matching_frauds"] += int(match and label == 1)
                counter["matching_legitimate"] += int(match and label == 0)
    for partition in ("train", "validation"):
        for counter in result[partition].values():
            count = counter["matching_transactions"]
            frauds = counter["frauds"]
            counter["precision_of_rule"] = (
                counter["matching_frauds"] / count if count else None
            )
            counter["recall_of_rule"] = (
                counter["matching_frauds"] / frauds if frauds else None
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = audit(args.csv)
    for partition in ("train", "validation"):
        print(f"\n{partition.upper()} (no data after step 306)")
        for kind in ("ALL", "TRANSFER", "CASH_OUT"):
            entry = result[partition].get(kind)
            if entry is None:
                continue
            print(f"{kind}: {entry['matching_frauds']}/{entry['frauds']} frauds "
                  f"match, {entry['matching_legitimate']} legitimate matches")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nSaved {args.report}")
    print("No model, calibrated policy, or held-out test changed.")


if __name__ == "__main__":
    main()
