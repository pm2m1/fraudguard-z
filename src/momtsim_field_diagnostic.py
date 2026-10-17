"""Read-only, label-blind field diagnostic for independent MoMTSim candidates.

Counts invalid amounts, old balances, and new balances separately. Does not
access row['isFraud'], does not print any account IDs or individual records,
does not score or retrain any model, and does not create output files.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED = (
    "step", "transactionType", "amount", "initiator",
    "oldBalInitiator", "newBalInitiator", "recipient",
    "oldBalRecipient", "newBalRecipient", "isFraud",
)
NUMERIC = (
    "amount", "oldBalInitiator", "newBalInitiator",
    "oldBalRecipient", "newBalRecipient",
)
MODEL_NUMERIC = ("amount", "oldBalInitiator")
PRE_BALANCE_FIELDS = ("oldBalInitiator", "oldBalRecipient")
POST_BALANCE_FIELDS = ("newBalInitiator", "newBalRecipient")


def numeric_status(raw):
    """Return validity class, never raw data."""
    try:
        if raw is None or not str(raw).strip():
            return "missing", None
        value = float(raw)
    except (ValueError, TypeError, OverflowError):
        return "unparseable", None
    if not math.isfinite(value):
        return "nonfinite", None
    if value < 0:
        return "negative", value
    return "ok", value


def field_diagnostic(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Source must be an existing regular non-symlink CSV")
    digest = hashlib.sha256()
    with path.open("rb") as binary:
        for block in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(block)

    types = Counter()
    invalid_fields = {name: Counter() for name in NUMERIC}
    invalid_by_type = defaultdict(Counter)
    rows = Counter()
    step_values = set()
    previous_step = None
    step_min = step_max = None
    positive_amount_by_type = Counter()
    sender_delta_by_type = defaultdict(Counter)

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)):
            raise ValueError("Missing or duplicate CSV header")
        if any(name not in headers for name in REQUIRED):
            raise ValueError("Unexpected MoMTSim CSV schema")
        for row in reader:
            rows["total"] += 1
            kind = (row.get("transactionType") or "").strip()
            types[kind] += 1
            statuses = {}
            values = {}
            for name in NUMERIC:
                status, value = numeric_status(row.get(name))
                statuses[name] = status
                values[name] = value
                invalid_fields[name][status] += 1
                if status == "ok" and value == 0:
                    invalid_fields[name]["zero_valid"] += 1

            invalid = {name for name in NUMERIC if statuses[name] != "ok"}
            if invalid:
                rows["any_invalid_numeric"] += 1
                invalid_by_type[kind]["any_invalid_numeric"] += 1

            input_bad = bool(invalid.intersection(MODEL_NUMERIC))
            pre_bad = bool(invalid.intersection(PRE_BALANCE_FIELDS))
            post_bad = bool(invalid.intersection(POST_BALANCE_FIELDS))
            if input_bad:
                rows["invalid_sender_model_inputs"] += 1
                invalid_by_type[kind]["invalid_sender_model_inputs"] += 1
            else:
                rows["valid_sender_model_numeric_inputs"] += 1
            if pre_bad:
                rows["invalid_any_pretransaction_balance"] += 1
            if post_bad:
                rows["invalid_any_posttransaction_balance"] += 1
            if post_bad and not input_bad and not pre_bad:
                rows["invalid_only_posttransaction_balance"] += 1
            if invalid and not input_bad:
                rows["invalid_but_sender_model_inputs_valid"] += 1
            if all(statuses[name] == "ok" for name in NUMERIC):
                rows["all_numeric_valid"] += 1

            try:
                raw_step = row["step"].strip()
                step = int(raw_step)
                if raw_step != str(step) or step < 0:
                    raise ValueError("Invalid step")
            except (ValueError, TypeError, AttributeError):
                rows["invalid_step"] += 1
            else:
                if previous_step is not None and step < previous_step:
                    rows["out_of_order_step"] += 1
                previous_step = step
                if step == 0:
                    rows["step_zero"] += 1
                step_values.add(step)
                step_min = step if step_min is None else min(step_min, step)
                step_max = step if step_max is None else max(step_max, step)

            if all(statuses[n] == "ok" for n in (
                "amount", "oldBalInitiator", "newBalInitiator",
            )) and values["amount"] > 0:
                positive_amount_by_type[kind] += 1
                delta = values["newBalInitiator"] - values["oldBalInitiator"]
                amount = values["amount"]
                if math.isclose(delta, -amount, abs_tol=0.02, rel_tol=1e-9):
                    classification = "sender_decreased_by_amount"
                elif math.isclose(delta, amount, abs_tol=0.02, rel_tol=1e-9):
                    classification = "sender_increased_by_amount"
                elif math.isclose(delta, 0, abs_tol=0.02, rel_tol=1e-9):
                    classification = "sender_unchanged"
                else:
                    classification = "other_balance_change"
                sender_delta_by_type[kind][classification] += 1
            # Intentionally NEVER access row['isFraud'].

    return {
        "status": "LABEL_BLIND_FIELD_DIAGNOSTIC_SEMANTICS_UNVERIFIED",
        "dataset_file": path.name,
        "source_sha256": digest.hexdigest(),
        "rows": rows["total"],
        "step_range": [step_min, step_max],
        "distinct_steps": len(step_values),
        "step_zero_rows": rows["step_zero"],
        "invalid_step_rows": rows["invalid_step"],
        "out_of_order_rows": rows["out_of_order_step"],
        "type_counts": dict(sorted(types.items())),
        "any_invalid_numeric_rows": rows["any_invalid_numeric"],
        "all_numeric_valid_rows": rows["all_numeric_valid"],
        "invalid_sender_model_input_rows": rows["invalid_sender_model_inputs"],
        "valid_sender_model_numeric_input_rows": rows["valid_sender_model_numeric_inputs"],
        "invalid_any_pretransaction_balance_rows":
            rows["invalid_any_pretransaction_balance"],
        "invalid_any_posttransaction_balance_rows":
            rows["invalid_any_posttransaction_balance"],
        "invalid_only_posttransaction_balance_rows":
            rows["invalid_only_posttransaction_balance"],
        "invalid_but_sender_model_inputs_valid_rows":
            rows["invalid_but_sender_model_inputs_valid"],
        "numeric_column_status_counts": {
            name: dict(sorted(invalid_fields[name].items())) for name in NUMERIC
        },
        "invalid_by_type": {
            name: dict(sorted(invalid_by_type[name].items())) for name in sorted(types)
        },
        "positive_amount_sender_balance_change_by_type": {
            name: {
                "valid_positive_amount_rows": positive_amount_by_type[name],
                **dict(sorted(sender_delta_by_type[name].items())),
            } for name in sorted(types)
        },
        "fraud_labels_accessed": False,
        "frozen_paysim_model_scored": False,
        "external_validation_complete": False,
        "warnings": [
            "Invalid post-transaction balance need not imply invalid frozen-model input.",
            "The frozen 21-feature PaySim engine additionally requires step >= 1,",
            "five exact PaySim types, stable participant roles, and valid account history.",
            "No automatic mapping from DEPOSIT/WITHDRAWAL to CASH_IN/CASH_OUT is approved.",
            "This diagnostic does not certify compatibility or label definitions.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"datasets": [field_diagnostic(p) for p in args.csv]},
                     indent=2))


if __name__ == "__main__":
    main()
