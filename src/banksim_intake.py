"""BankSim pre-evaluation intake and lock. NEVER reads fraud label values.

Usage:
    python -m src.banksim_intake --inspect
    python -m src.banksim_intake --lock

Inspect verifies CSV structure and chronological step composition only.
Lock stores SHA-256 of source dataset and research protocol; it never
evaluates models, scores, labels, or fraud outcomes. Commit the resulting
reports/banksim_protocol_lock.json BEFORE implementing/running final eval.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

DATA = Path("data/external/banksim/bs140513_032310.csv")
PROTOCOL = Path("notes/banksim_prospective_protocol.md")
LOCK = Path("reports/banksim_protocol_lock.json")
REQUIRED = {"step", "customer", "merchant", "category", "amount", "fraud"}
EXPECTED_ROWS = 594643


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_dataset(path=DATA):
    """Verify non-label metadata only; no fraud value is inspected."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; obtain authorized BankSim CSV first"
        )
    n = 0
    steps = set()
    categories = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not REQUIRED.issubset(fields):
            raise ValueError("Incorrect or duplicate BankSim CSV columns")
        for row in reader:
            n += 1
            step = int(row["step"])
            if step < 0:
                raise ValueError("Negative transaction step")
            amount = float(row["amount"])
            if not (0 <= amount < float("inf")):
                raise ValueError("Invalid transaction amount")
            if not row["customer"] or not row["merchant"] or not row["category"]:
                raise ValueError("Missing customer, merchant, or category")
            steps.add(step)
            categories.add(row["category"])
    if n != EXPECTED_ROWS:
        raise ValueError(
            f"Unexpected BankSim transaction count {n}; expected {EXPECTED_ROWS}. "
            "Do not lock unknown dataset variants."
        )
    sorted_steps = sorted(steps)
    if len(sorted_steps) < 5:
        raise ValueError("Not enough unique chronological steps")
    train_end = (len(sorted_steps) * 60) // 100
    tune_end = (len(sorted_steps) * 80) // 100
    if not 0 < train_end < tune_end < len(sorted_steps):
        raise ValueError("Cannot construct step-based splits")
    return {
        "rows": n,
        "unique_steps": len(sorted_steps),
        "earliest_step": sorted_steps[0],
        "latest_step": sorted_steps[-1],
        "first_tuning_step": sorted_steps[train_end],
        "first_final_step": sorted_steps[tune_end],
        "number_of_categories_all_periods": len(categories),
        "note": "No fraud label values were read or counted; demographic features unused."
    }


def build_lock(path=DATA):
    metadata = inspect_dataset(path)
    return {
        "lock_status": "PRE_EVALUATION_NO_FRAUD_LABELS_READ",
        "data_path": str(path),
        "dataset_sha256": sha256(path),
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "intake_script_sha256": sha256(Path(__file__).resolve()),
        "schema": sorted(REQUIRED),
        "metadata": metadata,
        "final_data_evaluated": False,
        "note": (
            "This only locks source-data bytes and protocol text. "
            "Evaluators must be separately frozen and reviewed before "
            "running a one-time final evaluation."
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--lock", action="store_true")
    parser.add_argument("--csv", type=Path, default=DATA)
    args = parser.parse_args()
    if args.inspect:
        print(json.dumps(inspect_dataset(args.csv), indent=2))
        return
    if LOCK.exists():
        raise RuntimeError(f"{LOCK} already exists; refusing to overwrite lock")
    document = build_lock(args.csv)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(document, indent=2) + "\n")
    print(f"Locked source and protocol metadata in {LOCK}")
    print("Commit/push the lock before building or running the one-time evaluator.")
    print("No fraud labels have been read or summarized by this tool.")


if __name__ == "__main__":
    main()
