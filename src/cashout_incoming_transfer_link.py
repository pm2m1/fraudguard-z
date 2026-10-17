"""Read-only PaySim development CASH_OUT prior-TRANSFER relationship diagnostic.

Previously inspected train/validation only. No model, policy or live decisions.
Output contains aggregate cohorts; no raw identities or event-level data.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from .full_context_materialize import DEFAULT_CSV, DEFAULT_MANIFEST, file_sha256

OUTPUT = Path("data/processed/cashout_incoming_transfer_link_v1.json")
REQUIRED = frozenset({
    "step", "type", "amount", "oldbalanceOrg", "nameOrig", "nameDest", "isFraud"
})
BANDS = ("le_100", "gt_100_le_1000", "gt_1000")


def amount_band(amount):
    return BANDS[0] if amount <= 100 else BANDS[1] if amount <= 1000 else BANDS[2]


class IncomingTransferLedger:
    """Sliding state containing strictly previously observed TRANSFER recipients."""

    def __init__(self):
        self.q6 = deque()
        self.q24 = deque()
        self.count6 = Counter()
        self.count24 = Counter()
        self.recipient_steps24 = defaultdict(deque)
        self.last_step = 0

    def advance(self, step):
        if step < 1 or step < self.last_step:
            raise ValueError("Nonchronological event step")
        self.last_step = step
        while self.q6 and self.q6[0][0] <= step - 6:
            _, recipient = self.q6.popleft()
            self.count6[recipient] -= 1
            if self.count6[recipient] == 0:
                del self.count6[recipient]
        while self.q24 and self.q24[0][0] <= step - 24:
            oldstep, recipient = self.q24.popleft()
            self.count24[recipient] -= 1
            if self.count24[recipient] == 0:
                del self.count24[recipient]
            per_account = self.recipient_steps24[recipient]
            if not per_account or per_account.popleft() != oldstep:
                raise RuntimeError("Transfer recipient expiry invariant violated")
            if not per_account:
                del self.recipient_steps24[recipient]

    def before(self, step, originator):
        self.advance(step)
        recent = self.recipient_steps24.get(originator)
        age = step - recent[-1] if recent else None
        return self.count6[originator], self.count24[originator], age

    def observe(self, step, kind, destination):
        self.advance(step)
        if kind == "TRANSFER":
            self.q6.append((step, destination))
            self.q24.append((step, destination))
            self.count6[destination] += 1
            self.count24[destination] += 1
            self.recipient_steps24[destination].append(step)


class Counts:
    def __init__(self):
        self.rows = 0
        self.frauds = 0
        self.link6 = [0, 0]
        self.link24 = [0, 0]
        self.count6_buckets = [[0] * 4 for _ in range(2)]
        self.count24_buckets = [[0] * 4 for _ in range(2)]
        self.age24 = [[0] * 24 for _ in range(2)]

    def add(self, label, n6, n24, age):
        self.rows += 1
        self.frauds += label
        if n6:
            self.link6[label] += 1
        if n24:
            self.link24[label] += 1
        self.count6_buckets[label][min(n6, 3)] += 1
        self.count24_buckets[label][min(n24, 3)] += 1
        if age is not None:
            if not 0 <= age < 24:
                raise RuntimeError("Transfer age outside rolling 24-hour window")
            self.age24[label][age] += 1

    def report(self):
        legit = self.rows - self.frauds
        linked6 = sum(self.link6)
        linked24 = sum(self.link24)
        return {
            "rows": self.rows,
            "frauds": self.frauds,
            "fraud_rate": self.frauds / self.rows if self.rows else None,
            "linked_6h": {"rows": linked6, "frauds": self.link6[1],
                "legitimate": self.link6[0],
                "fraud_rate": self.link6[1] / linked6 if linked6 else None},
            "unlinked_6h": {
                "rows": self.rows - linked6,
                "frauds": self.frauds - self.link6[1],
                "fraud_rate": ((self.frauds - self.link6[1]) /
                               (self.rows - linked6)) if self.rows > linked6 else None},
            "linked_24h": {"rows": linked24, "frauds": self.link24[1],
                "legitimate": self.link24[0],
                "fraud_rate": self.link24[1] / linked24 if linked24 else None},
            "unlinked_24h": {
                "rows": self.rows - linked24,
                "frauds": self.frauds - self.link24[1],
                "fraud_rate": ((self.frauds - self.link24[1]) /
                               (self.rows - linked24)) if self.rows > linked24 else None},
            "prior_transfer_count_6h_histogram": {
                "legitimate": self.count6_buckets[0],
                "fraud": self.count6_buckets[1],
                "bins": ["0", "1", "2", "3_plus"]},
            "prior_transfer_count_24h_histogram": {
                "legitimate": self.count24_buckets[0],
                "fraud": self.count24_buckets[1],
                "bins": ["0", "1", "2", "3_plus"]},
            "age_hours_since_last_incoming_transfer_within_24h": {
                "legitimate": self.age24[0], "fraud": self.age24[1],
                "hour_offsets": list(range(24))},
        }


def process_rows(reader, manifest, output):
    ends = manifest["split_end_rows"]
    stop = ends[1]
    if (manifest.get("dev_max_step") != 372
            or manifest.get("final_holdout_start_step") != 373
            or manifest.get("split_end_steps") != [238, 306, 333]
            or ends[:2] != [3177938, 4263279]
            or stop <= 0):
        raise RuntimeError("Unexpected development split contract")

    ledger = IncomingTransferLedger()
    counts = {
        name: {"all": Counts(), "without_balance_shortcut": Counts(),
               "by_amount_band": {band: Counts() for band in BANDS},
               "without_shortcut_by_amount_band":
                   {band: Counts() for band in BANDS}}
        for name in ("train", "validation")
    }
    last_step = 0
    seen = 0
    # 'isFraud' is only consumed for CASH_OUT aggregate diagnostics.
    for i in range(stop):
        try:
            row = next(reader)
        except StopIteration as exc:
            raise RuntimeError("CSV shorter than frozen train+validation") from exc
        step = int(row["step"])
        segment = "train" if i < ends[0] else "validation"
        max_step = 238 if segment == "train" else 306
        min_step = 1 if segment == "train" else 239
        if not min_step <= step <= max_step or step < last_step:
            raise RuntimeError(f"Chronology/split boundary violated at development row {i}")
        last_step = step
        kind = row["type"]
        origin, dest = row["nameOrig"], row["nameDest"]
        if not origin or not dest:
            raise ValueError("Empty PaySim account field")
        if kind == "CASH_OUT":
            label_raw = row["isFraud"]
            if label_raw not in ("0", "1"):
                raise ValueError("Invalid CASH_OUT label")
            label = int(label_raw)
            amount = float(row["amount"])
            balance = float(row["oldbalanceOrg"])
            if not 0 <= amount < float("inf") or not 0 <= balance < float("inf"):
                raise ValueError("Invalid numeric PaySim amount or balance")
            n6, n24, age = ledger.before(step, origin)
            band = amount_band(amount)
            shortcut = balance > 0 and abs(amount - balance) < 0.01
            cohorts = counts[segment]
            cohorts["all"].add(label, n6, n24, age)
            cohorts["by_amount_band"][band].add(label, n6, n24, age)
            if not shortcut:
                cohorts["without_balance_shortcut"].add(label, n6, n24, age)
                cohorts["without_shortcut_by_amount_band"][band].add(
                    label, n6, n24, age)
        ledger.observe(step, kind, dest)
        seen += 1
    if seen != stop:
        raise RuntimeError("Unexpected scan count")
    result = {
        "status": "POST_HOC_INCOMING_TRANSFER_CASHOUT_LINK_DIAGNOSTIC",
        "evidence_class": "ALREADY_INSPECTED_TRAIN_VALIDATION_ONLY",
        "processed_rows": seen,
        "train_end_row_exclusive": ends[0],
        "validation_end_row_exclusive": stop,
        "windows_step_hours": [6, 24],
        "link_definition": "Prior TRANSFER.nameDest == CASH_OUT.nameOrig",
        "read_before_write": True,
        "no_model_policy_training_or_selection": True,
        "block_enabled": False,
        "cohorts": {name: {
            "all": groups["all"].report(),
            "without_balance_shortcut": groups["without_balance_shortcut"].report(),
            "by_amount_band": {band: groups["by_amount_band"][band].report()
                               for band in BANDS},
            "without_shortcut_by_amount_band": {
                band: groups["without_shortcut_by_amount_band"][band].report()
                for band in BANDS}}
            for name, groups in counts.items()},
        "warning": (
            "Descriptive post-hoc PaySim training/validation inspection, no novel "
            "confirmatory evidence. Simulation identities do not prove funds flow. "
            "Current amount and balance used only for confounding diagnostics. "
            "No real-world predictive, financial or deployment claim; BLOCK off."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return result


def run(csv_path=DEFAULT_CSV, manifest_path=DEFAULT_MANIFEST, output=OUTPUT):
    csv_path, manifest_path, output = map(Path, (csv_path, manifest_path, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError("Existing diagnostic will not be overwritten")
    if not csv_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Frozen PaySim CSV and split manifest required")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("partitions", {}).get("train", {}).get("rows") != 3177938:
        raise RuntimeError("Unexpected PaySim training partition")
    if manifest.get("partitions", {}).get("validation", {}).get("rows") != 1085341:
        raise RuntimeError("Unexpected PaySim validation partition")
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not REQUIRED.issubset(reader.fieldnames or ()):
            raise ValueError("Original PaySim schema missing required columns")
        result = process_rows(reader, manifest, output)
    result["split_manifest_sha256"] = file_sha256(manifest_path)
    # report written in process_rows above, hash is printed separately
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()
    result = run(args.csv, args.manifest, args.output)
    print("Status:", result["status"])
    print("Rows:", result["processed_rows"])
    for name, group in result["cohorts"].items():
        for subset in ("all", "without_balance_shortcut"):
            summary = group[subset]
            print(name, subset,
                  "CASH_OUT rows", summary["rows"], "frauds", summary["frauds"],
                  "linked6", summary["linked_6h"],
                  "linked24", summary["linked_24h"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
