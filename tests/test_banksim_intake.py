import csv
import hashlib

import pytest

from src.banksim_intake import REQUIRED, inspect_dataset, sha256


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(REQUIRED))
        w.writeheader()
        w.writerows(rows)


def sample(step=1):
    return {
        "step": step, "customer": "C", "merchant": "M",
        "category": "es_food", "amount": "12.30", "fraud": "not_read",
    }


def test_sha256_bytes(tmp_path):
    file = tmp_path / "test.bin"
    file.write_bytes(b"hello")
    assert sha256(file) == hashlib.sha256(b"hello").hexdigest()


def test_intake_rejects_missing_columns(tmp_path):
    file = tmp_path / "bad.csv"
    file.write_text("step,amount\n1,12\n")
    with pytest.raises(ValueError, match="columns"):
        inspect_dataset(file)


def test_intake_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_dataset(tmp_path / "no.csv")


def test_intake_rejects_wrong_size_without_reading_fraud_values(tmp_path):
    file = tmp_path / "tiny.csv"
    write_csv(file, [sample(1), sample(2)])
    with pytest.raises(ValueError, match="transaction count"):
        inspect_dataset(file)


def test_intake_rejects_bad_amount_before_size_check(tmp_path):
    file = tmp_path / "amount.csv"
    row = sample()
    row["amount"] = "NaN"
    write_csv(file, [row])
    with pytest.raises(ValueError, match="amount"):
        inspect_dataset(file)
