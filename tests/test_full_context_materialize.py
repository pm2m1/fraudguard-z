import csv
import json

import numpy as np
import pytest

from src.full_context_materialize import materialize, validate_manifest


def fixture(tmp_path, *, corrupt=False):
    csv_path = tmp_path / "sample.csv"
    manifest_path = tmp_path / "split.json"
    rows = [
        (1, 0), (10, 1), (20, 0), (30, 1),
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "step", "type", "amount", "nameOrig", "nameDest",
            "oldbalanceOrg", "isFraud",
        ])
        writer.writeheader()
        for i, (step, label) in enumerate(rows):
            writer.writerow({
                "step": step, "type": "TRANSFER", "amount": i+1,
                "nameOrig": "A", "nameDest": "B", "oldbalanceOrg": 100,
                "isFraud": label,
            })
        # Invalid test label ensures we stop before reading future labels.
        writer.writerow({
            "step": 373, "type": "TRANSFER", "amount": 500,
            "nameOrig": "TEST", "nameDest": "TEST",
            "oldbalanceOrg": 100, "isFraud": "NOT_A_LABEL",
        })
    manifest = {
        "dev_max_step": 372,
        "final_holdout_start_step": 373,
        "split_end_steps": [1, 10, 20],
        "split_end_rows": [1, 2, 3],
        "development_row_count": 4,
        "partitions": {
            "train": {"rows": 1, "frauds": 0},
            "validation": {"rows": 1, "frauds": 1},
            "calibration": {"rows": 1, "frauds": 0},
            "policy": {"rows": 1, "frauds": 0 if corrupt else 1},
        }
    }
    manifest_path.write_text(json.dumps(manifest))
    return csv_path, manifest_path


def test_materialize_frozen_dev_only(tmp_path):
    csv_path, manifest_path = fixture(tmp_path)
    target = tmp_path / "output"
    result = materialize(csv_path, manifest_path, target)
    assert result["rows"] == 4
    assert result["last_step"] == 30
    assert result["partitions"]["policy"]["frauds"] == 1
    X = np.load(target / "features.npy", mmap_mode="r")
    y = np.load(target / "labels.npy", mmap_mode="r")
    assert X.shape == (4, 21)
    assert y.tolist() == [0, 1, 0, 1]
    assert all(np.isfinite(X.ravel()))
    assert (target / "materialization.json").is_file()


def test_never_overwrite_existing_results(tmp_path):
    csv_path, manifest_path = fixture(tmp_path)
    target = tmp_path / "output"
    materialize(csv_path, manifest_path, target)
    with pytest.raises(FileExistsError):
        materialize(csv_path, manifest_path, target)


def test_fraud_count_mismatch_never_publishes(tmp_path):
    csv_path, manifest_path = fixture(tmp_path, corrupt=True)
    target = tmp_path / "output"
    with pytest.raises(ValueError, match="Fraud count mismatch"):
        materialize(csv_path, manifest_path, target)
    assert not target.exists()


def test_refuse_inconsistent_split_manifest(tmp_path):
    csv_path, manifest_path = fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["final_holdout_start_step"] = 372
    with pytest.raises(ValueError, match="holdout"):
        validate_manifest(manifest)
    manifest["final_holdout_start_step"] = 373
    manifest["split_end_rows"] = [1, 1, 3]
    with pytest.raises(ValueError, match="manifest"):
        validate_manifest(manifest)
