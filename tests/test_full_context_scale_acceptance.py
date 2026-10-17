"""Safety and input-contract tests for Phase 6F (no full replay performed)."""
import csv
import json

import numpy as np
import pytest

from src import full_context_scale_acceptance as scale
from src.full_context import FULL_FEATURE_NAMES
from src.full_context_journal import FEATURE_SCHEMA_SHA256
from src.full_context_materialize import file_sha256


def fixture_files(tmp_path, n=16):
    csv_path = tmp_path / "transactions.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "step", "type", "amount", "oldbalanceOrg",
            "nameOrig", "nameDest", "isFraud",
        ])
        writer.writeheader()
        for i in range(n):
            writer.writerow(dict(step=1, type="TRANSFER", amount=10,
                                 oldbalanceOrg=100, nameOrig="a", nameDest="b",
                                 isFraud=1))
    manifest_path = tmp_path / "split.json"
    partitions = {
        name: {"rows": count, "frauds": 0}
        for name, count in [("train", n), ("validation", 0),
                            ("calibration", 0), ("policy", 0)]
    }
    manifest_path.write_text(json.dumps({
        "dev_max_step": 372, "final_holdout_start_step": 373,
        "development_row_count": n, "partitions": partitions
    }))
    ref = tmp_path / "reference"
    ref.mkdir()
    np.save(ref / "features.npy",
            np.zeros((n, len(FULL_FEATURE_NAMES)), dtype=np.float32))
    (ref / "materialization.json").write_text(json.dumps({
        "status": "DEVELOPMENT_ONLY_FEATURE_MATERIALIZATION",
        "schema": list(FULL_FEATURE_NAMES),
        "schema_sha256": FEATURE_SCHEMA_SHA256,
        "split_manifest_sha256": file_sha256(manifest_path),
        "dtype": "float32", "rows": n, "partitions": partitions,
    }))
    return csv_path, manifest_path, ref


def test_train_prefix_never_emits_fraud_label(tmp_path):
    csv_path, _, _ = fixture_files(tmp_path)
    rows = list(scale.train_source_events(csv_path, 10))
    assert len(rows) == 10
    assert all("isFraud" not in row for row in rows)
    assert [row["source_row_index"] for row in rows] == list(range(10))


def test_refuses_out_of_train_steps(tmp_path):
    csv_path, _, _ = fixture_files(tmp_path)
    text = csv_path.read_text().replace("1,TRANSFER", "239,TRANSFER", 1)
    csv_path.write_text(text)
    with pytest.raises(RuntimeError, match="Train-prefix boundary"):
        list(scale.train_source_events(csv_path, 10))


def test_credentials_bounds_and_overwrite(tmp_path):
    csv_path, manifest_path, ref = fixture_files(tmp_path)
    out = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="password"):
        scale.validate_inputs(10, "", "a" * 32, csv_path, ref, manifest_path, out)
    with pytest.raises(RuntimeError, match="privacy key"):
        scale.validate_inputs(10, "p" * 16, "short", csv_path, ref,
                              manifest_path, out)
    with pytest.raises(ValueError, match="between 10 and 25000"):
        scale.validate_inputs(9, "p" * 16, "k" * 32, csv_path, ref,
                              manifest_path, out)
    out.write_text("immutable")
    with pytest.raises(FileExistsError):
        scale.validate_inputs(10, "p" * 16, "k" * 32, csv_path, ref,
                              manifest_path, out)


def test_reference_provenance_and_readonly(tmp_path):
    csv_path, manifest_path, ref = fixture_files(tmp_path)
    out = tmp_path / "report.json"
    X, meta = scale.validate_inputs(
        10, "p" * 16, "k" * 32, csv_path, ref, manifest_path, out)
    assert X.shape == (16, 21)
    assert not X.flags.writeable
    assert meta["schema_sha256"] == FEATURE_SCHEMA_SHA256
    (ref / "materialization.json").write_text("{}")
    with pytest.raises(RuntimeError, match="provenance"):
        scale.validate_inputs(10, "p" * 16, "k" * 32, csv_path, ref,
                              manifest_path, out)
