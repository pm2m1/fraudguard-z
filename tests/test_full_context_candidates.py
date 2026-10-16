import json

import numpy as np
import pytest

from src.full_context import FULL_FEATURE_NAMES
from src.full_context_candidates import (
    fit_candidates, top_review_metrics, validate_inputs,
)
from src.full_context_journal import FEATURE_SCHEMA_SHA256
from src.full_context_materialize import file_sha256


def make_inputs(tmp_path):
    source = tmp_path / "materialized"
    source.mkdir()
    split = tmp_path / "splits.json"
    n = 16
    X = np.zeros((n, 21), dtype=np.float32)
    X[:, 0] = np.linspace(0, 4, n)
    X[:, 1] = 1
    X[:, 7] = np.linspace(0, 1, n)
    y = np.tile(np.array([0, 1, 0, 0], dtype=np.uint8), 4)
    np.save(source / "features.npy", X)
    np.save(source / "labels.npy", y)
    manifest = {
        "dev_max_step": 372, "final_holdout_start_step": 373,
        "split_end_steps": [10, 20, 30],
        "split_end_rows": [4, 8, 12],
        "development_row_count": n,
        "partitions": {
            name: {"rows": 4, "frauds": 1}
            for name in ("train", "validation", "calibration", "policy")
        },
    }
    split.write_text(json.dumps(manifest))
    metadata = {
        "status": "DEVELOPMENT_ONLY_FEATURE_MATERIALIZATION",
        "schema": list(FULL_FEATURE_NAMES),
        "schema_sha256": FEATURE_SCHEMA_SHA256,
        "split_manifest_sha256": file_sha256(split),
        "rows": n, "partitions": manifest["partitions"],
    }
    (source / "materialization.json").write_text(json.dumps(metadata))
    return source, split


def test_fixed_feature_groups_train_only_and_smoke_status(tmp_path):
    source, split = make_inputs(tmp_path)
    output = tmp_path / "candidates"
    report = fit_candidates(source, split, output, max_candidates=1)
    assert report["status"] == "PARTIAL_ENGINEERING_SMOKE_NOT_FOR_SELECTION"
    assert report["training_rows"] == 4
    assert report["validation_rows"] == 4
    assert len(report["candidates"]) == 1
    candidate = report["candidates"][0]
    assert candidate["candidate"] == "logistic_transaction"
    assert len(candidate["columns"]) == 9
    assert (output / "logistic_transaction.joblib").is_file()
    assert (output / "comparison.json").is_file()
    assert not (output / "hgb_full_context.joblib").exists()


def test_reject_existing_output(tmp_path):
    source, split = make_inputs(tmp_path)
    output = tmp_path / "already"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        fit_candidates(source, split, output, max_candidates=1)


def test_reject_wrong_manifest_hash(tmp_path):
    source, split = make_inputs(tmp_path)
    document = json.loads(split.read_text())
    document["note"] = "modified"
    split.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="split mismatch"):
        validate_inputs(source, split)


def test_reject_wrong_feature_width(tmp_path):
    source, split = make_inputs(tmp_path)
    np.save(source / "features.npy", np.zeros((16, 8), dtype=np.float32))
    with pytest.raises(RuntimeError, match="shape/dtype"):
        validate_inputs(source, split)


def test_top_one_percent_diagnostic():
    labels = np.array([0, 1, 0, 1], dtype=np.uint8)
    scores = np.array([0.1, 0.8, 0.3, 0.2])
    metrics = top_review_metrics(labels, scores)
    assert metrics["reviews"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["recall"] == pytest.approx(0.5)
