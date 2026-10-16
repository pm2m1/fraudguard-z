import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.full_context import FULL_FEATURE_NAMES
from src.full_context_calibration import INDICES, fit_sigmoid, run
from src.full_context_journal import FEATURE_SCHEMA_SHA256
from src.full_context_materialize import file_sha256


def fixture(tmp_path):
    source = tmp_path / "source"
    candidates = tmp_path / "candidates"
    source.mkdir()
    candidates.mkdir()
    split = tmp_path / "split.json"
    n = 80
    X = np.zeros((n, 21), dtype=np.float32)
    X[:, 1] = 1.0
    X[:, 17] = np.linspace(0, 4, n)
    y = np.array(([0, 0, 0, 1] * 20), dtype=np.uint8)
    np.save(source / "features.npy", X)
    np.save(source / "labels.npy", y)
    parts = {name: {"rows": 20, "frauds": 5}
             for name in ("train", "validation", "calibration", "policy")}
    manifest = {"dev_max_step": 372, "final_holdout_start_step": 373,
                "split_end_steps": [10, 20, 30],
                "split_end_rows": [20, 40, 60],
                "development_row_count": n, "partitions": parts}
    split.write_text(json.dumps(manifest))
    (source / "materialization.json").write_text(json.dumps({
        "status": "DEVELOPMENT_ONLY_FEATURE_MATERIALIZATION",
        "schema": list(FULL_FEATURE_NAMES), "schema_sha256": FEATURE_SCHEMA_SHA256,
        "split_manifest_sha256": file_sha256(split),
        "rows": n, "partitions": parts,
    }))
    # Artificial, trusted locally generated fitted model with monotonic signal.
    model = LogisticRegression(max_iter=100).fit(
        X[:40, INDICES], np.array([0, 0, 0, 1]*10, dtype=np.uint8)
    )
    path = candidates / "logistic_type_history_network.joblib"
    joblib.dump(model, path)
    (candidates / "comparison.json").write_text(json.dumps({
        "status": "EXPLORATORY_SHORTCUT_SENSITIVITY_NOT_FOR_PROMOTION",
        "candidates": [{
            "candidate": "logistic_type_history_network",
            "feature_names": [FULL_FEATURE_NAMES[i] for i in INDICES],
            "model_sha256": file_sha256(path),
        }],
    }))
    return source, split, candidates


def test_fit_sigmoid_guards():
    margins = np.array([-2., -1., 1., 2., -3., 3.])
    labels = np.array([0, 0, 1, 1, 0, 1])
    assert fit_sigmoid(margins, labels).coef_[0, 0] > 0
    with pytest.raises(ValueError):
        fit_sigmoid([1., 2., 3., 4.], [1, 1, 1, 1])
    with pytest.raises(ValueError):
        fit_sigmoid([1., np.nan, 3., 4.], [0, 1, 0, 1])


def test_calibration_partition_and_no_overwrite(tmp_path):
    source, split, candidates = fixture(tmp_path)
    dest = tmp_path / "out"
    result = run(source, split, candidates, dest)
    assert result["calibration_rows"] == 20
    assert result["calibration_frauds"] == 5
    assert result["calibration_row_start"] == 40
    assert result["calibration_row_end_exclusive"] == 60
    assert result["block_enabled"] is False
    assert (dest / "calibrator.joblib").exists()
    with pytest.raises(FileExistsError):
        run(source, split, candidates, dest)


def test_refuse_altered_model_before_deserialization(tmp_path):
    source, split, candidates = fixture(tmp_path)
    path = candidates / "logistic_type_history_network.joblib"
    path.write_bytes(path.read_bytes() + b"X")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        run(source, split, candidates, tmp_path / "out")
