import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.full_context import FULL_FEATURE_NAMES
from src.full_context_calibration import INDICES, run as calibrate
from src.full_context_journal import FEATURE_SCHEMA_SHA256
from src.full_context_materialize import file_sha256
from src.full_context_policy import (
    decisions_with_prefix_cap, frozen_threshold, run, summarize
)


def make_inputs(tmp_path):
    root = tmp_path / "source"
    models = tmp_path / "candidate"
    root.mkdir()
    models.mkdir()
    split = tmp_path / "split.json"
    n = 80
    X = np.zeros((n, 21), dtype=np.float32)
    X[:, 1] = 1.0
    X[:, 17] = np.linspace(0, 4, n)
    y = np.tile(np.array([0, 0, 0, 1], dtype=np.uint8), 20)
    np.save(root / "features.npy", X)
    np.save(root / "labels.npy", y)
    parts = {key: {"rows": 20, "frauds": 5}
             for key in ("train", "validation", "calibration", "policy")}
    manifest = {
        "dev_max_step": 372, "final_holdout_start_step": 373,
        "split_end_steps": [10, 20, 30], "split_end_rows": [20, 40, 60],
        "development_row_count": n, "partitions": parts,
    }
    split.write_text(json.dumps(manifest))
    (root / "materialization.json").write_text(json.dumps({
        "status": "DEVELOPMENT_ONLY_FEATURE_MATERIALIZATION",
        "schema": list(FULL_FEATURE_NAMES),
        "schema_sha256": FEATURE_SCHEMA_SHA256,
        "split_manifest_sha256": file_sha256(split),
        "rows": n, "partitions": parts,
    }))
    model = LogisticRegression(max_iter=100).fit(X[:40, :][:, INDICES], y[:40])
    path = models / "logistic_type_history_network.joblib"
    joblib.dump(model, path)
    (models / "comparison.json").write_text(json.dumps({
        "status": "EXPLORATORY_SHORTCUT_SENSITIVITY_NOT_FOR_PROMOTION",
        "candidates": [{
            "candidate": "logistic_type_history_network",
            "feature_names": [FULL_FEATURE_NAMES[i] for i in INDICES],
            "model_sha256": file_sha256(path),
        }],
    }))
    cal_path = tmp_path / "calibration"
    calibrate(root, split, models, cal_path)
    return root, split, models, cal_path


def test_review_threshold_tie_guard_and_prefix_cap():
    scores = np.array([.5, .5, .5, .8, .1])
    threshold = frozen_threshold(scores, 2)
    assert threshold > .5
    assert np.count_nonzero(scores >= threshold) == 1
    assert frozen_threshold(scores, 0) is None
    with pytest.raises(ValueError):
        frozen_threshold([.1, np.nan], 1)

    scores = np.ones(250, dtype=float)
    selected = decisions_with_prefix_cap(scores, 0.5)
    assert selected.sum() == 2
    assert not selected[:99].any()
    assert selected[99] and selected[199]
    for prefix in (1, 50, 99, 100, 150, 200, 250):
        assert selected[:prefix].sum() <= int(np.floor(0.01 * prefix))


def test_policy_isolated_no_block_and_checksums(tmp_path):
    root, split, models, cal = make_inputs(tmp_path)
    out = tmp_path / "policy"
    report = run(root, split, models, cal, out)
    assert report["policy_start_row"] == 60
    assert report["policy_end_row_exclusive"] == 80
    assert report["review_capacity"] == 0
    assert report["actual_policy_outcomes"]["reviews"] == 0
    assert report["block_enabled"] is False
    assert report["blocks"] == 0
    assert (out / "policy.json").is_file()
    with pytest.raises(FileExistsError):
        run(root, split, models, cal, out)


def test_refuse_calibrator_tamper(tmp_path):
    root, split, models, cal = make_inputs(tmp_path)
    artifact = cal / "calibrator.joblib"
    artifact.write_bytes(artifact.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="checksum"):
        run(root, split, models, cal, tmp_path / "out")


def test_summarize_uses_global_selection():
    stats = summarize(
        np.asarray([0, 1, 0, 1], dtype=np.uint8),
        np.asarray([True, False, False, True], dtype=bool)
    )
    assert stats["reviews"] == 2
    assert stats["true_positives"] == 1
    assert stats["false_positives"] == 1
