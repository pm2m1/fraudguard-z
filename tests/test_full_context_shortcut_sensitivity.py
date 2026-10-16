import json

import numpy as np
import pytest

from src.full_context import FULL_FEATURE_NAMES
from src.full_context_journal import FEATURE_SCHEMA_SHA256
from src.full_context_materialize import file_sha256
from src.full_context_shortcut_sensitivity import FORBIDDEN, GROUPS, run


def test_feature_isolation():
    assert sorted(FORBIDDEN) == [0, 6, 7, 8, 12]
    assert [len(v) for v in GROUPS.values()] == [5, 11, 16]
    assert all(not set(cols) & FORBIDDEN for cols in GROUPS.values())


def test_sensitivity_stays_in_dev_and_never_overwrites(tmp_path):
    source = tmp_path / "dev"
    source.mkdir()
    split = tmp_path / "split.json"
    output = tmp_path / "shortcuts"
    n = 40
    X = np.zeros((n, 21), dtype=np.float32)
    X[:, 1] = 1
    X[:, 9] = np.arange(n)
    X[:, 16] = np.arange(n) % 3
    y = np.tile(np.array([0, 0, 1, 0], dtype=np.uint8), 10)
    np.save(source / "features.npy", X)
    np.save(source / "labels.npy", y)
    partitions = {k: {"rows": 10, "frauds": int(y[a:b].sum())}
                  for k, a, b in (
                      ("train", 0, 10), ("validation", 10, 20),
                      ("calibration", 20, 30), ("policy", 30, 40))}
    split.write_text(json.dumps({
        "dev_max_step": 372,
        "final_holdout_start_step": 373,
        "split_end_steps": [10, 20, 30],
        "split_end_rows": [10, 20, 30],
        "development_row_count": n,
        "partitions": partitions,
    }))
    (source / "materialization.json").write_text(json.dumps({
        "status": "DEVELOPMENT_ONLY_FEATURE_MATERIALIZATION",
        "schema": list(FULL_FEATURE_NAMES),
        "schema_sha256": FEATURE_SCHEMA_SHA256,
        "split_manifest_sha256": file_sha256(split),
        "rows": n, "partitions": partitions,
    }))
    result = run(source, split, output)
    assert len(result["candidates"]) == 6
    assert result["train_rows"] == 10
    assert result["validation_rows"] == 10
    assert (output / "comparison.json").is_file()
    with pytest.raises(FileExistsError):
        run(source, split, output)
