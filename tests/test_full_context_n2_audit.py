import json

import numpy as np
import pytest

from src.full_context import FULL_FEATURE_NAMES
from src.full_context_journal import FEATURE_SCHEMA_SHA256
from src.full_context_materialize import file_sha256
from src.full_context_n2_audit import audit


def test_n2_audit_aggregation_and_no_overwrite(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    split = tmp_path / "split.json"
    out = tmp_path / "n2.json"
    n = 40
    X = np.zeros((n, 21), dtype=np.float32)
    X[:, 5] = 1.0  # TRANSFER
    X[:, 17] = np.log1p(np.arange(n) % 5)
    y = np.asarray(([0, 1, 0, 0] * 10), dtype=np.uint8)
    np.save(src / "features.npy", X)
    np.save(src / "labels.npy", y)
    parts = {name: {"rows": 10, "frauds": int(y[i*10:(i+1)*10].sum())}
             for i, name in enumerate(("train", "validation", "calibration", "policy"))}
    split.write_text(json.dumps({
        "dev_max_step": 372, "final_holdout_start_step": 373,
        "split_end_steps": [10, 20, 30],
        "split_end_rows": [10, 20, 30],
        "development_row_count": 40, "partitions": parts,
    }))
    (src / "materialization.json").write_text(json.dumps({
        "status": "DEVELOPMENT_ONLY_FEATURE_MATERIALIZATION",
        "schema": list(FULL_FEATURE_NAMES),
        "schema_sha256": FEATURE_SCHEMA_SHA256,
        "split_manifest_sha256": file_sha256(split),
        "rows": 40, "partitions": parts,
    }))
    report = audit(src, split, out)
    assert report["train_rows"] == 10
    assert report["validation_rows"] == 10
    groups = report["results"]["validation"]["ALL"]
    assert sum(x["rows"] for x in groups.values()) == 10
    assert sum(x["frauds"] for x in groups.values()) == 3
    assert len(report["results"]["validation_quarters"]) == 4
    assert out.exists()
    with pytest.raises(FileExistsError):
        audit(src, split, out)
