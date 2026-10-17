import json

import pytest

from src.banksim_native_fit import split_boundaries, fit
from src.banksim_native_prepare import FEATURE_NAMES, PARTITIONS, SCHEMA_SHA256, SOURCE_SHA256


def good_meta():
    counts = dict(zip(PARTITIONS, (332592, 63740, 65080, 66259, 66972)))
    return {
        "feature_schema_sha256": SCHEMA_SHA256,
        "feature_names": list(FEATURE_NAMES),
        "source_sha256": SOURCE_SHA256,
        "rows": 594643,
        "feature_shape": [594643, 24],
        "partition_rows": counts,
        "development_labels_rows": 527671,
        "final_labels_accessed": False,
    }


def test_exact_chronological_boundaries_and_isolation():
    bounds = split_boundaries(good_meta())
    assert bounds == [0, 332592, 396332, 461412, 527671, 594643]


@pytest.mark.parametrize("mutation", [
    {"feature_schema_sha256": "wrong"},
    {"source_sha256": "wrong"},
    {"final_labels_accessed": True},
    {"development_labels_rows": 594643},
    {"feature_shape": [594643, 23]},
])
def test_reject_unapproved_materialization(mutation):
    meta = good_meta()
    meta.update(mutation)
    with pytest.raises(RuntimeError):
        split_boundaries(meta)


def test_refuses_overwrite_before_accessing_prepared_data(tmp_path):
    output = tmp_path / "fit"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        fit(tmp_path / "does_not_exist", output)
