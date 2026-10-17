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


# Local policy-diagnostic gates. Use disposable arrays and report files only.
from src.banksim_native_policy_diagnostic import diagnostic, prefix_reviews
import numpy as np


def test_policy_review_prefix_capacity_under_all_high_scores():
    decisions = prefix_reviews(np.ones(250), cutoff=0.5)
    assert int(np.count_nonzero(decisions == "REVIEW")) == 2
    assert decisions[0] == "ALLOW"
    assert decisions[98] == "ALLOW"
    assert decisions[99] == "REVIEW"
    assert decisions[199] == "REVIEW"
    assert set(decisions) == {"ALLOW", "REVIEW"}


def test_policy_review_threshold_with_all_low_scores():
    decisions = prefix_reviews(np.full(300, 0.3), cutoff=0.5)
    assert np.all(decisions == "ALLOW")


@pytest.mark.parametrize("bad", [
    np.asarray([float("nan")]),
    np.asarray([float("inf")]),
    np.ones((2, 2)),
])
def test_policy_review_invalid_scores_fail_closed(bad):
    with pytest.raises(ValueError, match="Invalid score"):
        prefix_reviews(bad)


def test_policy_diagnostic_refuses_report_overwrite(tmp_path):
    report = tmp_path / "existing.json"
    report.write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="overwrite"):
        diagnostic(tmp_path / "not-prepared", tmp_path / "not-fit", report)
    assert report.read_text() == "do not overwrite"
