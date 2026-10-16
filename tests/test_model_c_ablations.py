import csv

import numpy as np
import pytest

from src.model_c_ablations import (
    VARIANTS, validate_variants, validation_rule, rule_report,
)
from src.model_c import MODEL_C_FEATURE_NAMES


def test_ablation_feature_sets():
    validate_variants()
    assert len(VARIANTS["C0_non_linear_original"]) == 8
    assert len(VARIANTS["C1_original_plus_history"]) == 14
    assert len(VARIANTS["C2_type_plus_history_only"]) == 11
    assert len(VARIANTS["C3_full"]) == len(MODEL_C_FEATURE_NAMES) == 17


def test_history_only_excludes_current_amount_and_balance():
    selected = set(VARIANTS["C2_type_plus_history_only"])
    assert selected.isdisjoint({0, 6, 8, 9, 10})


def test_validation_rule_only_reads_validation_rows(tmp_path):
    path = tmp_path / "events.csv"
    fields = ["step", "type", "amount", "oldbalanceOrg"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"step": 1, "type": "TRANSFER", "amount": 10, "oldbalanceOrg": 10},
            {"step": 2, "type": "TRANSFER", "amount": 20, "oldbalanceOrg": 20},
            {"step": 3, "type": "PAYMENT", "amount": 20, "oldbalanceOrg": 20},
            {"step": 4, "type": "CASH_OUT", "amount": 40, "oldbalanceOrg": 40},
        ])
    manifest = {"split_end_steps": [1, 3],
                "partitions": {"validation": {"rows": 2}}}
    assert validation_rule(path, manifest).tolist() == [True, False]


def test_exact_rule_metrics():
    labels = np.asarray([1, 0, 1, 0], dtype=np.uint8)
    flags = np.asarray([True, True, False, False])
    result = rule_report(labels, None, flags)
    assert result["reviewed"] == 2
    assert result["frauds_detected"] == 1
    assert result["false_alerts"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5


def test_rule_rejects_misaligned_rows():
    with pytest.raises(ValueError, match="length mismatch"):
        rule_report(np.array([0, 1]), None, np.array([True]))
