import hashlib

import numpy as np
import pytest

from src.banksim_native_final_test import read_final_labels, evaluate


def test_explicit_label_reader_uses_only_final_period(tmp_path):
    source = tmp_path / "synthetic.csv"
    source.write_text(
        "step,customer,age,gender,zipcodeOri,merchant,zipMerchant,category,amount,fraud\n"
        "0,c,20,F,z,m,z,'es_food',1,NOT_A_LABEL\n"
        "162,c,20,F,z,m,z,'es_food',2,1\n"
        "179,c,20,F,z,m,z,'es_food',3,0\n"
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    labels = read_final_labels(
        source, expected_sha256=digest, expected_before=1, expected_final=2
    )
    assert labels.tolist() == [1, 0]


def test_label_reader_rejects_source_hash_mismatch(tmp_path):
    source = tmp_path / "synthetic.csv"
    source.write_text("unexpected contents")
    with pytest.raises(RuntimeError, match="checksum"):
        read_final_labels(source, expected_sha256="0" * 64)


def test_final_evaluation_never_overwrites_report(tmp_path):
    report = tmp_path / "result.json"
    report.write_text("old results")
    with pytest.raises(FileExistsError, match="never overwrite"):
        evaluate(report=report)
    assert report.read_text() == "old results"


def test_final_evaluation_never_retries_after_attempt(tmp_path):
    report = tmp_path / "result.json"
    marker = report.with_name(report.name + ".attempt.json")
    marker.write_text("previous attempt")
    with pytest.raises(FileExistsError, match="never overwrite or retry"):
        evaluate(report=report)
    assert not report.exists()
