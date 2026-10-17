import pytest

from src.banksim_structural_audit import audit


def test_structural_audit_never_reads_fraud_label_values(tmp_path):
    path = tmp_path / "bank.csv"
    path.write_text(
        "step,customer,age,gender,zipcodeOri,merchant,zipMerchant,category,amount,fraud\n"
        "0,c1,21,M,1,m1,2,food,3.2,DO_NOT_PARSE\n"
        "1,c1,21,M,1,m2,2,travel,2.1,DO_NOT_PARSE\n"
    )
    report = audit(path)
    assert report["status"] == "STRUCTURAL_AUDIT_PASSED_NOT_VALIDATION"
    assert report["rows"] == 2
    assert report["distinct_steps"] == 2
    assert report["category_counts"] == {"food": 1, "travel": 1}
    assert report["fraud_label_values_inspected"] is False
    assert report["independent_validation_of_frozen_paysim_model"] is False


def test_structural_audit_flags_reordering_without_fraud_inspection(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "step,customer,age,gender,zipcodeOri,merchant,zipMerchant,category,amount,fraud\n"
        "2,c1,21,M,1,m1,2,food,3.2,1\n"
        "1,c1,21,M,1,m2,2,food,2.0,0\n"
    )
    report = audit(path)
    assert report["nonmonotonic_step_rows"] == 1
    assert report["status"] == "STRUCTURAL_AUDIT_REQUIRES_REVIEW"


def test_structural_audit_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("step,customer,amount,fraud\n1,c1,3.2,0\n")
    with pytest.raises(ValueError, match="Missing BankSim fields"):
        audit(path)
