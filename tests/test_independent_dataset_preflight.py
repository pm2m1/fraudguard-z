import pytest

from src.independent_dataset_preflight import (
    REQUIRED, inspect_csv, inspect_headers,
)


def test_complete_header_only_admits_schema_review_not_validation():
    report = inspect_headers(list(REQUIRED) + ["other_field"])
    assert report["schema_fields_present"] is True
    assert report["status"] == "SCHEMA_ONLY_CANDIDATE_REQUIRES_INDEPENDENT_REVIEW"
    assert report["independent_fraud_validation"] is False
    assert report["missing"] == []


@pytest.mark.parametrize("external_header", [
    ("step", "customer", "merchant", "category", "amount", "fraud"),
    ("tran_id", "orig_acct", "bene_acct", "tx_type",
     "base_amt", "tran_timestamp", "is_sar"),
    ("TransactionID", "TransactionDT", "TransactionAmt",
     "ProductCD", "isFraud"),
])
def test_incompatible_public_dataset_headers_rejected(external_header):
    report = inspect_headers(external_header)
    assert report["status"] == "UNSUITABLE_FOR_UNMODIFIED_FROZEN_MODEL"
    assert report["schema_fields_present"] is False
    assert report["independent_fraud_validation"] is False
    assert report["missing"]


def test_duplicate_headers_rejected_even_if_all_required_are_present():
    report = inspect_headers(list(REQUIRED) + ["nameOrig"])
    assert report["status"] == "UNSUITABLE_FOR_UNMODIFIED_FROZEN_MODEL"
    assert report["duplicate_headers"] == ["nameOrig"]


def test_csv_header_only_and_empty_csv_refusal(tmp_path):
    path = tmp_path / "candidate.csv"
    path.write_text(",".join(REQUIRED) + "\n" +
                    "bad intentionally unparsed row\n", encoding="utf-8")
    result = inspect_csv(path)
    assert result["schema_fields_present"] is True
    assert result["independent_fraud_validation"] is False

    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty CSV"):
        inspect_csv(path)
