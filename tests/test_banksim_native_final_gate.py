import pytest

from src.banksim_native_final_gate import verify, PINS, EXPECTED_POLICY


def test_pinned_model_and_policy_counts():
    assert len(PINS) == 5
    assert EXPECTED_POLICY["rows"] == 66259
    assert EXPECTED_POLICY["reviewed"] == 607
    assert EXPECTED_POLICY["would_block"] == 0


def test_final_gate_rejects_missing_artifacts(tmp_path):
    with pytest.raises(RuntimeError, match="Frozen file digest mismatch"):
        verify(tmp_path / "prepared", tmp_path / "fit", tmp_path / "diagnostic.json")


def test_final_gate_rejects_unapproved_model(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    fit = tmp_path / "fit"
    fit.mkdir()
    (fit / "model.joblib").write_bytes(b"unapproved")
    with pytest.raises(RuntimeError, match="Frozen file digest mismatch"):
        verify(prepared, fit, tmp_path / "diagnostic.json")
