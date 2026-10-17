"""Phase 6I end-to-end synthetic local acceptance test."""
import json

import pytest

from src.full_context_failure_acceptance import run


def test_phase6i_temporary_fault_matrix_no_overwrite(tmp_path):
    output = tmp_path / "acceptance.json"
    result = run(output)
    assert result["status"] == "PASS_LOCAL_IN_PROCESS_FAILURE_ACCEPTANCE"
    assert result["checks_passed"] == 8
    assert result["blocked"] == 0
    assert result["production_validated"] is False
    assert all(item["passed"] for item in result["checks"])
    assert json.loads(output.read_text())["checks_passed"] == 8
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        run(output)
