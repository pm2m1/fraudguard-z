import json
import math
from pathlib import Path

import joblib
import pytest

from src.decision_engine import DecisionEngine
from src.features import MODEL_B_FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]


def transaction(event_id, step, index, recipient="C100"):
    return {
        "event_id": event_id,
        "source_row_index": index,
        "step": step,
        "type": "TRANSFER",
        "amount": 181.0,
        "nameOrig": f"C{index + 1}",
        "nameDest": recipient,
        "oldbalanceOrg": 181.0,
    }


def test_real_model_with_sqlite(tmp_path):
    model_path = ROOT / "models/model_b.joblib"
    calibrator_path = ROOT / "models/calibrator_b.joblib"

    if not model_path.exists() or not calibrator_path.exists():
        pytest.skip("Local model artifacts are unavailable")

    model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path)

    with (ROOT / "reports/split_manifest.json").open() as file:
        manifest = json.load(file)

    with (ROOT / "reports/review_policy.json").open() as file:
        policy = json.load(file)

    assert manifest["schema_b"] == list(MODEL_B_FEATURE_NAMES)
    assert model.n_features_in_ == 8
    assert calibrator.n_features_in_ == 1
    assert policy["selected_model"] == "B"
    assert policy["block_enabled"] is False

    path = tmp_path / "live.sqlite3"
    threshold = policy["review_threshold"]

    engine = DecisionEngine(
        path, model, calibrator, threshold
    )

    try:
        first = transaction("E1", 1, 0)
        second = transaction("E2", 2, 1)

        result1 = engine.score(first)
        result2 = engine.score(second)

        assert result1["n3"] == 0
        assert result2["n3"] == 1

        for result in (result1, result2):
            assert 0 <= result["risk"] <= 1
            assert result["decision"] == (
                "REVIEW"
                if result["risk"] >= threshold
                else "ALLOW"
            )

        duplicate = engine.score(first)

        assert duplicate["duplicate"] is True
        assert duplicate["risk"] == result1["risk"]
        assert duplicate["decision"] == result1["decision"]
        assert engine.status()["committed_events"] == 2

        saved = engine.db.execute(
            "SELECT features_json FROM decisions "
            "WHERE event_id = ?",
            ("E2",),
        ).fetchone()

        features = json.loads(saved["features_json"])

        assert len(features) == 8
        assert features[-1] == pytest.approx(math.log1p(1))

    finally:
        engine.close()

    restarted = DecisionEngine(
        path, model, calibrator, threshold
    )

    try:
        third = restarted.score(transaction("E3", 3, 2))

        assert third["n3"] == 2
        assert restarted.status()["committed_events"] == 3

    finally:
        restarted.close()