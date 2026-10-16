import json

import joblib
import numpy as np
import pytest

from src.explanations import ExplanationUnavailable, explain_record


class Scaler:
    def transform(self, X):
        return (X - 2.0) / 2.0


class Logistic:
    coef_ = np.array([[1.0, -0.5, 0.0, 2.0, -1.0, 0.3, 0.2, 1.5]])
    intercept_ = np.array([0.4])


class Model:
    named_steps = {"standardscaler": Scaler(), "logisticregression": Logistic()}

    def decision_function(self, X):
        return Scaler().transform(X) @ Logistic.coef_[0] + Logistic.intercept_[0]


class Calibrator:
    coef_ = np.array([[1.0]])

    def predict_proba(self, margins):
        p = 1 / (1 + np.exp(-np.asarray(margins)[:, 0]))
        return np.column_stack((1-p, p))


class Engine:
    model = Model()
    calibrator = Calibrator()


def saved_record():
    X = np.arange(1, 9, dtype=np.float32)
    margin = float(Engine.model.decision_function(X.reshape(1, -1))[0])
    p = float(Engine.calibrator.predict_proba([[margin]])[0, 1])
    return {
        "event_id": "E1", "decision": "REVIEW", "risk": p,
        "features_json": json.dumps(X.tolist()), "schema_version": "fgz_ab_v1"
    }


def test_explanation_adds_to_real_margin():
    result = explain_record(Engine(), saved_record())
    assert len(result["drivers"]) == 8
    total = result["baseline_intercept"] + sum(
        driver["margin_contribution"] for driver in result["drivers"]
    )
    assert total == pytest.approx(result["raw_model_margin"])
    assert result["risk"] == saved_record()["risk"]


def test_explanation_does_not_modify_saved_record():
    row = saved_record()
    before = dict(row)
    explain_record(Engine(), row)
    assert row == before


def test_rejects_different_stored_risk():
    row = saved_record()
    row["risk"] = 0
    with pytest.raises(ExplanationUnavailable):
        explain_record(Engine(), row)


def test_rejects_incompatible_schema():
    row = saved_record()
    row["schema_version"] = "future_version"
    with pytest.raises(ExplanationUnavailable):
        explain_record(Engine(), row)


def test_rejects_wrong_feature_shape():
    row = saved_record()
    row["features_json"] = json.dumps([1, 2])
    with pytest.raises(ExplanationUnavailable):
        explain_record(Engine(), row)


def test_real_model_if_available():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    model_file = root / "models/model_b.joblib"
    calibrator_file = root / "models/calibrator_b.joblib"
    if not model_file.exists() or not calibrator_file.exists():
        pytest.skip("Local model binaries not available")
    model = joblib.load(model_file)
    calibrator = joblib.load(calibrator_file)
    X = np.array([8.5, 0, 0, 0, 0, 1, 8.5, 0], dtype=np.float32)
    margin = float(model.decision_function(X.reshape(1, -1))[0])
    risk = float(calibrator.predict_proba([[margin]])[0, 1])
    row = {
        "event_id": "real", "decision": "REVIEW" if risk > 0.011417806148529053 else "ALLOW",
        "risk": risk, "features_json": json.dumps(X.tolist()),
        "schema_version": "fgz_ab_v1",
    }
    result = explain_record(type("E", (), {"model": model, "calibrator": calibrator})(), row)
    assert result["raw_model_margin"] == pytest.approx(margin, abs=1e-5)
