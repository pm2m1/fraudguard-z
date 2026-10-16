"""Faithful additive explanations for the frozen Model B logistic pipeline.

Contributions are to the model's pre-calibration decision margin.
They are not causal explanations, probability contributions, or reasons
that a transaction is known to be fraudulent.
"""
import json
from math import isfinite

import numpy as np

from .features import MODEL_B_FEATURE_NAMES


class ExplanationUnavailable(Exception):
    pass


def explain_record(engine, record):
    """Explain a *committed* decision without changing engine state."""
    try:
        scaler = engine.model.named_steps["standardscaler"]
        classifier = engine.model.named_steps["logisticregression"]
        weights = np.asarray(classifier.coef_, dtype=np.float64)
        intercept = float(classifier.intercept_[0])
        features = np.asarray(json.loads(record["features_json"]), dtype=np.float32)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ExplanationUnavailable("Unsupported model or feature record") from exc

    size = len(MODEL_B_FEATURE_NAMES)
    if (
        features.shape != (size,)
        or weights.shape != (1, size)
        or not np.all(np.isfinite(features))
        or not np.all(np.isfinite(weights))
        or not isfinite(intercept)
        or record["schema_version"] != "fgz_ab_v1"
    ):
        raise ExplanationUnavailable("Model and feature schema mismatch")

    X = features.reshape(1, -1)
    standardized = np.asarray(scaler.transform(X), dtype=np.float64)
    if standardized.shape != (1, size):
        raise ExplanationUnavailable("Unexpected scaler output")

    contributions = standardized[0] * weights[0]
    reconstructed_margin = intercept + float(contributions.sum())
    actual_margin = float(np.asarray(engine.model.decision_function(X)).reshape(-1)[0])

    if not np.isclose(reconstructed_margin, actual_margin, rtol=1e-6, atol=1e-5):
        raise ExplanationUnavailable("Contribution reconstruction failed")

    try:
        calibrated_slope = float(engine.calibrator.coef_[0, 0])
        calculated_risk = float(engine.calibrator.predict_proba(
            np.asarray([[actual_margin]], dtype=np.float64)
        )[0, 1])
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ExplanationUnavailable("Unsupported calibrator") from exc

    if (
        calibrated_slope <= 0
        or not isfinite(calculated_risk)
        or not np.isclose(calculated_risk, record["risk"], rtol=1e-6, atol=1e-7)
    ):
        raise ExplanationUnavailable("Stored risk does not match current model")

    drivers = [
        {
            "feature": feature,
            "feature_value": float(features[i]),
            "margin_contribution": float(contributions[i]),
            "effect": ("increases_score" if contributions[i] > 0
                       else "decreases_score" if contributions[i] < 0
                       else "no_change"),
        }
        for i, feature in enumerate(MODEL_B_FEATURE_NAMES)
    ]
    drivers.sort(key=lambda driver: abs(driver["margin_contribution"]), reverse=True)

    return {
        "event_id": record["event_id"],
        "decision": record["decision"],
        "risk": record["risk"],
        "model": "B",
        "raw_model_margin": actual_margin,
        "baseline_intercept": intercept,
        "reconstructed_model_margin": reconstructed_margin,
        "drivers": drivers,
        "interpretation": (
            "Signed contributions to the pre-calibration logistic margin, "
            "relative to training-set standardized feature means. Positive "
            "contributions raise the model score; negative contributions "
            "lower it. These are not causal factors or probability increments."
        ),
    }
