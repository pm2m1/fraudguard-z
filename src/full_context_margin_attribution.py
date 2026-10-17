"""Faithful attribution to a frozen logistic model's *margin*, not fraud cause."""
import numpy as np
from .full_context import FULL_FEATURE_NAMES
from .full_context_calibration import INDICES

def explain_margin(model, full_features):
    x = np.asarray(full_features, dtype=np.float32)
    if x.shape != (21,) or not np.isfinite(x).all():
        raise ValueError("Expected one finite 21-feature vector")
    if not hasattr(model, "named_steps"):
        raise ValueError("Only explicit logistic pipeline supported")
    try:
        scaler = model.named_steps["standardscaler"]
        classifier = model.named_steps["logisticregression"]
    except (KeyError, TypeError):
        raise ValueError("Unsupported pipeline") from None
    coefs = np.asarray(classifier.coef_, dtype=np.float64)
    if coefs.shape != (1, len(INDICES)) or getattr(classifier,"classes_",None) is None:
        raise ValueError("Invalid logistic coefficients")
    if list(classifier.classes_) != [0,1]:
        raise ValueError("Unexpected class order")
    transformed = np.asarray(scaler.transform(x[list(INDICES)].reshape(1,-1)),dtype=np.float64)
    if transformed.shape != (1,len(INDICES)) or not np.isfinite(transformed).all():
        raise ValueError("Invalid transformed features")
    contribution = transformed[0]*coefs[0]
    intercept = float(classifier.intercept_[0])
    margin = float(intercept+contribution.sum())
    model_margin = float(np.asarray(model.decision_function(x[list(INDICES)].reshape(1,-1))).item())
    # The deployed pipeline receives float32 features, and StandardScaler
    # preserves float32 transforms. A manual contribution reduction and
    # sklearn's BLAS-backed decision_function can differ slightly in rounding.
    # Bound reconstruction error by a few float32 ULPs, scaled to the absolute
    # terms to handle cancellation (rather than demanding bitwise identity).
    scale = 1.0 + abs(intercept) + float(np.abs(contribution).sum())
    tolerance = 8.0 * float(np.finfo(np.float32).eps) * scale
    if (not np.isfinite(model_margin) or
            abs(margin - model_margin) > tolerance):
        raise RuntimeError("Feature attribution does not reconstruct model margin")
    return {
        "kind":"logistic_precalibration_margin_contributions",
        "not_probability_or_causal_attribution":True,
        "intercept":intercept,
        "margin":model_margin,
        "items":[{"feature":FULL_FEATURE_NAMES[index],"contribution":float(value)}
                 for index,value in zip(INDICES,contribution)]
    }
