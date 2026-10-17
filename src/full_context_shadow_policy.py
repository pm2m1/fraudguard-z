"""Simulation-only three-action research evaluator. NEVER routes/blocks payments.

Independent evidence and business loss bounds are required before even a
shadow BLOCK policy may be described as safe.
"""
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True)
class ShadowThresholds:
    review: float
    block: float
    max_review_fraction: float = 0.01

    def __post_init__(self):
        if not all(type(v) in (int,float) and isfinite(v) for v in
                   (self.review,self.block,self.max_review_fraction)):
            raise ValueError("Nonfinite or invalid threshold")
        if not 0 < self.review < self.block <= 1:
            raise ValueError("Require strictly ordered review/block cutoffs")
        if not 0 < self.max_review_fraction <= 1:
            raise ValueError("Invalid review budget")


def simulate_decisions(scores, config: ShadowThresholds):
    """Return labels for hypothetical decisions; no database or API effects.

    A BLOCK *shadow label* is always prefixed WOULD_, and is not an actual
    transaction action. A blocked shadow transaction still consumes an
    intervention slot under the combined 1% budget.
    """
    if not isinstance(config, ShadowThresholds):
        raise TypeError("Explicit shadow configuration required")
    result = []
    used = 0
    for ordinal, raw in enumerate(scores, 1):
        if type(raw) not in (float,int) or not isfinite(raw) or not 0 <= raw <= 1:
            raise ValueError("Invalid simulated risk")
        available = int(config.max_review_fraction * ordinal) > used
        if available and raw >= config.block:
            label = "WOULD_BLOCK_SIMULATION_ONLY"
        elif available and raw >= config.review:
            label = "WOULD_REVIEW_SIMULATION_ONLY"
        else:
            label = "ALLOW_SIMULATION_ONLY"
        if label != "ALLOW_SIMULATION_ONLY":
            used += 1
        result.append(label)
    return result
