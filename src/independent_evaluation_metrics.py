"""Metrics-only independent-validation building block.

This function does not load data or certify provenance. An independent run
requires a separately frozen licensed dataset, compatible feature replay and
verified model/policy identity. Synthetic fixture tests are NOT validation.
"""
from math import isfinite
import numpy as np
from scipy.stats import beta
from sklearn.metrics import average_precision_score


def upper_false_block_rate(false_blocks, legitimate_count, confidence=.95):
    """Exact one-sided binomial upper confidence bound (Clopper-Pearson)."""
    if (type(false_blocks) is not int or type(legitimate_count) is not int
            or legitimate_count <= 0 or not 0 <= false_blocks <= legitimate_count
            or confidence != .95):
        raise ValueError("Invalid false-block bound inputs")
    if false_blocks == legitimate_count:
        return 1.0
    return float(beta.ppf(confidence, false_blocks+1,
                          legitimate_count-false_blocks))


def score_report(labels, scores, decisions, types):
    """Evaluate a fixed predeclared decision vector, never select thresholds.

    A caller MUST independently establish dataset provenance, label semantics,
    pre-registration, absence of leakage and frozen model/policy parameters.
    """
    y=np.asarray(labels)
    risk=np.asarray(scores,dtype=np.float64)
    d=np.asarray(decisions)
    t=np.asarray(types)
    n=len(y)
    if (not n or y.shape!=(n,) or risk.shape!=(n,) or d.shape!=(n,)
            or t.shape!=(n,) or not np.isin(y,[0,1]).all()
            or not np.isfinite(risk).all() or np.any((risk<0)|(risk>1))
            or not np.isin(d,["ALLOW","REVIEW","BLOCK"]).all()):
        raise ValueError("Invalid fixed-policy evaluation arrays")
    if not np.any(y==0) or not np.any(y==1):
        raise ValueError("Require both fraud and legitimate rows")
    flagged=d!="ALLOW"
    positive=y==1
    negative=~positive
    reviewed=d=="REVIEW"
    blocked=d=="BLOCK"
    tp=int(np.count_nonzero(flagged & positive))
    fp=int(np.count_nonzero(flagged & negative))
    fblocks=int(np.count_nonzero(blocked & negative))
    return {
        "rows":n,
        "frauds":int(positive.sum()),
        "reviewed":int(reviewed.sum()),
        "would_block":int(blocked.sum()),
        "false_blocks":fblocks,
        "precision_intervention":tp/(tp+fp) if tp+fp else None,
        "recall_intervention":tp/int(positive.sum()),
        "false_positive_rate":fp/int(negative.sum()),
        "pr_auc":float(average_precision_score(y,risk)),
        "false_block_rate_upper_95":upper_false_block_rate(fblocks,int(negative.sum())),
        "by_type": {
            str(kind):{
                "rows":int((t==kind).sum()),
                "frauds":int(np.count_nonzero((t==kind)&positive)),
                "interventions":int(np.count_nonzero((t==kind)&flagged)),
                "frauds_intervened":int(np.count_nonzero((t==kind)&flagged&positive)),
                "legitimate_interventions":int(np.count_nonzero((t==kind)&flagged&negative))
            } for kind in np.unique(t)
        },
        "not_an_independent_validation_without_provenance":True
    }
