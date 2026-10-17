import pytest
from src.independent_evaluation_metrics import upper_false_block_rate,score_report

def test_exact_false_block_upper_bound_for_zero_events():
    value=upper_false_block_rate(0,100)
    assert value==pytest.approx(1-0.05**(1/100),rel=1e-12)
    assert value>0
    with pytest.raises(ValueError):
        upper_false_block_rate(0,0)

def test_fixed_decision_metrics_are_consistent_but_not_independent_evidence():
    result=score_report(
        labels=[0,1,0,1],
        scores=[.1,.95,.8,.4],
        decisions=["ALLOW","BLOCK","REVIEW","ALLOW"],
        types=["TRANSFER","TRANSFER","CASH_OUT","CASH_OUT"])
    assert result["rows"]==4
    assert result["frauds"]==2
    assert result["would_block"]==1
    assert result["false_blocks"]==0
    assert result["precision_intervention"]==pytest.approx(.5)
    assert result["recall_intervention"]==pytest.approx(.5)
    assert result["false_positive_rate"]==pytest.approx(.5)
    assert result["by_type"]["CASH_OUT"]["frauds_intervened"]==0
    assert result["not_an_independent_validation_without_provenance"] is True

def test_metrics_reject_unusable_labels_and_scores():
    with pytest.raises(ValueError):
        score_report([0,0],[.1,.2],["ALLOW","REVIEW"],["A","B"])
    with pytest.raises(ValueError):
        score_report([0,1],[.1,float("nan")],["ALLOW","REVIEW"],["A","B"])
