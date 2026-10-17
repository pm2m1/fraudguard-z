import hashlib

from src.momtsim_compatibility_audit import audit


def test_label_blind_schema_and_balance_patterns(tmp_path):
    p = tmp_path / 'tiny.csv'
    p.write_text(
        'step,transactionType,amount,initiator,oldBalInitiator,newBalInitiator,recipient,oldBalRecipient,newBalRecipient,isFraud\n'
        '0,TRANSFER,10,s1,50,40,r1,20,30,UNREADABLE_LABEL\n'
        '1,PAYMENT,5,s2,0,0,r2,0,0,UNREADABLE_LABEL\n'
    )
    result = audit(p)
    assert result['rows'] == 2
    assert result['step_zero_present'] is True
    assert result['fraud_label_values_inspected'] is False
    assert result['fraud_detection_evaluated'] is False
    assert result['balance_patterns_by_type']['TRANSFER']['sender_balance_decreased_by_amount'] == 1
    assert result['balance_patterns_by_type']['TRANSFER']['recipient_balance_increased_by_amount'] == 1
    assert result['sha256'] == hashlib.sha256(p.read_bytes()).hexdigest()


def test_audit_fails_on_bad_order_and_missing_identity(tmp_path):
    p = tmp_path / 'tiny.csv'
    p.write_text(
        'step,transactionType,amount,initiator,oldBalInitiator,newBalInitiator,recipient,oldBalRecipient,newBalRecipient,isFraud\n'
        '2,TRANSFER,10,s1,50,40,r1,20,30,0\n'
        '1,PAYMENT,5,,0,0,r2,0,0,1\n'
    )
    result = audit(p)
    assert result['structural_issues_detected']
    assert result['out_of_order_rows'] == 1
    assert result['blank_identity_rows'] == 1
