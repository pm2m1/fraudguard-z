import hashlib

from src.momtsim_lineage_coverage_audit import lineage_audit


def test_taint_propagation_is_label_blind(tmp_path):
    p = tmp_path / 'toy.csv'
    p.write_text(
        'step,transactionType,amount,initiator,oldBalInitiator,recipient,isFraud\n'
        '0,PAYMENT,3,a,30,b,NOT_READ\n'
        '1,TRANSFER,5,a,-5,c,NOT_READ\n'
        '2,TRANSFER,4,c,30,d,NOT_READ\n'
        '3,PAYMENT,2,x,8,y,NOT_READ\n'
    )
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    report = lineage_audit(p, expected_sha256=sha)
    assert report['rows'] == 4
    assert report['direct_invalid_sender_input_rows'] == 1
    assert report['history_affected_after_invalid_event_rows'] == 1
    assert report['no_known_invalid_lineage_rows'] == 2
    assert report['fraud_labels_accessed'] is False
    assert report['frozen_paysim_model_scored'] is False


def test_taint_propagates_to_new_participants(tmp_path):
    p = tmp_path / 'toy.csv'
    p.write_text(
        'step,transactionType,amount,initiator,oldBalInitiator,recipient,isFraud\n'
        '0,TRANSFER,1,a,-1,b,UNREADABLE\n'
        '1,TRANSFER,1,b,1,c,UNREADABLE\n'
        '2,TRANSFER,1,c,1,d,UNREADABLE\n'
    )
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    report = lineage_audit(p, expected_sha256=sha)
    assert report['history_affected_after_invalid_event_rows'] == 2
    assert report['no_known_invalid_lineage_rows'] == 0
