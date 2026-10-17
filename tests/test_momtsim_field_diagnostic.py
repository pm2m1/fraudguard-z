from src.momtsim_field_diagnostic import field_diagnostic, numeric_status


def test_numeric_status():
    assert numeric_status('-1')[0] == 'negative'
    assert numeric_status('nan')[0] == 'nonfinite'
    assert numeric_status('')[0] == 'missing'
    assert numeric_status('1.25')[0] == 'ok'


def test_only_post_balance_invalid_not_invalid_model_input(tmp_path):
    path = tmp_path / 'synthetic.csv'
    path.write_text('step,transactionType,amount,initiator,oldBalInitiator,newBalInitiator,recipient,oldBalRecipient,newBalRecipient,isFraud\n0,TRANSFER,10,c1,50,-1,m1,20,30,LABEL_NOT_READ\n')
    result = field_diagnostic(path)
    assert result['rows'] == 1
    assert result['invalid_sender_model_input_rows'] == 0
    assert result['invalid_only_posttransaction_balance_rows'] == 1
    assert result['fraud_labels_accessed'] is False


def test_invalid_old_sender_balance_counts_as_invalid_input(tmp_path):
    path = tmp_path / 'synthetic.csv'
    path.write_text('step,transactionType,amount,initiator,oldBalInitiator,newBalInitiator,recipient,oldBalRecipient,newBalRecipient,isFraud\n0,TRANSFER,10,c1,-1,0,m1,20,30,LABEL_NOT_READ\n')
    result = field_diagnostic(path)
    assert result['invalid_sender_model_input_rows'] == 1
    assert result['invalid_any_pretransaction_balance_rows'] == 1
    assert result['fraud_labels_accessed'] is False
