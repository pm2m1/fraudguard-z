# FraudGuard Z feature contract

Schema: fgz_ab_v1
Model: B, recipient-history logistic regression

## Input features, in order

0. log_amount = log1p(amount)
1. txn_type_CASH_IN
2. txn_type_CASH_OUT
3. txn_type_DEBIT
4. txn_type_PAYMENT
5. txn_type_TRANSFER
6. log_sender_pre_balance = log1p(oldbalanceOrg)
7. log_recipient_in_count_6h = log1p(N3)

N3 counts earlier events with the same nameDest and
step > current_step - 6.

Transactions with the same step are processed in their
original input order.

All transaction types contribute to recipient history.

## Model inputs

Model A uses columns 0 through 6.
Model B uses columns 0 through 7.

Never use fraud labels, post-transaction balances or
oldbalanceDest as model inputs.

## Decision policy

Model B is the selected model.

Scores are calibrated using a separate chronological
calibration period.

ALLOW: risk < frozen review threshold.
REVIEW: risk >= frozen review threshold.

Automatic BLOCK decisions are disabled.

The threshold is recorded in reports/review_policy.json.
Do not adjust it based on final holdout results.

## Processing requirements

- Validate the incoming transaction.
- Reject out-of-order new events.
- Return the stored decision for duplicate event IDs.
- Calculate features from previously observed events.
- Commit the decision to SQLite before updating state.
- Recover state from committed events after a restart.
- Stop scoring if committed state cannot be reconstructed.
