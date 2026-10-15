# FraudGuard Z — P0 Feature Contract

Schema: fgz_schema_p0_v1

## Replay ordering
Sort by (step, source_row_index).
Compute features using historical state.
Never update state before scoring.

## Feature order
1. log_amount
2. txn_type_CASH_IN
3. txn_type_CASH_OUT
4. txn_type_DEBIT
5. txn_type_PAYMENT
6. txn_type_TRANSFER
7. log_sender_pre_balance
8. log_sender_prior_out_count
9. sender_out_count_6h
10. sender_amount_vs_hist_mean
11. log_pair_prior_count
12. first_time_recipient
13. recipient_in_count_6h

## Definitions
T1 = log1p(amount)
T3 = log1p(oldbalanceOrg)

B1 = log1p(prior_sender_count)
B2 = count(prior sender events with step > current_step - 6)
B4 = log((amount + 1) / (prior_sender_mean_amount + 1))
     If no sender history, B4 = 0
B5 = log1p(prior_pair_count)
B7 = int(prior_pair_count == 0)
N3 = count(prior recipient events with step > current_step - 6)

## Input exclusions
Never include:
- newbalanceOrig
- newbalanceDest
- isFraud
- isFlaggedFraud
- oldbalanceDest

## Transaction processing
1. Validate event and event ID.
2. Check for an existing committed event ID.
3. Compute past-only features.
4. Run model and calibrated risk estimate.
5. Apply ALLOW/REVIEW/BLOCK policy.
6. Commit event, features and decision to SQLite.
7. Update in-memory historical state.
8. Return committed decision.

Serialize stateful scoring and updates.

## Recovery
Rebuild observational state from committed events.
Never re-score previously committed decisions.
