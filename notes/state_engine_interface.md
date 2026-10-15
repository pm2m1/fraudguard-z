# FraudGuard Z — State Engine Interface

Owner: Parth Mishra
Integration: Nirdesh Bhandari / Bibek Chaudhary
Feature schema: fgz_schema_p0_v1

## Transaction input

Required fields:
- event_id: string, unique replay event identifier
- source_row_index: integer, original CSV row index
- step: integer, PaySim simulation hour
- type: string, PaySim transaction type
- amount: nonnegative finite number
- nameOrig: string, originator identifier
- oldbalanceOrg: nonnegative finite number
- nameDest: string, destination identifier

## Core operations

compute_features(transaction):
  Read historical state.
  Return B1, B2, B4, B5, B7 and N3.
  Must not mutate historical state.

observe_committed(transaction):
  Update originator, pair and recipient history.
  Called only after the SQLite transaction commits.

rebuild(committed_events):
  Reset in-memory state.
  Replay committed observations in order.
  Do not re-score historical events.

## Ordering

Process events by (step, source_row_index).
Same-step CSV row order is a replay assumption,
not a real transaction timestamp.

## Feature semantics

Use nameOrig as the originator and nameDest as
the destination for all five PaySim transaction
types, including CASH_IN and CASH_OUT.

The historical 'out' feature names refer to
nameOrig activity, not verified fund direction.

Six-hour window:
  Include prior events with step > current_step - 6.

For sender history:
  B1 = log1p(prior_originator_count)
  B2 = prior_originator_count_6h
  B4 = log((amount+1)/(prior_originator_mean+1))
  B4 = 0 if no previous originator activity

For sender-recipient pairs:
  B5 = log1p(prior_pair_count)
  B7 = 1 if prior_pair_count == 0, else 0

For recipient history:
  N3 = prior_destination_count_6h

## Safety and correctness

- No current-event state update before scoring.
- No future information in feature calculation.
- A duplicate event must not update state twice.
- Serialize scoring, commit and state transitions.
- Rebuild from committed observations only.
- After post-commit state-update failure, stop
  scoring until state reconstruction completes.
