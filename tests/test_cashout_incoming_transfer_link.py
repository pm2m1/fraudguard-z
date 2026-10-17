"""Focused tests; no PaySim dataset required."""
import json
from io import StringIO

import pytest

from src.cashout_incoming_transfer_link import (
    BANDS, Counts, IncomingTransferLedger, amount_band, run,
)


def test_read_before_write_and_correct_identity_direction():
    ledger = IncomingTransferLedger()
    assert ledger.before(1, "B") == (0, 0, None)
    ledger.observe(1, "TRANSFER", "B")
    assert ledger.before(1, "B") == (1, 1, 0)
    assert ledger.before(1, "A") == (0, 0, None)
    ledger.observe(1, "CASH_OUT", "B")
    assert ledger.before(1, "B") == (1, 1, 0)


def test_expiry_boundaries_and_repeated_recipient_steps():
    ledger = IncomingTransferLedger()
    ledger.observe(1, "TRANSFER", "B")
    ledger.observe(2, "TRANSFER", "B")
    # At step 6, step 1 is still inside (prior_step > 6 - 6).
    assert ledger.before(6, "B") == (2, 2, 4)
    assert ledger.before(7, "B") == (1, 2, 5)
    assert ledger.before(8, "B") == (0, 2, 6)
    assert ledger.before(25, "B") == (0, 1, 23)
    assert ledger.before(26, "B") == (0, 0, None)
    assert not ledger.recipient_steps24


def test_same_step_repeat_and_no_future_observation():
    ledger = IncomingTransferLedger()
    ledger.observe(10, "TRANSFER", "B")
    ledger.observe(10, "TRANSFER", "B")
    assert ledger.before(10, "B") == (2, 2, 0)
    assert ledger.before(16, "B") == (0, 2, 6)
    with pytest.raises(ValueError, match="Nonchronological"):
        ledger.before(9, "B")


def test_cohort_aggregate_no_account_identifiers():
    c = Counts()
    c.add(1, 1, 2, 0)
    c.add(0, 0, 0, None)
    c.add(0, 0, 1, 3)
    r = c.report()
    assert r["rows"] == 3
    assert r["frauds"] == 1
    assert r["linked_24h"]["rows"] == 2
    assert r["linked_24h"]["frauds"] == 1
    assert r["unlinked_24h"]["fraud_rate"] == 0
    assert r["linked_24h"]["fraud_rate"] == pytest.approx(.5)
    assert r["age_hours_since_last_incoming_transfer_within_24h"]["fraud"][0] == 1
    assert "nameOrig" not in json.dumps(r)


def test_amount_strata_boundary():
    assert amount_band(0) == BANDS[0]
    assert amount_band(100) == BANDS[0]
    assert amount_band(100.01) == BANDS[1]
    assert amount_band(1000) == BANDS[1]
    assert amount_band(1000.01) == BANDS[2]


def test_report_refuses_existing_output_before_input(tmp_path):
    dest = tmp_path / "out.json"
    dest.write_text("DO NOT REPLACE")
    with pytest.raises(FileExistsError, match="overwritten"):
        run(tmp_path / "missing.csv", tmp_path / "missing.json", dest)
    assert dest.read_text() == "DO NOT REPLACE"
