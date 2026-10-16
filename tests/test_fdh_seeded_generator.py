import dataclasses

import pytest

from src.fdh_seeded_generator import (
    REGIMES, SEEDS, SimulationConfig, fingerprint,
    generate_events, generator_manifest, stream, validate,
)


TINY = SimulationConfig(
    n_customers=16, n_terminals=40, days=12,
    final_period_first_day=9, proximity_radius=24.,
)


def test_seed_determinism_and_different_seed():
    a = generate_events(104729, "default", TINY)
    b = generate_events(104729, "default", TINY)
    c = generate_events(130363, "default", TINY)
    assert len(a) > 0
    assert a == b
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(c)


def test_substream_independence():
    first = stream(104729, 11).random(10)
    _ = stream(104729, 20, 4).random(500)
    again = stream(104729, 11).random(10)
    assert (first == again).all()
    assert not (first == stream(130363, 11).random(10)).all()


def test_no_amount_shortcut_only_changes_scenario_labeling():
    base = generate_events(104729, "default", TINY)
    alt = generate_events(104729, "no_amount_shortcut", TINY)
    assert len(base) == len(alt)
    assert all(
        (a.time_seconds, a.customer_id, a.terminal_id, a.amount)
        == (b.time_seconds, b.customer_id, b.terminal_id, b.amount)
        for a, b in zip(base, alt)
    )
    assert all(b.fraud_scenario != 1 for b in alt)


def test_chronological_monotonicity_and_ids():
    events = generate_events(104729, "default", TINY)
    assert [e.transaction_id for e in events] == list(range(len(events)))
    assert [e.time_seconds for e in events] == sorted(e.time_seconds for e in events)
    assert all(0 <= e.day < TINY.days for e in events)
    assert all(e.fraud == int(e.fraud_scenario != 0) for e in events)


def test_no_global_numpy_rng_reset():
    import numpy as np
    np.random.seed(987)
    expected = np.random.random(4)
    np.random.seed(987)
    generate_events(104729, "default", TINY)
    actual = np.random.random(4)
    assert (expected == actual).all()


def test_fraud_shift_does_not_change_past_period_activity():
    default = generate_events(104729, "default", TINY)
    shift = generate_events(104729, "final_period_shift", TINY)
    before_default = [e for e in default if e.day < TINY.final_period_first_day]
    before_shift = [e for e in shift if e.day < TINY.final_period_first_day]
    assert before_default == before_shift


def test_reject_bad_configuration():
    with pytest.raises(ValueError):
        generate_events(-1, "default", TINY)
    with pytest.raises(ValueError):
        generate_events(1, "invalid", TINY)
    with pytest.raises(ValueError):
        generate_events(
            1, "default", dataclasses.replace(TINY, proximity_radius=0)
        )


def test_public_spec_has_fixed_regimes_and_seeds():
    spec = generator_manifest()
    assert spec["seeds"] == list(SEEDS)
    assert spec["regimes"] == list(REGIMES)
    assert spec["no_test_generation_without_lock"] is True
