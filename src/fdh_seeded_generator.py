"""Fresh Fraud Detection Handbook-inspired simulator, independently seeded.

This is an original implementation of the *published scenario descriptions*,
not copied notebook code. The upstream simulator code is GPL-3.0; license
compatibility must be reviewed before distributing adapted components.

IMPORTANT: The command line intentionally provides ONLY --spec. No full
experiment dataset can be created from the CLI until the generator and
evaluator, their fixed settings, and the replication lock are committed.
Unit tests may call generate_events with deliberately tiny fixtures.

No global random state, no labels used to generate transaction features, and
different random streams for profiles, activity and fraud-scenario schedules.
"""
import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np

UPSTREAM_NOTEBOOK_COMMIT = "eabce290e52013e1672963d3728e3d63cd8b0705"
UPSTREAM_NOTEBOOK_BLOB = "3af320f0098f23c11e7fb7ff8262d9cca59daffe"
SEEDS = (104729, 130363, 155921)
REGIMES = ("default", "no_amount_shortcut", "final_period_shift")


@dataclass(frozen=True)
class SimulationConfig:
    n_customers: int = 600
    n_terminals: int = 3000
    days: int = 90
    proximity_radius: float = 12.0
    mean_tx_per_day_min: float = 0.3
    mean_tx_per_day_max: float = 2.2
    amount_mean_min: float = 5.0
    amount_mean_max: float = 100.0
    fraud_amount_threshold: float = 220.0
    terminal_compromises_per_day: int = 2
    terminal_compromise_duration_days: int = 28
    customer_compromises_per_day: int = 3
    customer_compromise_duration_days: int = 14
    compromised_customer_tx_fraction: float = 1.0 / 3.0
    compromised_amount_multiplier: float = 5.0
    final_period_first_day: int = 72
    final_period_amount_multiplier: float = 1.35
    final_period_terminal_compromises_per_day: int = 3
    final_period_customer_compromises_per_day: int = 5


class Event(NamedTuple):
    transaction_id: int
    time_seconds: int
    day: int
    customer_id: int
    terminal_id: int
    amount: float
    fraud: int
    fraud_scenario: int


def validate(config, seed, regime):
    if (
        not isinstance(seed, int) or seed < 0 or seed >= 2**32
        or regime not in REGIMES
        or config.n_customers < 1 or config.n_terminals < 1
        or config.days < 3 or config.proximity_radius <= 0
        or not 0 < config.mean_tx_per_day_min <= config.mean_tx_per_day_max
        or not 0 < config.amount_mean_min <= config.amount_mean_max
        or config.fraud_amount_threshold <= 0
        or not 0 <= config.compromised_customer_tx_fraction <= 1
        or config.compromised_amount_multiplier <= 1
        or config.terminal_compromise_duration_days < 1
        or config.customer_compromise_duration_days < 1
        or config.terminal_compromises_per_day < 0
        or config.customer_compromises_per_day < 0
        or config.final_period_first_day < 1
        or config.final_period_amount_multiplier <= 0
        or config.final_period_terminal_compromises_per_day < 0
        or config.final_period_customer_compromises_per_day < 0
    ):
        raise ValueError("Invalid frozen simulator configuration")
    values = asdict(config).values()
    if any(isinstance(x, float) and not math.isfinite(x) for x in values):
        raise ValueError("Nonfinite simulator setting")


def stream(seed, domain, key=0):
    """Stable substream; NEVER uses Python hash(), random.seed or np.seed."""
    return np.random.default_rng(np.random.SeedSequence([seed, domain, key]))


def generator_manifest():
    return {
        "purpose": "CONFIG_INSPECTION_ONLY_NO_DATA_GENERATION",
        "upstream_notebook_commit": UPSTREAM_NOTEBOOK_COMMIT,
        "upstream_notebook_blob": UPSTREAM_NOTEBOOK_BLOB,
        "implementation": "original FDH-inspired separate seeded generator",
        "randomness": "numpy SeedSequence([seed, domain, identity]); no global RNG",
        "seeds": list(SEEDS),
        "regimes": list(REGIMES),
        "settings": asdict(SimulationConfig()),
        "source_license_notice": (
            "Upstream Fraud Detection Handbook notebook code: GPL-3.0. "
            "Current implementation is independently written from concepts; "
            "attribution/license review required before reuse in submission."
        ),
        "no_test_generation_without_lock": True,
    }


def generate_events(seed, regime, config=None):
    """Create a new synthetic sequence; DO NOT call on full settings pre-lock.

    Profiles and base activities of a given seed are the same across regimes.
    Different seeds change ALL profile/activity/fraud substreams. Overlapping
    active compromise windows label transactions; scenario 3 multiplies the
    transaction amount once, and overrides earlier scenario labels.
    """
    if config is None:
        config = SimulationConfig()
    validate(config, seed, regime)

    # Separate streams guarantee fraud configuration does not alter activity.
    customers = stream(seed, 11)
    terminal_rng = stream(seed, 12)
    positions = terminal_rng.uniform(0.0, 100.0, (config.n_terminals, 2))
    cx = customers.uniform(0.0, 100.0, (config.n_customers, 2))
    mean_amounts = customers.uniform(
        config.amount_mean_min, config.amount_mean_max, config.n_customers
    )
    daily_rates = customers.uniform(
        config.mean_tx_per_day_min, config.mean_tx_per_day_max,
        config.n_customers
    )

    activities = []
    seq = 0
    for customer_id in range(config.n_customers):
        dist = np.sum((positions - cx[customer_id]) ** 2, axis=1)
        eligible = np.flatnonzero(dist < config.proximity_radius ** 2)
        if not len(eligible):
            # Explicit fallback to nearest terminal prevents empty profiles.
            eligible = np.array([int(dist.argmin())])
        rng = stream(seed, 20, customer_id)
        for day in range(config.days):
            n = int(rng.poisson(daily_rates[customer_id]))
            for _ in range(n):
                second = int(rng.normal(43200.0, 20000.0))
                if second <= 0 or second >= 86400:
                    continue
                amount = rng.normal(mean_amounts[customer_id],
                                    mean_amounts[customer_id] / 2)
                if amount < 0:
                    amount = rng.uniform(0, mean_amounts[customer_id] * 2)
                if (regime == "final_period_shift"
                        and day >= config.final_period_first_day):
                    amount *= config.final_period_amount_multiplier
                amount = round(float(amount), 2)
                merchant = int(rng.choice(eligible))
                activities.append(
                    (day * 86400 + second, seq, day, customer_id,
                     merchant, amount)
                )
                seq += 1
    activities.sort(key=lambda e: (e[0], e[1]))

    terminal_schedule = []
    customer_schedule = []
    for day in range(config.days):
        shifted = (
            regime == "final_period_shift"
            and day >= config.final_period_first_day
        )
        t_count = (
            config.final_period_terminal_compromises_per_day if shifted
            else config.terminal_compromises_per_day
        )
        c_count = (
            config.final_period_customer_compromises_per_day if shifted
            else config.customer_compromises_per_day
        )
        t_rng = stream(seed, 31, day)
        c_rng = stream(seed, 32, day)
        terminal_schedule.append(set(map(int, t_rng.choice(
            config.n_terminals,
            size=min(config.n_terminals, t_count),
            replace=False
        ))))
        customer_schedule.append(set(map(int, c_rng.choice(
            config.n_customers,
            size=min(config.n_customers, c_count),
            replace=False
        ))))

    active_terminals = []
    active_customers = []
    for day in range(config.days):
        active_terminals.append(set().union(
            *terminal_schedule[max(0, day -
                config.terminal_compromise_duration_days + 1):day + 1]
        ))
        active_customers.append(set().union(
            *customer_schedule[max(0, day -
                config.customer_compromise_duration_days + 1):day + 1]
        ))

    events = []
    fraud_rng = None
    current_day = -1
    for time_seconds, _, day, customer, terminal, initial_amount in activities:
        if day != current_day:
            fraud_rng = stream(seed, 41, day)
            current_day = day
        amount = initial_amount
        fraud_scenario = 0

        # Explicitly simple amount shortcut, entirely removed in ablation.
        if regime != "no_amount_shortcut" and amount > config.fraud_amount_threshold:
            fraud_scenario = 1
        if terminal in active_terminals[day]:
            fraud_scenario = 2
        if (customer in active_customers[day]
                and fraud_rng.random() < config.compromised_customer_tx_fraction):
            amount = round(amount * config.compromised_amount_multiplier, 2)
            fraud_scenario = 3

        events.append(Event(
            transaction_id=len(events), time_seconds=time_seconds, day=day,
            customer_id=customer, terminal_id=terminal, amount=amount,
            fraud=int(fraud_scenario != 0), fraud_scenario=fraud_scenario
        ))
    return events


def fingerprint(events):
    """Stable sample fingerprint; never uses object repr or hash()."""
    hasher = hashlib.sha256()
    for event in events:
        hasher.update((json.dumps(event._asdict(), sort_keys=True,
                                  separators=(",", ":")) + "\n").encode("utf-8"))
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", action="store_true", required=True,
                        help="Print fixed parameters; do not generate data")
    parser.parse_args()
    print(json.dumps(generator_manifest(), indent=2))


if __name__ == "__main__":
    main()
