"""Independent full-context feature engine; never modifies frozen Model B.

Feature order and semantics are in notes/full_context_prelinuxone_plan.md.
Current event is observed only *after* a successful persistent decision commit.
"""
from collections import Counter, defaultdict, deque
from math import isfinite, log, log1p

from .features import TRANSACTION_TYPES, TYPE_TO_INDEX

FULL_FEATURE_NAMES = (
    "log_amount",
    *(f"txn_type_{kind}" for kind in TRANSACTION_TYPES),
    "log_sender_pre_balance",
    "amount_to_sender_balance",
    "sender_balance_zero",
    "log_sender_prior_out_count",
    "sender_out_count_6h",
    "log_sender_out_amount_6h",
    "sender_amount_vs_hist_mean",
    "log_pair_prior_count",
    "pair_count_24h",
    "first_time_recipient",
    "log_sender_distinct_dest_count",
    "log_recipient_distinct_source_count",
    "recipient_in_count_6h",
    "recipient_cashout_count_6h",
    "sender_current_dest_share_24h",
)
assert len(FULL_FEATURE_NAMES) == 21


def validate_event(row):
    kind = str(row["type"])
    sender, dest = str(row["nameOrig"]).strip(), str(row["nameDest"]).strip()
    step = int(row["step"])
    amount, balance = float(row["amount"]), float(row["oldbalanceOrg"])
    if kind not in TYPE_TO_INDEX or not sender or not dest or step < 1:
        raise ValueError("Invalid category, account or step")
    if not all(isfinite(v) and v >= 0 for v in (amount, balance)):
        raise ValueError("Invalid amount or balance")
    return (step, kind, sender, dest, amount, balance)


class FullContextHistory:
    """Read-before-write, deterministic PaySim event context.

    This in-memory component does not supply transaction idempotency or durable
    recovery by itself. The integration layer must enforce commit order, then
    call observe exactly once for each committed transaction.
    """

    def __init__(self):
        self.last_step = -1
        self.recent6 = deque()   # (step, sender, dest, amount, kind)
        self.recent24 = deque()  # (step, sender, dest)
        self.sender6_count = Counter()
        self.sender6_amount = defaultdict(float)
        self.recipient6_count = Counter()
        self.cashout6_count = Counter()  # account acting as origin on CASH_OUT
        self.sender24_count = Counter()
        self.pair24_count = Counter()
        self.sender_total_count = Counter()
        self.sender_total_amount = defaultdict(float)
        self.pair_total_count = Counter()
        self.sender_dests = defaultdict(set)
        self.recipient_sources = defaultdict(set)

    @staticmethod
    def _decrement(counter, key):
        counter[key] -= 1
        if counter[key] == 0:
            del counter[key]

    def _expire(self, step):
        if step < self.last_step:
            raise ValueError("Out-of-order transaction step")
        self.last_step = step
        while self.recent6 and self.recent6[0][0] <= step - 6:
            _, sender, dest, amount, kind = self.recent6.popleft()
            self._decrement(self.sender6_count, sender)
            self.sender6_amount[sender] -= amount
            if self.sender6_count.get(sender, 0) == 0:
                del self.sender6_amount[sender]
            self._decrement(self.recipient6_count, dest)
            if kind == "CASH_OUT":
                self._decrement(self.cashout6_count, sender)
        while self.recent24 and self.recent24[0][0] <= step - 24:
            _, sender, dest = self.recent24.popleft()
            self._decrement(self.sender24_count, sender)
            self._decrement(self.pair24_count, (sender, dest))

    def features(self, row):
        step, kind, sender, dest, amount, balance = validate_event(row)
        self._expire(step)
        total = self.sender_total_count[sender]
        mean = self.sender_total_amount[sender] / total if total else None
        pair = (sender, dest)
        sender24 = self.sender24_count[sender]
        values = (
            log1p(amount),
            *(float(kind == name) for name in TRANSACTION_TYPES),
            log1p(balance),
            min(amount / (balance + 1.0), 1e6),
            float(balance == 0),
            log1p(total),
            float(self.sender6_count[sender]),
            log1p(max(0.0, self.sender6_amount[sender])),
            log((amount + 1.0) / (mean + 1.0)) if mean is not None else 0.0,
            log1p(self.pair_total_count[pair]),
            float(self.pair24_count[pair]),
            float(self.pair_total_count[pair] == 0),
            log1p(len(self.sender_dests[sender])),
            log1p(len(self.recipient_sources[dest])),
            float(self.recipient6_count[dest]),
            float(self.cashout6_count[dest]),
            self.pair24_count[pair] / sender24 if sender24 else 0.0,
        )
        assert len(values) == len(FULL_FEATURE_NAMES)
        if not all(isfinite(v) for v in values):
            raise ValueError("Nonfinite context feature")
        return values

    def observe(self, row):
        """Call once only after corresponding SQLite decision has committed."""
        step, kind, sender, dest, amount, _ = validate_event(row)
        self._expire(step)
        pair = (sender, dest)
        self.recent6.append((step, sender, dest, amount, kind))
        self.sender6_count[sender] += 1
        self.sender6_amount[sender] += amount
        self.recipient6_count[dest] += 1
        if kind == "CASH_OUT":
            self.cashout6_count[sender] += 1
        self.recent24.append((step, sender, dest))
        self.sender24_count[sender] += 1
        self.pair24_count[pair] += 1
        self.sender_total_count[sender] += 1
        self.sender_total_amount[sender] += amount
        self.pair_total_count[pair] += 1
        self.sender_dests[sender].add(dest)
        self.recipient_sources[dest].add(sender)
