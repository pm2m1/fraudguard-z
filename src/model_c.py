"""Past-only transaction features for experimental Model C.

Never reads fraud labels or post-transaction balances. The event is observed
only after its features have been extracted. This is a research feature engine;
it does not replace the production Model B schema or DecisionEngine.
"""

from collections import Counter, defaultdict, deque
from math import isfinite, log1p

from .features import (
    BASE_FEATURE_NAMES,
    TRANSACTION_TYPES,
    TYPE_TO_INDEX,
)


MODEL_C_FEATURE_NAMES = (
    *BASE_FEATURE_NAMES,
    "log_recipient_in_count_6h",
    "log_amount_over_sender_balance",
    "sender_balance_nearly_drained",
    "amount_exceeds_sender_balance",
    "log_sender_out_count_6h",
    "log_sender_out_amount_6h",
    "log_recipient_in_amount_6h",
    "log_recipient_unique_senders_6h",
    "log_steps_since_sender_event_capped_7h",
    "log_steps_since_recipient_event_capped_7h",
)


class ModelCHistory:
    """Aggregate sender and recipient activity in the prior six steps.

    For current step s, events from steps > s - 6 are active.
    Same-step events become visible only after observe is called.
    """

    def __init__(self, window_steps=6):
        if window_steps < 1:
            raise ValueError("Window length must be positive")
        self.window_steps = window_steps
        self.events = deque()
        self.sender_counts = Counter()
        self.recipient_counts = Counter()
        self.sender_amount = defaultdict(float)
        self.recipient_amount = defaultdict(float)
        self.recipient_senders = defaultdict(Counter)
        self.sender_last_step = {}
        self.recipient_last_step = {}
        self.last_step = -1

    def expire(self, step):
        if step < self.last_step:
            raise ValueError("Events are not step-ordered")
        self.last_step = step
        cutoff = step - self.window_steps
        while self.events and self.events[0][0] <= cutoff:
            _, sender, recipient, amount = self.events.popleft()
            self.sender_counts[sender] -= 1
            self.recipient_counts[recipient] -= 1
            self.sender_amount[sender] -= amount
            self.recipient_amount[recipient] -= amount
            pair_counts = self.recipient_senders[recipient]
            pair_counts[sender] -= 1
            if pair_counts[sender] == 0:
                del pair_counts[sender]
            if not pair_counts:
                del self.recipient_senders[recipient]
            if self.sender_counts[sender] == 0:
                del self.sender_counts[sender]
                del self.sender_amount[sender]
                del self.sender_last_step[sender]
            if self.recipient_counts[recipient] == 0:
                del self.recipient_counts[recipient]
                del self.recipient_amount[recipient]
                del self.recipient_last_step[recipient]

    def features(self, row):
        step = int(row["step"])
        sender = str(row["nameOrig"])
        recipient = str(row["nameDest"])
        kind = str(row["type"])
        amount = float(row["amount"])
        balance = float(row["oldbalanceOrg"])
        if kind not in TYPE_TO_INDEX:
            raise ValueError("Unknown transaction type")
        if not sender or not recipient or not sender.strip() or not recipient.strip():
            raise ValueError("Missing account ID")
        if step < 1 or not all(isfinite(v) and v >= 0 for v in (amount, balance)):
            raise ValueError("Invalid step, amount, or balance")

        self.expire(step)

        def recency(last_seen):
            # Seven is a distinct no-recent-event sentinel.
            return (log1p(step - last_seen) if last_seen is not None
                    else log1p(self.window_steps + 1))

        ratio = amount / (balance + 1.0)
        nearly_drained = balance > 0 and 0.95 * balance <= amount <= 1.01 * balance

        return (
            log1p(amount),
            *(float(kind == name) for name in TRANSACTION_TYPES),
            log1p(balance),
            log1p(self.recipient_counts[recipient]),
            log1p(min(ratio, 1e12)),
            float(nearly_drained),
            float(amount > balance),
            log1p(self.sender_counts[sender]),
            log1p(max(0.0, self.sender_amount[sender])),
            log1p(max(0.0, self.recipient_amount[recipient])),
            log1p(len(self.recipient_senders.get(recipient, ()))),
            recency(self.sender_last_step.get(sender)),
            recency(self.recipient_last_step.get(recipient)),
        )

    def observe(self, row):
        step = int(row["step"])
        sender = str(row["nameOrig"])
        recipient = str(row["nameDest"])
        amount = float(row["amount"])
        if step < 1 or not isfinite(amount) or amount < 0 or not sender or not recipient:
            raise ValueError("Invalid observation")
        self.expire(step)
        self.events.append((step, sender, recipient, amount))
        self.sender_counts[sender] += 1
        self.recipient_counts[recipient] += 1
        self.sender_amount[sender] += amount
        self.recipient_amount[recipient] += amount
        self.recipient_senders[recipient][sender] += 1
        self.sender_last_step[sender] = step
        self.recipient_last_step[recipient] = step
