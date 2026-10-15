"""Chronological, past-only features for FraudGuard Z."""

from collections import Counter, deque
from math import isfinite, log1p

TRANSACTION_TYPES = (
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
)

TYPE_TO_INDEX = {
    name: i for i, name in enumerate(TRANSACTION_TYPES)
}

BASE_FEATURE_NAMES = (
    "log_amount",
    *(f"txn_type_{name}" for name in TRANSACTION_TYPES),
    "log_sender_pre_balance",
)

MODEL_B_FEATURE_NAMES = (
    *BASE_FEATURE_NAMES,
    "log_recipient_in_count_6h",
)


class RecipientHistory:
    """Recent destination activity in original CSV replay order.

    Retains events from steps strictly greater than step - 6.
    The current event is observed only after its features are read.
    """

    def __init__(self, window_steps=6):
        self.window_steps = window_steps
        self.events = deque()
        self.counts = Counter()
        self.last_step = -1

    def prior_count(self, recipient, step):
        if step < self.last_step:
            raise ValueError("Transactions are not step-ordered")

        self.last_step = step

        cutoff = step - self.window_steps

        while self.events and self.events[0][0] <= cutoff:
            _, old_recipient = self.events.popleft()
            self.counts[old_recipient] -= 1

            if self.counts[old_recipient] == 0:
                del self.counts[old_recipient]

        return self.counts.get(recipient, 0)

    def observe(self, recipient, step):
        """Call after the current row has been feature-extracted.

        During offline replay, this means the row is now observed.
        The live system will later call it only after DB commit.
        """
        self.prior_count(recipient, step)
        self.events.append((step, recipient))
        self.counts[recipient] += 1


def make_feature_row(row, history):
    """Return eight numeric features.

    Columns 0..6: Model A.
    Columns 0..7: Model B.

    Fraud labels and post-transaction balances are never read.
    """
    step = int(row["step"])
    txn_type = row["type"]
    recipient = row["nameDest"]

    if txn_type not in TYPE_TO_INDEX:
        raise ValueError(f"Unknown transaction type: {txn_type}")

    amount = float(row["amount"])
    balance = float(row["oldbalanceOrg"])

    if not isfinite(amount) or amount < 0:
        raise ValueError("Invalid amount")

    if not isfinite(balance) or balance < 0:
        raise ValueError("Invalid oldbalanceOrg")

    n3 = history.prior_count(recipient, step)

    return (
        log1p(amount),
        *(float(txn_type == t) for t in TRANSACTION_TYPES),
        log1p(balance),
        log1p(n3),
    )
