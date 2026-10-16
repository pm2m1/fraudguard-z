"""Pure, preregistered FDH-inspired synthetic benchmark functions.

No full-data command-line entry point here. Full experiments require the
separately hash-locked runner. All training labels come from train days 0..53;
tuning uses *scores only* on days 54..71; days 72..89 remain reserved.

The generator is an independent, simplified FDH-inspired simulator, not
equivalent to original Handbook code or evidence from real transactions.
"""
import math
from collections import Counter, defaultdict, deque

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

STATIC_NAMES = (
    "log_amount",
    "sin_time_of_day", "cos_time_of_day",
    "sin_day_of_week", "cos_day_of_week",
)
HISTORY_NAMES = (
    "log_customer_count_prior_7days",
    "log_customer_amount_prior_7days",
    "log_terminal_count_prior_7days",
    "log_terminal_amount_prior_7days",
    "log_customer_terminal_pair_prior_7days",
)
TRAIN_END = 54
TUNE_END = 72
FINAL_END = 90
HISTORY_WINDOW_DAYS = 7
LABEL_DELAY_DAYS = 7
BUDGET = 0.01
BLOCK = 10000
MODEL_NAMES = ("hgb_static", "hgb_history")


def past_only_arrays(events, window_days=HISTORY_WINDOW_DAYS):
    """Construct chronological predictors from visible transaction fields.

    Fraud and fraud_scenario are never used by the feature engine. Each
    transaction is added to history *after* computing its own predictors.
    """
    if window_days < 1:
        raise ValueError("History window must be positive")
    size = len(events)
    x = np.empty((size, len(STATIC_NAMES) + len(HISTORY_NAMES)),
                 dtype=np.float32)
    days = np.empty(size, dtype=np.int16)
    y = np.empty(size, dtype=np.uint8)
    scenarios = np.empty(size, dtype=np.uint8)
    active = deque()
    customer_n = Counter()
    terminal_n = Counter()
    pair_n = Counter()
    customer_amount = defaultdict(float)
    terminal_amount = defaultdict(float)
    previous_time = -1

    for index, ev in enumerate(events):
        timestamp = int(ev.time_seconds)
        day = int(ev.day)
        amount = float(ev.amount)
        if (
            timestamp < previous_time or day != timestamp // 86400
            or day < 0 or not math.isfinite(amount) or amount < 0
        ):
            raise ValueError("Unsorted, inconsistent or invalid synthetic event")
        previous_time = timestamp
        customer = int(ev.customer_id)
        terminal = int(ev.terminal_id)
        while active and active[0][0] <= day - window_days:
            _, c, t, a = active.popleft()
            customer_n[c] -= 1
            terminal_n[t] -= 1
            pair_n[(c, t)] -= 1
            customer_amount[c] -= a
            terminal_amount[t] -= a
            if customer_n[c] == 0:
                del customer_n[c]
                del customer_amount[c]
            if terminal_n[t] == 0:
                del terminal_n[t]
                del terminal_amount[t]
            if pair_n[(c, t)] == 0:
                del pair_n[(c, t)]

        tod = (timestamp % 86400) / 86400
        dow = (day % 7) / 7
        x[index] = (
            math.log1p(amount),
            math.sin(2 * math.pi * tod), math.cos(2 * math.pi * tod),
            math.sin(2 * math.pi * dow), math.cos(2 * math.pi * dow),
            math.log1p(customer_n[customer]),
            math.log1p(max(0.0, customer_amount[customer])),
            math.log1p(terminal_n[terminal]),
            math.log1p(max(0.0, terminal_amount[terminal])),
            math.log1p(pair_n[(customer, terminal)]),
        )
        days[index] = day
        y[index] = int(ev.fraud)
        scenarios[index] = int(ev.fraud_scenario)
        if y[index] not in (0, 1) or scenarios[index] not in (0, 1, 2, 3):
            raise ValueError("Invalid synthetic label or fraud scenario")
        if (y[index] == 0) != (scenarios[index] == 0):
            raise ValueError("Fraud label disagrees with scenario")
        active.append((day, customer, terminal, amount))
        customer_n[customer] += 1
        terminal_n[terminal] += 1
        pair_n[(customer, terminal)] += 1
        customer_amount[customer] += amount
        terminal_amount[terminal] += amount
    return x, days, y, scenarios


def make_classifier():
    """Frozen HGB parameters match the previously fixed model family."""
    return HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=15,
        max_bins=128,
        learning_rate=0.07,
        l2_regularization=10.0,
        class_weight="balanced",
        early_stopping=False,
        random_state=42,
    )


def cutoff_from_unlabeled_scores(scores, fraction=BUDGET):
    """Compute score cutoff from tuning scores, never from fraud labels."""
    scores = np.asarray(scores, dtype=float)
    if (scores.ndim != 1 or not len(scores) or
            not np.isfinite(scores).all() or not 0 < fraction <= 1):
        raise ValueError("Invalid tuning score vector or budget")
    k = math.floor(len(scores) * fraction)
    if k == 0:
        raise ValueError("Insufficient tuning events")
    return float(np.partition(scores, -k)[-k])


def online_selection(scores, cutoff, budget=BUDGET, block=BLOCK):
    """Causal cutoff decisions bounded at every prefix of each 10k block."""
    scores = np.asarray(scores, dtype=float)
    if (
        scores.ndim != 1 or not np.isfinite(scores).all()
        or not math.isfinite(cutoff) or not 0 < budget <= 1
        or not isinstance(block, int) or block <= 0
    ):
        raise ValueError("Invalid streaming parameters")
    selected = np.zeros(len(scores), dtype=bool)
    consumed = 0
    for index, score in enumerate(scores):
        pos = index % block + 1
        if pos == 1:
            consumed = 0
        if score >= cutoff and consumed < math.floor(pos * budget + 1e-12):
            selected[index] = True
            consumed += 1
    return selected


def delayed_label_report(days, labels, selected, lag=LABEL_DELAY_DAYS):
    """Reveal test labels in reporting order; never update any policy."""
    if lag < 1 or not (len(days) == len(labels) == len(selected)):
        raise ValueError("Incorrect label delay inputs")
    waiting = deque()
    revealed = 0
    detected = 0
    previous_day = -1
    largest_pending = 0
    for day, fraud, decision in zip(days, labels, selected):
        day = int(day)
        if day < previous_day:
            raise ValueError("Days must be chronological")
        previous_day = day
        while waiting and waiting[0][0] <= day:
            _, y, alert = waiting.popleft()
            revealed += 1
            detected += y * alert
        waiting.append((day + lag, int(fraud), int(decision)))
        largest_pending = max(largest_pending, len(waiting))
    remaining = len(waiting)
    for _, fraud, decision in waiting:
        revealed += 1
        detected += fraud * decision
    return {
        "lag_days": lag,
        "released_during_final_stream": revealed - remaining,
        "released_after_final_watermark": remaining,
        "total_eventual_labels": revealed,
        "all_labels_released": revealed == len(labels),
        "detected_after_reporting_delay": detected,
        "maximum_pending": largest_pending,
    }


def describe(y, scores, reviews, days, scenarios):
    detected = int(y[reviews].sum())
    reviewed = int(reviews.sum())
    frauds = int(y.sum())
    recall = detected / frauds if frauds else None
    return {
        "transactions": int(len(y)),
        "frauds": frauds,
        "reviews": reviewed,
        "review_fraction": reviewed / len(y),
        "detected": detected,
        "false_alerts": reviewed - detected,
        "missed_frauds": frauds - detected,
        "precision": detected / reviewed if reviewed else None,
        "recall": recall,
        "pr_auc_ranking_only": (
            float(average_precision_score(y, scores))
            if 0 < frauds < len(y) else None
        ),
        "reviews_by_day": {
            str(int(day)): int(reviews[days == day].sum())
            for day in np.unique(days)
        },
        "detected_by_scenario": {
            str(s): int(np.sum(reviews & (y == 1) & (scenarios == s)))
            for s in (1, 2, 3)
        },
        "frauds_by_scenario": {
            str(s): int(np.sum((y == 1) & (scenarios == s)))
            for s in (1, 2, 3)
        },
        "delayed_label_accounting": delayed_label_report(days, y, reviews),
    }


def evaluate_events(events):
    """Run the two preregistered models on one COMPLETE synthetic run.

    Called only by the external hash-gated runner after all data files are
    generated, committed manifest hashes checked, and the one-time marker
    created. Tests may call it on hand-made data with meaningful class splits.
    """
    x, days, labels, scenarios = past_only_arrays(events)
    if (len(events) < 100 or days.min() != 0 or days.max() != 89 or
            set(np.unique(days)) != set(range(FINAL_END))):
        raise ValueError("Incomplete 90-day simulator stream")
    train = days < TRAIN_END
    tuning = (days >= TRAIN_END) & (days < TUNE_END)
    final = days >= TUNE_END
    if (
        not train.any() or not tuning.any() or not final.any()
        or set(np.unique(days[train])) != set(range(TRAIN_END))
        or set(np.unique(days[tuning])) != set(range(TRAIN_END, TUNE_END))
        or set(np.unique(days[final])) != set(range(TUNE_END, FINAL_END))
        or len(np.unique(labels[train])) != 2
    ):
        raise ValueError("Invalid training or chronological split")

    output = {}
    masks = {}
    final_days, final_y = days[final], labels[final]
    final_scenarios = scenarios[final]
    for model_name, width in (("hgb_static", len(STATIC_NAMES)),
                              ("hgb_history", x.shape[1])):
        classifier = make_classifier()
        classifier.fit(x[train, :width], labels[train])
        tune_scores = classifier.predict_proba(x[tuning, :width])[:, 1]
        cutoff = cutoff_from_unlabeled_scores(tune_scores)
        scores = classifier.predict_proba(x[final, :width])[:, 1]
        selected = online_selection(scores, cutoff)
        masks[model_name] = selected
        output[model_name] = {
            "feature_count": width,
            "tuning_cutoff_uncalibrated": cutoff,
            "evaluation": describe(
                final_y, scores, selected, final_days, final_scenarios
            ),
        }
    static = masks["hgb_static"]
    hist = masks["hgb_history"]
    only_hist = hist & ~static
    only_static = static & ~hist
    paired = {
        "extra_detected_history_minus_static": int(
            int(final_y[hist].sum()) - int(final_y[static].sum())
        ),
        "history_only_detected": int(np.sum(only_hist & (final_y == 1))),
        "static_only_detected": int(np.sum(only_static & (final_y == 1))),
        "history_only_reviews": int(only_hist.sum()),
        "static_only_reviews": int(only_static.sum()),
        "day_by_day_extra_detection": {
            str(day): int(np.sum((hist & (final_y == 1)) & (final_days == day))
                          - np.sum((static & (final_y == 1)) & (final_days == day)))
            for day in range(TUNE_END, FINAL_END)
        },
    }
    if (paired["extra_detected_history_minus_static"] !=
            paired["history_only_detected"] - paired["static_only_detected"]):
        raise RuntimeError("Paired detection accounting inconsistent")
    return {
        "train_rows": int(train.sum()),
        "tuning_rows": int(tuning.sum()),
        "final_rows": int(final.sum()),
        "train_steps": [0, 53],
        "tuning_steps": [54, 71],
        "final_steps": [72, 89],
        "maximum_review_fraction": BUDGET,
        "block_size": BLOCK,
        "label_delay_days": LABEL_DELAY_DAYS,
        "model_results": output,
        "paired_comparison": paired,
        "warning": (
            "Scores uncalibrated; same maximum 1% budget, not necessarily "
            "equal actual review volume. Label delay only changes accounting. "
            "Synthetic simulator results, not validation on real payments."
        ),
    }
