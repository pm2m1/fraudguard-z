"""Independent simulator benchmark: transaction-only vs past-only behavioral features.

Research only. Does not read or modify PaySim reports, frozen models, policy,
or live scoring database. The published Fraud Detection Handbook data is
synthetic, with a different schema, so this is NOT a cross-dataset test of
the trained PaySim Model B weights.

Input: trusted locally acquired daily .pkl files from the Handbook dataset.
WARNING: pandas pickle loading executes arbitrary code for malicious files.
Only open files downloaded from a source that you independently trust.
"""
import argparse
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from math import cos, isfinite, log1p, pi, sin
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DAY = 86400
BASIC_NAMES = ("log_amount", "hour_sin", "hour_cos", "weekend")
BEHAVIOR_NAMES = (
    *BASIC_NAMES,
    "log_customer_count_24h",
    "log_customer_mean_amount_24h",
    "log_amount_to_customer_mean",
    "log_terminal_count_24h",
    "log_terminal_unique_customers_24h",
    "log_customer_unique_terminals_24h",
    "log_customer_terminal_repeat_24h",
    "log_terminal_mean_amount_24h",
)
INPUT_COLUMNS = (
    "TRANSACTION_ID", "TX_TIME_SECONDS", "TX_TIME_DAYS",
    "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT", "TX_FRAUD",
)


class PastOnlyActivity:
    """A bounded 24-hour activity graph with no label-derived features."""

    def __init__(self, window_seconds=DAY):
        if window_seconds < 1:
            raise ValueError("Window must be positive")
        self.window = int(window_seconds)
        self.events = deque()
        self.customer = Counter()
        self.terminal = Counter()
        self.pair = Counter()
        self.customer_total = defaultdict(float)
        self.terminal_total = defaultdict(float)
        self.terminal_customers = defaultdict(Counter)
        self.customer_terminals = defaultdict(Counter)
        self.last_time = -1

    def expire(self, t):
        if t < self.last_time:
            raise ValueError("Transactions must be chronologically ordered")
        self.last_time = t
        cutoff = t - self.window
        while self.events and self.events[0][0] <= cutoff:
            _, customer, terminal, amount = self.events.popleft()
            self.customer[customer] -= 1
            self.terminal[terminal] -= 1
            self.pair[(customer, terminal)] -= 1
            self.customer_total[customer] -= amount
            self.terminal_total[terminal] -= amount
            self.terminal_customers[terminal][customer] -= 1
            self.customer_terminals[customer][terminal] -= 1
            if self.customer[customer] == 0:
                del self.customer[customer]
                del self.customer_total[customer]
            if self.terminal[terminal] == 0:
                del self.terminal[terminal]
                del self.terminal_total[terminal]
            if self.pair[(customer, terminal)] == 0:
                del self.pair[(customer, terminal)]
            if self.terminal_customers[terminal][customer] == 0:
                del self.terminal_customers[terminal][customer]
            if not self.terminal_customers[terminal]:
                del self.terminal_customers[terminal]
            if self.customer_terminals[customer][terminal] == 0:
                del self.customer_terminals[customer][terminal]
            if not self.customer_terminals[customer]:
                del self.customer_terminals[customer]

    def extract(self, *, timestamp, customer, terminal, amount):
        t = int(timestamp)
        a = float(amount)
        customer, terminal = str(customer), str(terminal)
        if t < 0 or not isfinite(a) or a < 0 or not customer or not terminal:
            raise ValueError("Invalid transaction")
        self.expire(t)
        hour = (t % DAY) / 3600
        weekday = (t // DAY) % 7  # simulator's day-zero offset; no label use
        c_count = self.customer[customer]
        t_count = self.terminal[terminal]
        mean_c = self.customer_total[customer] / c_count if c_count else 0.0
        mean_t = self.terminal_total[terminal] / t_count if t_count else 0.0
        basic = (
            log1p(a),
            sin(2 * pi * hour / 24),
            cos(2 * pi * hour / 24),
            float(weekday in (5, 6)),
        )
        behavioral = basic + (
            log1p(c_count),
            log1p(mean_c),
            log1p(a / (1.0 + mean_c)) if c_count else 0.0,
            log1p(t_count),
            log1p(len(self.terminal_customers[terminal])),
            log1p(len(self.customer_terminals[customer])),
            log1p(self.pair[(customer, terminal)]),
            log1p(mean_t),
        )
        return basic, behavioral

    def observe(self, *, timestamp, customer, terminal, amount):
        t = int(timestamp)
        a = float(amount)
        customer, terminal = str(customer), str(terminal)
        if t < 0 or not isfinite(a) or a < 0 or not customer or not terminal:
            raise ValueError("Invalid transaction")
        self.expire(t)
        self.events.append((t, customer, terminal, a))
        self.customer[customer] += 1
        self.terminal[terminal] += 1
        self.pair[(customer, terminal)] += 1
        self.customer_total[customer] += a
        self.terminal_total[terminal] += a
        self.terminal_customers[terminal][customer] += 1
        self.customer_terminals[customer][terminal] += 1


def load_features(input_dir, *, max_days=None):
    """Return arrays and day IDs; observations persist across splits, labels don't."""
    files = sorted(Path(input_dir).glob("????-??-??.pkl"))
    if max_days is not None:
        if max_days < 1:
            raise ValueError("max_days must be positive")
        files = files[:max_days]
    if len(files) < 10:
        raise ValueError("Need at least 10 daily files for temporal validation")

    # Two arrays of float32 rather than retaining all pandas frames in memory.
    basic_parts, enriched_parts, label_parts, day_parts = [], [], [], []
    history = PastOnlyActivity()
    prev_id = -1
    for day_no, file in enumerate(files):
        # Only load trusted original official dataset files; never untrusted pickle.
        df = pd.read_pickle(file)
        if not set(INPUT_COLUMNS).issubset(df.columns):
            raise ValueError(f"Missing expected columns in {file.name}")
        df = df.sort_values(["TX_TIME_SECONDS", "TRANSACTION_ID"], kind="stable")
        basic = np.empty((len(df), len(BASIC_NAMES)), dtype=np.float32)
        enriched = np.empty((len(df), len(BEHAVIOR_NAMES)), dtype=np.float32)
        labels = np.empty(len(df), dtype=np.uint8)
        for i, (tid, t, d, customer, terminal, amount, fraud) in enumerate(
            df.loc[:, INPUT_COLUMNS].itertuples(index=False, name=None)
        ):
            t, d, tid = int(t), int(d), int(tid)
            if tid <= prev_id:
                raise ValueError("Transaction IDs must strictly increase after sorting")
            prev_id = tid
            if d != t // DAY:
                raise ValueError("Transaction day and seconds do not agree")
            if fraud not in (0, 1):
                raise ValueError("Expected binary TX_FRAUD label")
            simple, complex_ = history.extract(
                timestamp=t, customer=customer, terminal=terminal, amount=amount)
            basic[i] = simple
            enriched[i] = complex_
            labels[i] = int(fraud)
            # Fraud label is never observed, even when stepping into validation/test.
            history.observe(timestamp=t, customer=customer,
                            terminal=terminal, amount=amount)
        basic_parts.append(basic)
        enriched_parts.append(enriched)
        label_parts.append(labels)
        day_parts.append(np.full(len(df), day_no, dtype=np.int16))
        if (day_no + 1) % 10 == 0:
            print(f"Prepared {day_no+1}/{len(files)} days", flush=True)

    Xb = np.concatenate(basic_parts)
    Xh = np.concatenate(enriched_parts)
    y = np.concatenate(label_parts)
    days = np.concatenate(day_parts)
    if not len(y) or not np.all(np.isfinite(Xh)):
        raise ValueError("Empty data or invalid features")
    return Xb, Xh, y, days, [f.name for f in files]


def temporal_masks(days, day_count):
    """Split by whole days; never by randomly shuffled transactions."""
    if day_count < 10:
        raise ValueError("At least ten complete days required")
    train_end = int(day_count * .60)
    valid_end = int(day_count * .80)
    masks = (days < train_end,
             (days >= train_end) & (days < valid_end),
             days >= valid_end)
    if any(not mask.any() for mask in masks):
        raise ValueError("Empty time partition")
    return masks, (train_end, valid_end)


def ranking_metrics(y, scores, fraction=.01):
    y, scores = np.asarray(y), np.asarray(scores)
    if len(y) == 0 or y.shape != scores.shape or y.ndim != 1:
        raise ValueError("Invalid metric vectors")
    if not np.all(np.isin(y, (0, 1))) or not np.all(np.isfinite(scores)):
        raise ValueError("Invalid labels or scores")
    if not 0 < fraction <= 1:
        raise ValueError("Invalid review budget")
    k = max(1, int(np.ceil(len(y) * fraction)))
    order = np.lexsort((np.arange(len(scores)), -scores))
    positives = int(y.sum())
    caught = int(y[order[:k]].sum())
    return {
        "rows": int(len(y)), "frauds": positives, "reviews": k,
        "true_positives": caught, "false_positives": k-caught,
        "precision_at_1pct": caught / k,
        "recall_at_1pct": caught / positives if positives else None,
        "pr_auc": float(average_precision_score(y, scores))
        if 0 < positives < len(y) else None,
    }


def benchmark(input_dir, *, max_days=None):
    Xb, Xh, y, days, filenames = load_features(input_dir, max_days=max_days)
    (train, valid, test), (train_end, valid_end) = temporal_masks(
        days, len(filenames))
    if any(len(np.unique(y[mask])) < 2 for mask in (train, valid, test)):
        raise RuntimeError("Each partition must contain fraud and legitimate examples")
    models = {
        "logistic_baseline": (Xb, make_pipeline(
            StandardScaler(), LogisticRegression(
                class_weight="balanced", max_iter=300, random_state=42))),
        "logistic_history": (Xh, make_pipeline(
            StandardScaler(), LogisticRegression(
                class_weight="balanced", max_iter=300, random_state=42))),
        "hgb_baseline": (Xb, HistGradientBoostingClassifier(
            max_iter=80, max_leaf_nodes=15, min_samples_leaf=50,
            learning_rate=.06, l2_regularization=10, random_state=42,
            early_stopping=False, class_weight="balanced")),
        "hgb_history": (Xh, HistGradientBoostingClassifier(
            max_iter=80, max_leaf_nodes=15, min_samples_leaf=50,
            learning_rate=.06, l2_regularization=10, random_state=42,
            early_stopping=False, class_weight="balanced")),
    }
    report = {
        "status": "INDEPENDENT SYNTHETIC SIMULATOR RESEARCH; NOT PRODUCTION",
        "source": "Fraud Detection Handbook, locally provided daily pickle files",
        "limitations": (
            "Separate models trained on a second synthetic simulator. This does NOT "
            "test the trained PaySim Model B on real transactions or show transfer "
            "of its weights. Top 1 percent is a retrospective GLOBAL ranking, not "
            "an online operational review policy. Frauds are model-independent labels."
        ),
        "feature_safety": "No fraud label, post-event field, or future transaction in inputs",
        "day_count": len(filenames), "rows": int(len(y)),
        "first_file": filenames[0], "last_file": filenames[-1],
        "train_days": [0, train_end-1],
        "validation_days": [train_end, valid_end-1],
        "test_days": [valid_end, len(filenames)-1],
        "basic_features": list(BASIC_NAMES),
        "history_features": list(BEHAVIOR_NAMES),
        "validation_results": {},
        "test_results": {},
    }
    trained = {}
    for name, (features, model) in models.items():
        print(f"Fitting {name}...", flush=True)
        model.fit(features[train], y[train])
        trained[name] = (features, model)
        val_scores = model.predict_proba(features[valid])[:, 1]
        report["validation_results"][name] = ranking_metrics(y[valid], val_scores)

    # A single predeclared selection rule, before accessing test labels.
    # Compare BOTH feature sets for the chosen model class in final report,
    # so a claim of feature improvement is apples-to-apples.
    def valid_score(name):
        res = report["validation_results"][name]
        return res["pr_auc"], res["recall_at_1pct"]
    winner = max(trained, key=lambda name: (valid_score(name), name))
    report["selected_on_validation"] = winner
    winner_family = winner.split("_")[0]
    test_names = [winner_family + "_baseline", winner_family + "_history"]
    for name in test_names:
        features, model = trained[name]
        scores = model.predict_proba(features[test])[:, 1]
        report["test_results"][name] = ranking_metrics(y[test], scores)
    report["run_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--max-days", type=int, default=None,
                        help="Developer smoke test ONLY; not independent final evaluation")
    parser.add_argument("--report", type=Path,
                        default=Path("reports/handbook_behavioral_research.json"))
    args = parser.parse_args()
    report = benchmark(args.input_dir, max_days=args.max_days)
    if args.max_days is not None:
        report["status"] = "PARTIAL-DATA DIAGNOSTIC ONLY; NOT FINAL EVALUATION"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["test_results"], indent=2))
    print(f"Report saved: {args.report}")


if __name__ == "__main__":
    main()
