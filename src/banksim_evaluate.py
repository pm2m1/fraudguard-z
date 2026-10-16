"""One-time BankSim synthetic evaluation. Requires a committed hash lock.

Read the protocol before running. No final labels are read before all
locked input hashes and committed Git HEAD contents are verified.
"""
import csv
import json
import math
import subprocess
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .banksim_intake import DATA, LOCK, PROTOCOL, EXPECTED_ROWS, sha256

SCRIPT = Path(__file__).resolve()
OUTPUT = Path("reports/banksim_final_evaluation.json")
STARTED = Path("reports/banksim_final_evaluation_started.json")
MODEL_NAMES = ("lr_static", "hgb_static", "hgb_history", "hgb_amount_only")
BUDGET = .01
BLOCK = 10000
DELAY = 7
TUNE_START = 108
FINAL_START = 144


def committed(path):
    try:
        return subprocess.run(
            ["git", "show", "HEAD:" + path], check=True, capture_output=True
        ).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Lock and evaluator must be committed first") from error


def verify_lock():
    if not LOCK.exists():
        raise RuntimeError("Missing protocol lock: run banksim_intake --lock")
    raw = LOCK.read_bytes()
    if raw != committed(str(LOCK)) or SCRIPT.read_bytes() != committed(
        "src/banksim_evaluate.py"
    ):
        raise RuntimeError("Lock or evaluator does not match committed Git HEAD")
    lock = json.loads(raw)
    expected = {
        "dataset_sha256": sha256(DATA),
        "protocol_sha256": sha256(PROTOCOL),
        "intake_script_sha256": sha256(Path(__file__).with_name("banksim_intake.py")),
        "evaluator_script_sha256": sha256(SCRIPT),
    }
    if any(lock.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Frozen source, dataset or protocol checksum mismatch")
    meta = lock.get("metadata", {})
    if (
        lock.get("data_path") != str(DATA)
        or lock.get("final_data_evaluated") is not False
        or meta.get("rows") != EXPECTED_ROWS
        or meta.get("unique_steps") != 180
        or meta.get("earliest_step") != 0
        or meta.get("latest_step") != 179
        or meta.get("first_tuning_step") != TUNE_START
        or meta.get("first_final_step") != FINAL_START
    ):
        raise RuntimeError("Unexpected locked dataset or chronological boundaries")
    return lock


def read_rows():
    rows = []
    with DATA.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for index, row in enumerate(reader):
            step = int(row["step"])
            amount = float(row["amount"])
            label = int(row["fraud"])
            if (
                not 0 <= step <= 179 or not math.isfinite(amount)
                or amount < 0 or label not in (0, 1)
                or not all(row[name] for name in ("customer", "merchant", "category"))
            ):
                raise ValueError(f"Invalid BankSim transaction at row {index}")
            rows.append(
                (step, index, row["customer"], row["merchant"],
                 row["category"], amount, label)
            )
    if len(rows) != EXPECTED_ROWS:
        raise ValueError("BankSim row count changed")
    rows.sort(key=lambda row: (row[0], row[1]))
    if {row[0] for row in rows} != set(range(180)):
        raise ValueError("BankSim step sequence changed")
    return rows


def make_features(rows, tune_start=TUNE_START, window=7):
    """Use only current transaction and prior observed unlabeled events."""
    if window < 1:
        raise ValueError("Invalid feature history window")
    vocab = sorted({row[4] for row in rows if row[0] < tune_start})
    if not vocab:
        raise ValueError("No training categories")
    category_to_col = {category: index + 1 for index, category in enumerate(vocab)}
    width = 1 + len(vocab) + 5
    x = np.zeros((len(rows), width), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.uint8)
    steps = np.empty(len(rows), dtype=np.int16)
    active = deque()
    customer_n = Counter()
    merchant_n = Counter()
    pair_n = Counter()
    customer_amount = defaultdict(float)
    merchant_amount = defaultdict(float)
    previous_step = -1
    for i, row in enumerate(rows):
        step, _, customer, merchant, category, amount, label = row
        if step < previous_step:
            raise ValueError("Rows must be in chronological order")
        previous_step = step
        while active and active[0][0] <= step - window:
            _, c, m, a = active.popleft()
            customer_n[c] -= 1
            merchant_n[m] -= 1
            pair_n[(c, m)] -= 1
            customer_amount[c] -= a
            merchant_amount[m] -= a
            if customer_n[c] == 0:
                del customer_n[c]
                del customer_amount[c]
            if merchant_n[m] == 0:
                del merchant_n[m]
                del merchant_amount[m]
            if pair_n[(c, m)] == 0:
                del pair_n[(c, m)]
        x[i, 0] = math.log1p(amount)
        if category in category_to_col:
            x[i, category_to_col[category]] = 1.0
        first = 1 + len(vocab)
        x[i, first:] = [
            math.log1p(customer_n[customer]),
            math.log1p(max(0., customer_amount[customer])),
            math.log1p(merchant_n[merchant]),
            math.log1p(max(0., merchant_amount[merchant])),
            math.log1p(pair_n[(customer, merchant)]),
        ]
        steps[i] = step
        labels[i] = label
        active.append((step, customer, merchant, amount))
        customer_n[customer] += 1
        merchant_n[merchant] += 1
        pair_n[(customer, merchant)] += 1
        customer_amount[customer] += amount
        merchant_amount[merchant] += amount
    return x, steps, labels, vocab


def hgb():
    return HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=15, max_bins=128, learning_rate=.07,
        l2_regularization=10., class_weight="balanced", early_stopping=False,
        random_state=42
    )


def candidate_models(n_categories):
    static = list(range(n_categories + 1))
    history = list(range(n_categories + 6))
    return {
        "lr_static": (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced", solver="lbfgs",
                    max_iter=300, random_state=42
                ),
            ), static,
        ),
        "hgb_static": (hgb(), static),
        "hgb_history": (hgb(), history),
        "hgb_amount_only": (hgb(), [0]),
    }


def tuning_cutoff(scores, budget=BUDGET):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid validation scores")
    quota = math.floor(len(scores) * budget)
    if quota < 1:
        raise ValueError("Too few validation observations")
    return float(np.partition(scores, -quota)[-quota])


def streaming_reviews(scores, cutoff, budget=BUDGET, block=BLOCK):
    """Online block-prefix cap; no lookahead and no labels."""
    scores = np.asarray(scores, dtype=float)
    if (
        scores.ndim != 1 or not np.isfinite(scores).all()
        or not np.isfinite(cutoff) or not 0 < budget <= 1 or block < 1
    ):
        raise ValueError("Invalid streaming review inputs")
    decisions = np.zeros(len(scores), dtype=bool)
    used = 0
    for i, score in enumerate(scores):
        at = i % block + 1
        if at == 1:
            used = 0
        if used < math.floor(at * budget + 1e-12) and score >= cutoff:
            decisions[i] = True
            used += 1
    return decisions


def delayed_accounting(steps, labels, decisions, delay=DELAY):
    """Release labels only for reporting, never to scoring or policy."""
    if delay < 1:
        raise ValueError("Invalid reporting delay")
    waiting = deque()
    released = hits = 0
    max_pending = 0
    previous_step = -1
    for step, label, decision in zip(steps, labels, decisions):
        step = int(step)
        if step < previous_step:
            raise ValueError("Nonchronological label stream")
        previous_step = step
        while waiting and waiting[0][0] <= step:
            _, fraud, selected = waiting.popleft()
            released += 1
            hits += fraud * selected
        waiting.append((step + delay, int(label), int(decision)))
        max_pending = max(max_pending, len(waiting))
    after_final = len(waiting)
    for _, fraud, selected in waiting:
        released += 1
        hits += fraud * selected
    return {
        "delay_steps": delay,
        "labels_released_during_stream": released - after_final,
        "labels_released_after_final_watermark": after_final,
        "all_labels_released": released == len(labels),
        "true_positives_after_reporting_delay": hits,
        "max_pending_labels": max_pending,
    }


def describe(y, scores, selected, steps):
    reviews = int(selected.sum())
    true_positives = int(y[selected].sum())
    frauds = int(y.sum())
    report = {
        "transactions": len(y),
        "frauds": frauds,
        "reviews": reviews,
        "true_positives": true_positives,
        "false_positives": reviews - true_positives,
        "false_negatives": frauds - true_positives,
        "precision": true_positives / reviews if reviews else None,
        "recall": true_positives / frauds if frauds else None,
        "pr_auc_ranking_diagnostic": (
            float(average_precision_score(y, scores))
            if 0 < frauds < len(y) else None
        ),
        "review_counts_by_step": {
            str(int(step)): int(selected[steps == step].sum())
            for step in np.unique(steps)
        },
    }
    report["delayed_label_accounting"] = delayed_accounting(steps, y, selected)
    if report["delayed_label_accounting"]["true_positives_after_reporting_delay"] != true_positives:
        raise RuntimeError("Delayed label counts inconsistent with outcomes")
    return report


def evaluate():
    if STARTED.exists() or OUTPUT.exists():
        raise RuntimeError("Final evaluation already started: rerun refused")
    lock = verify_lock()
    STARTED.parent.mkdir(parents=True, exist_ok=True)
    with STARTED.open("x") as stream:
        json.dump({
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": lock["dataset_sha256"]
        }, stream, indent=2)
        stream.write("\n")
    print("Preparing locked BankSim data and past-only features...", flush=True)
    x, steps, y, categories = make_features(read_rows())
    train = steps < TUNE_START
    tune = (steps >= TUNE_START) & (steps < FINAL_START)
    final = steps >= FINAL_START
    if (
        not train.any() or not tune.any() or not final.any()
        or set(steps[train]) != set(range(108))
        or set(steps[tune]) != set(range(108, 144))
        or set(steps[final]) != set(range(144, 180))
    ):
        raise RuntimeError("Invalid locked chronological split")

    results = {}
    models = candidate_models(len(categories))
    for name in MODEL_NAMES:
        estimator, cols = models[name]
        print("Training " + name + "...", flush=True)
        estimator.fit(x[train][:, cols], y[train])
        score_method = (
            estimator.decision_function if name == "lr_static"
            else lambda values: estimator.predict_proba(values)[:, 1]
        )
        tuning_scores = score_method(x[tune][:, cols])
        cutoff = tuning_cutoff(tuning_scores)
        final_scores = score_method(x[final][:, cols])
        decisions = streaming_reviews(final_scores, cutoff)
        results[name] = {
            "unlabeled_tuning_score_cutoff": cutoff,
            "number_of_predictors": len(cols),
            "final": describe(y[final], final_scores, decisions, steps[final]),
        }
        print(
            name + ": reviews=" + str(results[name]["final"]["reviews"]) +
            " detected=" + str(results[name]["final"]["true_positives"]),
            flush=True,
        )
    base = results["hgb_static"]["final"]
    enhanced = results["hgb_history"]["final"]
    report = {
        "status": "ONE_TIME_INDEPENDENT_SIMULATOR_FINAL",
        "dataset_sha256": lock["dataset_sha256"],
        "evaluator_script_sha256": lock["evaluator_script_sha256"],
        "train_steps": [0, 107],
        "tuning_steps": [108, 143],
        "final_steps": [144, 179],
        "training_category_count": len(categories),
        "block_size": BLOCK,
        "max_review_fraction": BUDGET,
        "fraud_label_reporting_delay_steps": DELAY,
        "primary_comparison_hgb_history_minus_hgb_static": {
            "true_positive_difference": enhanced["true_positives"] - base["true_positives"],
            "reviews_difference": enhanced["reviews"] - base["reviews"],
            "pr_auc_difference": (
                enhanced["pr_auc_ranking_diagnostic"] -
                base["pr_auc_ranking_diagnostic"]
            ),
        },
        "models": results,
        "limitations": (
            "Synthetic BankSim payments, NOT real banking and not PaySim "
            "CASH_OUT. Thresholds based on unlabeled tuning scores, fixed "
            "hard 1% block-prefix review cap. Unused capacity not backfilled. "
            "Score calibration not established; PR-AUC retrospective ranking "
            "diagnostic only. Label delay affects accounting, not adaptation. "
            "No post-test model selection or deployment."
        ),
    }
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("Saved one-time report: " + str(OUTPUT))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", required=True)
    parser.parse_args()
    evaluate()


if __name__ == "__main__":
    main()
