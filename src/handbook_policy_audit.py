"""Post-hoc operational audit of the *already inspected* Handbook holdout.

Trains the previously selected histogram-gradient-boosting family with the
same declared hyperparameters. Cutoffs are frozen using validation scores.
Reports test workload drift and day-block-bootstrap uncertainty.

Research only: does NOT update models, policies, or frozen PaySim artifacts.
This is NOT a fresh independent test. It is follow-up diagnostics on an
already inspected Handbook test partition.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .handbook_research import load_features, ranking_metrics, temporal_masks


def validation_cutoff(scores, budget=.01):
    """Select cutoff with validation ONLY (ties can increase review volume)."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or not len(s) or not np.all(np.isfinite(s)):
        raise ValueError("Expected nonempty finite validation scores")
    if not 0 < budget <= 1:
        raise ValueError("Budget must be between 0 and 1")
    k = max(1, int(np.ceil(len(s) * budget)))
    return float(np.partition(s, len(s) - k)[len(s) - k])


def fixed_threshold_metrics(labels, scores, cutoff):
    y = np.asarray(labels)
    scores = np.asarray(scores)
    if y.ndim != 1 or y.shape != scores.shape or not len(y):
        raise ValueError("Invalid labels/scores")
    if not np.all(np.isin(y, (0, 1))) or not np.all(np.isfinite(scores)):
        raise ValueError("Invalid labels/scores")
    if not np.isfinite(cutoff):
        raise ValueError("Nonfinite cutoff")
    selection = scores >= cutoff
    tp = int(y[selection].sum())
    reviews = int(selection.sum())
    frauds = int(y.sum())
    return {
        "rows": len(y),
        "frauds": frauds,
        "reviews": reviews,
        "true_positives": tp,
        "false_positives": reviews - tp,
        "false_negatives": frauds - tp,
        "review_rate": reviews / len(y),
        "precision": tp / reviews if reviews else None,
        "recall": tp / frauds if frauds else None,
    }


def daily_fixed_counts(days, labels, scores, cutoff):
    """Per-day sufficient statistics for conditional paired bootstrap.

    Columns: total, fraud, reviewed, true_positives.
    """
    d = np.asarray(days)
    y = np.asarray(labels)
    s = np.asarray(scores)
    if not (d.ndim == y.ndim == s.ndim == 1 and len(d) == len(y) == len(s)):
        raise ValueError("Mismatched arrays")
    if len(d) == 0 or not np.all(np.isfinite(s)):
        raise ValueError("Empty or nonfinite inputs")
    unique_days = np.unique(d)
    rows = np.zeros((len(unique_days), 4), dtype=np.int64)
    for i, day in enumerate(unique_days):
        match = d == day
        selected = match & (s >= cutoff)
        rows[i] = (int(match.sum()), int(y[match].sum()),
                   int(selected.sum()), int(y[selected].sum()))
    return unique_days, rows


def moving_block_confidence_intervals(baseline_daily, historical_daily,
                                      *, iterations=2000, block_days=7,
                                      random_seed=2026):
    """Circular, paired moving-block bootstrap of differences.

    Identical day blocks are drawn for both methods to preserve the pairing.
    These CIs are descriptive, not proof of causal or real-world superiority.
    """
    b = np.asarray(baseline_daily, dtype=np.int64)
    h = np.asarray(historical_daily, dtype=np.int64)
    if b.ndim != 2 or b.shape != h.shape or b.shape[1] != 4 or not len(b):
        raise ValueError("Expected aligned daily (total, fraud, reviews, TP)")
    if block_days < 1 or iterations < 10:
        raise ValueError("Invalid bootstrap configuration")
    n = len(b)
    rng = np.random.default_rng(random_seed)
    delta_recall = []
    delta_review_rate = []
    delta_precision = []
    for _ in range(iterations):
        starts = rng.integers(0, n, size=(n + block_days - 1) // block_days)
        chosen = np.concatenate([
            (int(start) + np.arange(block_days)) % n for start in starts
        ])[:n]
        bs = b[chosen].sum(axis=0)
        hs = h[chosen].sum(axis=0)
        fraud = int(bs[1])
        if fraud:
            delta_recall.append(hs[3] / fraud - bs[3] / fraud)
        delta_review_rate.append((hs[2] - bs[2]) / bs[0])
        if bs[2] and hs[2]:
            delta_precision.append(hs[3] / hs[2] - bs[3] / bs[2])

    def summarise(values):
        if not values:
            return {"lower_95pct": None, "upper_95pct": None, "median": None}
        v = np.asarray(values)
        q = np.quantile(v, [.025, .5, .975])
        return {"lower_95pct": float(q[0]),
                "median": float(q[1]),
                "upper_95pct": float(q[2])}

    return {
        "method": "paired circular moving-day-block bootstrap",
        "iterations": iterations, "block_days": block_days,
        "random_seed": random_seed,
        "history_minus_baseline_recall": summarise(delta_recall),
        "history_minus_baseline_review_rate": summarise(delta_review_rate),
        "history_minus_baseline_precision": summarise(delta_precision),
        "interpretation": (
            "Descriptive uncertainty on the previously inspected synthetic test "
            "period; consecutive days may be dependent, and block length is "
            "an analysis choice. No multiple-comparison adjustment."
        ),
    }


def make_candidate():
    return HistGradientBoostingClassifier(
        max_iter=80, max_leaf_nodes=15, min_samples_leaf=50,
        learning_rate=.06, l2_regularization=10, random_state=42,
        early_stopping=False, class_weight="balanced",
    )


def audit(input_dir, baseline_report, *, bootstrap_iterations=2000):
    prior = json.loads(Path(baseline_report).read_text())
    if (
        prior.get("status") != "INDEPENDENT SYNTHETIC SIMULATOR RESEARCH; NOT PRODUCTION"
        or prior.get("selected_on_validation") != "hgb_history"
        or prior.get("day_count") != 183
    ):
        raise RuntimeError("Expected complete, published Handbook HGB research report")
    Xb, Xh, y, days, files = load_features(input_dir)
    if len(files) != 183 or len(y) != prior["rows"]:
        raise RuntimeError("Dataset does not match original research run")
    (train, valid, test), _ = temporal_masks(days, len(files))
    comparisons = {}
    raw_daily = {}
    for name, X in (("hgb_baseline", Xb), ("hgb_history", Xh)):
        print(f"Training audit {name}...", flush=True)
        model = make_candidate()
        model.fit(X[train], y[train])
        validation_scores = model.predict_proba(X[valid])[:, 1]
        threshold = validation_cutoff(validation_scores)
        test_scores = model.predict_proba(X[test])[:, 1]

        # Fail closed on unexpected discrepancies with the first test evaluation.
        check = ranking_metrics(y[test], test_scores)
        expected = prior["test_results"][name]
        if (check["true_positives"] != expected["true_positives"]
                or not np.isclose(check["pr_auc"], expected["pr_auc"],
                                  atol=1e-10, rtol=1e-10)):
            raise RuntimeError(f"{name} predictions do not reproduce prior report")

        daily_days, counts = daily_fixed_counts(
            days[test], y[test], test_scores, threshold
        )
        raw_daily[name] = counts
        comparisons[name] = {
            "cutoff_from_validation": threshold,
            "validation_review_rate": fixed_threshold_metrics(
                y[valid], validation_scores, threshold)["review_rate"],
            "test_fixed_threshold": fixed_threshold_metrics(
                y[test], test_scores, threshold),
            "test_retrospective_global_top_1pct": check,
            "daily_operational_counts": [
                {"day_index": int(day), "transactions": int(row[0]),
                 "frauds": int(row[1]), "reviews": int(row[2]),
                 "true_positives": int(row[3])}
                for day, row in zip(daily_days, counts)
            ],
        }

    base = comparisons["hgb_baseline"]["test_fixed_threshold"]
    hist = comparisons["hgb_history"]["test_fixed_threshold"]
    bootstrap = moving_block_confidence_intervals(
        raw_daily["hgb_baseline"], raw_daily["hgb_history"],
        iterations=bootstrap_iterations
    )
    return {
        "status": "FOLLOW-UP AUDIT; TEST ALREADY INSPECTED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_report": str(baseline_report),
        "total_rows": len(y),
        "test_days": int(len(np.unique(days[test]))),
        "method": (
            "Reuse HGB models with identical training hyperparameters. Select "
            "separate score thresholds on validation for approximate 1% review. "
            "Measure actual (potentially different) test review rates and daily "
            "paired moving-block confidence intervals."
        ),
        "caveats": (
            "Test labels and ranking results were already inspected before this "
            "audit; do not claim this as a fresh holdout. Cutoffs are based on "
            "validation and can drift away from 1% on test. HGB scores with "
            "balanced class weights are NOT calibrated probabilities. The "
            "dataset remains synthetic. Offline feature computation uses prior "
            "unlabeled events in temporal order."
        ),
        "comparisons": comparisons,
        "delta_fixed_threshold": {
            "additional_frauds": (
                hist["true_positives"] - base["true_positives"]),
            "additional_false_alerts": (
                hist["false_positives"] - base["false_positives"]),
            "review_rate_percentage_points": (
                (hist["review_rate"] - base["review_rate"]) * 100),
            "recall_percentage_points": (
                (hist["recall"] - base["recall"]) * 100),
        },
        "paired_uncertainty": bootstrap,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--baseline-report", type=Path,
                        default=Path("reports/handbook_behavioral_research.json"))
    parser.add_argument("--report", type=Path,
                        default=Path("reports/handbook_policy_audit.json"))
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()
    if not 100 <= args.bootstrap_iterations <= 10000:
        parser.error("--bootstrap-iterations must be between 100 and 10000")
    result = audit(args.input_dir, args.baseline_report,
                   bootstrap_iterations=args.bootstrap_iterations)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "test_fixed_threshold": {
            k: v["test_fixed_threshold"] for k, v in result["comparisons"].items()
        },
        "delta_fixed_threshold": result["delta_fixed_threshold"],
        "paired_uncertainty": result["paired_uncertainty"],
    }, indent=2))
    print(f"Saved {args.report}")


if __name__ == "__main__":
    main()
