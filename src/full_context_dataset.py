"""Development-only, chronological 21-feature dataset builder.

This module never fits a model, calibrates, evaluates a final holdout, or
changes the live decision engine. A caller supplies a CSV and explicitly
chooses a development boundary. The default matches the frozen PaySim
development cut (step <= 372); no label at a later step is read.
"""
import csv
from pathlib import Path

import numpy as np

from .full_context import FULL_FEATURE_NAMES, FullContextHistory, validate_event

DEVELOPMENT_END_STEP = 372
REQUIRED_COLUMNS = {
    "step", "type", "amount", "nameOrig", "nameDest",
    "oldbalanceOrg", "isFraud",
}


def iter_labelled_development_csv(path, *, max_step=DEVELOPMENT_END_STEP):
    """Yield raw rows in CSV source order; stop BEFORE final holdout rows.

    Does not cache or shuffle events. Caller must ensure the supplied CSV
    is the same documented, licensed PaySim copy, not a modified holdout.
    """
    if not isinstance(max_step, int) or not 1 <= max_step <= DEVELOPMENT_END_STEP:
        raise ValueError("Unsafe development boundary")
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or ()):
            raise ValueError("Missing required training CSV columns")
        previous_step = -1
        for index, row in enumerate(reader):
            step = int(row["step"])
            if step < previous_step:
                raise ValueError("CSV event steps are not chronological")
            if step > max_step:
                return  # Never consume fraud labels from already inspected test.
            previous_step = step
            event = {
                "source_row_index": index,
                "step": step,
                "type": row["type"],
                "amount": row["amount"],
                "oldbalanceOrg": row["oldbalanceOrg"],
                "nameOrig": row["nameOrig"],
                "nameDest": row["nameDest"],
            }
            # isFraud is read ONLY as the target, never added to the feature row.
            yield event, row["isFraud"]


def iter_feature_records(labelled_events, *, max_step=DEVELOPMENT_END_STEP):
    """Yield (cursor, 21 floats, binary label) in strict event order.

    A fresh FullContextHistory is created on each invocation; no current or
    future event is included in features. The cursor is (step, source index).
    """
    if not isinstance(max_step, int) or not 1 <= max_step <= DEVELOPMENT_END_STEP:
        raise ValueError("Unsafe development boundary")
    state = FullContextHistory()
    previous = (-1, -1)
    for event, label in labelled_events:
        step, *_ = validate_event(event)
        if step > max_step:
            return
        if "source_row_index" not in event:
            raise ValueError("Missing source row index")
        index = event["source_row_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Invalid source row index")
        cursor = (step, index)
        if cursor <= previous:
            raise ValueError("Rows must be strictly ordered by step and source index")
        if str(label) not in ("0", "1"):
            raise ValueError("Fraud label must be binary")
        features = state.features(event)
        state.observe(event)
        previous = cursor
        yield cursor, features, int(label)


def build_feature_batch(labelled_events, *, max_step=DEVELOPMENT_END_STEP):
    """Small in-memory test/dev helper; not suitable for all PaySim rows.

    Full training should stream iter_feature_records into bounded on-disk
    batches or a preallocated development matrix with a verified row count.
    """
    rows = list(iter_feature_records(labelled_events, max_step=max_step))
    X = np.asarray([features for _, features, _ in rows], dtype=np.float32)
    if not rows:
        X = np.empty((0, len(FULL_FEATURE_NAMES)), dtype=np.float32)
    y = np.asarray([label for _, _, label in rows], dtype=np.uint8)
    cursors = [cursor for cursor, _, _ in rows]
    if X.shape != (len(y), len(FULL_FEATURE_NAMES)):
        raise RuntimeError("Feature schema width mismatch")
    return X, y, cursors
