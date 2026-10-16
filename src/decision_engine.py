import hashlib
import json
import sqlite3
import threading
from math import expm1, isfinite
from pathlib import Path

import numpy as np

from .privacy import AccountPseudonymizer, verify_database_mode

from .features import (
    MODEL_B_FEATURE_NAMES,
    RecipientHistory,
    TYPE_TO_INDEX,
    make_feature_row,
)


SCHEMA_VERSION = "fgz_ab_v1"


def normalize_transaction(raw):
    transaction = {
        "event_id": str(raw["event_id"]).strip(),
        "source_row_index": int(raw["source_row_index"]),
        "step": int(raw["step"]),
        "type": str(raw["type"]),
        "amount": float(raw["amount"]),
        "nameOrig": str(raw["nameOrig"]).strip(),
        "nameDest": str(raw["nameDest"]).strip(),
        "oldbalanceOrg": float(raw["oldbalanceOrg"]),
    }

    if not transaction["event_id"]:
        raise ValueError("Missing event ID")

    if transaction["step"] < 1:
        raise ValueError("Invalid step")

    if transaction["source_row_index"] < 0:
        raise ValueError("Invalid source row index")

    if transaction["type"] not in TYPE_TO_INDEX:
        raise ValueError("Unknown transaction type")

    if not transaction["nameOrig"] or not transaction["nameDest"]:
        raise ValueError("Missing account identifier")

    for field in ("amount", "oldbalanceOrg"):
        value = transaction[field]
        if not isfinite(value) or value < 0:
            raise ValueError(f"Invalid {field}")

    return transaction


def payload_hash(transaction):
    content = json.dumps(
        transaction,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(content.encode()).hexdigest()


class DecisionEngine:
    def __init__(self, db_path, model, calibrator, threshold, privacy_key=None):
        if len(MODEL_B_FEATURE_NAMES) != 8:
            raise ValueError("Unexpected feature schema")

        if not 0 < threshold < 1:
            raise ValueError("Invalid review threshold")

        self.pseudonymizer = (AccountPseudonymizer(privacy_key) if privacy_key is not None else None)
        self.model = model
        self.calibrator = calibrator
        self.threshold = float(threshold)
        self.lock = threading.RLock()
        self.healthy = False

        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.db = sqlite3.connect(
            path,
            check_same_thread=False,
            timeout=10,
            isolation_level=None,
        )
        self.db.row_factory = sqlite3.Row

        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA busy_timeout=10000")

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                event_id TEXT PRIMARY KEY,
                payload_hash TEXT NOT NULL,
                step INTEGER NOT NULL,
                source_row_index INTEGER NOT NULL,
                originator TEXT NOT NULL,
                recipient TEXT NOT NULL,
                txn_type TEXT NOT NULL,
                amount REAL NOT NULL,
                sender_balance REAL NOT NULL,
                n3 INTEGER NOT NULL,
                features_json TEXT NOT NULL,
                risk REAL NOT NULL,
                decision TEXT NOT NULL
                    CHECK(decision IN ('ALLOW', 'REVIEW')),
                schema_version TEXT NOT NULL,
                UNIQUE(step, source_row_index)
            )
        """)

        verify_database_mode(self.db, self.pseudonymizer)
        self.rebuild()

    def rebuild(self):
        with self.lock:
            self.healthy = False
            history = RecipientHistory(window_steps=6)
            last_position = (-1, -1)

            rows = self.db.execute("""
                SELECT step, source_row_index, recipient,
                       schema_version
                FROM decisions
                ORDER BY step, source_row_index
            """)

            for row in rows:
                if row["schema_version"] != SCHEMA_VERSION:
                    raise ValueError("Database schema mismatch")

                history.observe(row["recipient"], row["step"])
                last_position = (
                    row["step"],
                    row["source_row_index"],
                )

            self.history = history
            self.last_position = last_position
            self.healthy = True

    def score(self, raw):
        transaction = normalize_transaction(raw)
        if self.pseudonymizer is not None:
            transaction["nameOrig"] = self.pseudonymizer.encode(transaction["nameOrig"])
            transaction["nameDest"] = self.pseudonymizer.encode(transaction["nameDest"])
        fingerprint = payload_hash(transaction)
        event_id = transaction["event_id"]

        with self.lock:
            if not self.healthy:
                raise RuntimeError(
                    "State is unavailable. Rebuild required."
                )

            state_touched = False

            try:
                self.db.execute("BEGIN IMMEDIATE")

                existing = self.db.execute(
                    """
                    SELECT payload_hash, risk, decision, n3
                    FROM decisions WHERE event_id = ?
                    """,
                    (event_id,),
                ).fetchone()

                if existing is not None:
                    self.db.rollback()

                    if existing["payload_hash"] != fingerprint:
                        raise ValueError(
                            "Event ID already exists with different data"
                        )

                    return {
                        "event_id": event_id,
                        "risk": existing["risk"],
                        "decision": existing["decision"],
                        "n3": existing["n3"],
                        "duplicate": True,
                    }

                latest = self.db.execute("""
                    SELECT step, source_row_index
                    FROM decisions
                    ORDER BY step DESC, source_row_index DESC
                    LIMIT 1
                """).fetchone()

                db_position = (
                    (latest["step"], latest["source_row_index"])
                    if latest is not None else (-1, -1)
                )

                if db_position != self.last_position:
                    self.healthy = False
                    raise RuntimeError(
                        "Database changed. Rebuild required."
                    )

                position = (
                    transaction["step"],
                    transaction["source_row_index"],
                )

                if position <= self.last_position:
                    raise ValueError(
                        "New transactions must be in replay order"
                    )

                state_touched = True
                features = make_feature_row(
                    transaction, self.history
                )

                n3 = int(round(expm1(features[-1])))

                X = np.asarray(
                    [features], dtype=np.float32
                )

                margin = self.model.decision_function(X)
                margin = np.asarray(margin).reshape(-1, 1)

                risk = float(
                    self.calibrator.predict_proba(margin)[0, 1]
                )

                if not isfinite(risk) or not 0 <= risk <= 1:
                    raise ValueError("Invalid model output")

                decision = (
                    "REVIEW" if risk >= self.threshold else "ALLOW"
                )

                self.db.execute("""
                    INSERT INTO decisions (
                        event_id, payload_hash, step,
                        source_row_index, originator, recipient,
                        txn_type, amount, sender_balance,
                        n3, features_json, risk, decision,
                        schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_id,
                    fingerprint,
                    transaction["step"],
                    transaction["source_row_index"],
                    transaction["nameOrig"],
                    transaction["nameDest"],
                    transaction["type"],
                    transaction["amount"],
                    transaction["oldbalanceOrg"],
                    n3,
                    json.dumps(list(features)),
                    risk,
                    decision,
                    SCHEMA_VERSION,
                ))

                self.db.commit()

            except Exception:
                if self.db.in_transaction:
                    self.db.rollback()

                if state_touched:
                    self.healthy = False

                raise

            # SQLite is committed before history changes.
            try:
                self.history.observe(
                    transaction["nameDest"],
                    transaction["step"],
                )
                self.last_position = position

            except Exception as error:
                self.healthy = False
                raise RuntimeError(
                    "Decision committed but state update failed. "
                    "Rebuild required."
                ) from error

            return {
                "event_id": event_id,
                "risk": risk,
                "decision": decision,
                "n3": n3,
                "duplicate": False,
            }

    def status(self):
        with self.lock:
            count = self.db.execute(
                "SELECT COUNT(*) FROM decisions"
            ).fetchone()[0]

            return {
                "healthy": self.healthy,
                "committed_events": count,
                "last_position": self.last_position,
            }

    def close(self):
        with self.lock:
            self.db.close()
