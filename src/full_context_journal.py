"""Experimental durable journal for the frozen 21-column full-context schema.

Do not connect this module to the existing Flask service or frozen Model B.
A caller must provide a separately calibrated score function and a validated
policy function. Automatic BLOCK stays disabled until a later signed-off gate.
"""
import hashlib
import json
import sqlite3
import threading
from math import isfinite
from pathlib import Path

import numpy as np

from .decision_engine import normalize_transaction, payload_hash
from .full_context import FULL_FEATURE_NAMES, FullContextHistory
from .full_context_evidence import canonical_evidence_json, build_evidence
from .privacy import AccountPseudonymizer, verify_database_mode

SCHEMA_ID = "fgz_full_context_evidence_v2"
FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(FULL_FEATURE_NAMES, separators=(",", ":")).encode()
).hexdigest()

DECISION_COLUMNS = {
    "event_id", "payload_hash", "step", "source_row_index", "originator",
    "recipient", "txn_type", "amount", "sender_balance", "features_json",
    "calibrated_risk", "decision", "feature_schema_sha256", "model_version",
    "policy_version", "evidence_json", "evidence_sha256",
}


class FullContextJournal:
    """Single-owner SQLite feature/decision journal for a NEW database.

    score_calibrated(X) must return a scalar probability from a separately
    calibrated fitted model; choose_action(p) must return ALLOW or REVIEW.
    Neither callable is provided by this module. No production deployment.
    """

    def __init__(
        self, db_path, score_calibrated, choose_action, *,
        model_version, policy_version, privacy_key,
        block_enabled=False,
    ):
        if block_enabled:
            raise RuntimeError(
                "Automatic BLOCK is disabled until conservative policy validation"
            )
        if not callable(score_calibrated) or not callable(choose_action):
            raise TypeError("Calibrated scorer and policy must be callable")
        if any(not isinstance(v, str) or not v.strip()
               for v in (model_version, policy_version)):
            raise ValueError("Model and policy versions must be nonempty")
        self.pseudonymizer = AccountPseudonymizer(privacy_key)
        self.score_calibrated = score_calibrated
        self.choose_action = choose_action
        self.model_version = model_version
        self.policy_version = policy_version
        self.lock = threading.RLock()
        self.healthy = False

        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(
            path, isolation_level=None, check_same_thread=False, timeout=10
        )
        self.db.row_factory = sqlite3.Row
        try:
            # Validate existing DB *before* changing journal mode or schema.
            tables = {
                row["name"]
                for row in self.db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "decisions" in tables and "full_context_meta" not in tables:
                raise RuntimeError(
                    "Refusing existing legacy decisions database. "
                    "Use a NEW full-context SQLite path."
                )
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA busy_timeout=10000")
            self._initialize_schema()
            self.rebuild()
        except Exception:
            self.db.close()
            raise

    def _initialize_schema(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    event_id TEXT PRIMARY KEY,
                    payload_hash TEXT NOT NULL,
                    step INTEGER NOT NULL CHECK(step >= 1),
                    source_row_index INTEGER NOT NULL CHECK(source_row_index >= 0),
                    originator TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    txn_type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    sender_balance REAL NOT NULL,
                    features_json TEXT NOT NULL,
                    calibrated_risk REAL NOT NULL,
                    decision TEXT NOT NULL CHECK(decision IN ('ALLOW','REVIEW')),
                    feature_schema_sha256 TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    evidence_sha256 TEXT NOT NULL,
                    UNIQUE(step, source_row_index)
                )
            """)
            columns = {
                row["name"] for row in self.db.execute("PRAGMA table_info(decisions)")
            }
            if columns != DECISION_COLUMNS:
                raise RuntimeError("Unexpected full-context SQLite decision schema")
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS full_context_meta (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    schema_id TEXT NOT NULL,
                    schema_hash TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    policy_version TEXT NOT NULL
                )
            """)
            meta = self.db.execute(
                "SELECT * FROM full_context_meta WHERE singleton=1"
            ).fetchone()
            required = (
                SCHEMA_ID, FEATURE_SCHEMA_SHA256,
                self.model_version, self.policy_version,
            )
            if meta is None:
                self.db.execute(
                    "INSERT INTO full_context_meta VALUES(1,?,?,?,?)", required
                )
            elif (
                meta["schema_id"], meta["schema_hash"],
                meta["model_version"], meta["policy_version"]
            ) != required:
                raise RuntimeError(
                    "Full-context schema/model/policy mismatch. "
                    "Use a separate database for each frozen artifact set."
                )
            verify_database_mode(self.db, self.pseudonymizer)
            self.db.commit()
        except Exception:
            if self.db.in_transaction:
                self.db.rollback()
            raise

    @staticmethod
    def _row_event(row):
        return {
            "step": row["step"],
            "source_row_index": row["source_row_index"],
            "type": row["txn_type"],
            "amount": row["amount"],
            "oldbalanceOrg": row["sender_balance"],
            "nameOrig": row["originator"],
            "nameDest": row["recipient"],
        }

    def rebuild(self):
        """Recompute past-only state from committed rows, never rescore them."""
        with self.lock:
            self.healthy = False
            state = FullContextHistory()
            position = (-1, -1)
            count = 0
            reviewed = 0
            rows = self.db.execute("""
                SELECT * FROM decisions ORDER BY step, source_row_index
            """)
            for row in rows:
                cursor = (row["step"], row["source_row_index"])
                if cursor <= position:
                    raise RuntimeError("Committed chronology inconsistent")
                if (
                    row["feature_schema_sha256"] != FEATURE_SCHEMA_SHA256
                    or row["model_version"] != self.model_version
                    or row["policy_version"] != self.policy_version
                ):
                    raise RuntimeError("Committed feature/model/policy version mismatch")
                event = self._row_event(row)
                reconstructed = state.features(event)
                saved = json.loads(row["features_json"])
                if list(reconstructed) != saved:
                    raise RuntimeError("Committed feature history mismatch")
                evidence = canonical_evidence_json(reconstructed)
                if (row["evidence_json"] != evidence or
                        row["evidence_sha256"] != hashlib.sha256(evidence.encode()).hexdigest()):
                    raise RuntimeError("Committed evidence integrity mismatch")
                state.observe(event)
                position = cursor
                count += 1
                if row["decision"] == "REVIEW":
                    reviewed += 1
                elif row["decision"] != "ALLOW":
                    raise RuntimeError("Committed invalid decision")
            self.history = state
            self.last_position = position
            self.committed_events = count
            self.reviewed_events = reviewed
            self.healthy = True

    def _insert(self, transaction, fingerprint, features, risk, decision, evidence):
        self.db.execute("""
            INSERT INTO decisions (
                event_id, payload_hash, step, source_row_index,
                originator, recipient, txn_type, amount, sender_balance,
                features_json, calibrated_risk, decision,
                feature_schema_sha256, model_version, policy_version,
                evidence_json, evidence_sha256
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            transaction["event_id"], fingerprint, transaction["step"],
            transaction["source_row_index"], transaction["nameOrig"],
            transaction["nameDest"], transaction["type"], transaction["amount"],
            transaction["oldbalanceOrg"], json.dumps(list(features)),
            risk, decision, FEATURE_SCHEMA_SHA256,
            self.model_version, self.policy_version,
            evidence, hashlib.sha256(evidence.encode()).hexdigest(),
        ))

    @staticmethod
    def _result(event_id, risk, decision, features, duplicate):
        return {
            "event_id": event_id,
            "risk": float(risk),
            "decision": decision,
            "features": dict(zip(FULL_FEATURE_NAMES, features)),
            "duplicate": duplicate,
            "evidence": build_evidence(features),
        }

    def process(self, raw):
        transaction = normalize_transaction(raw)
        transaction["nameOrig"] = self.pseudonymizer.encode(transaction["nameOrig"])
        transaction["nameDest"] = self.pseudonymizer.encode(transaction["nameDest"])
        fingerprint = payload_hash(transaction)
        event_id = transaction["event_id"]

        with self.lock:
            if not self.healthy:
                raise RuntimeError("Full-context state unavailable; rebuild required")
            state_touched = False
            try:
                self.db.execute("BEGIN IMMEDIATE")
                existing = self.db.execute(
                    "SELECT * FROM decisions WHERE event_id=?", (event_id,)
                ).fetchone()
                if existing is not None:
                    if existing["payload_hash"] != fingerprint:
                        self.db.rollback()
                        raise ValueError("Event ID already exists with different data")
                    try:
                        saved = json.loads(existing["features_json"])
                        canonical = canonical_evidence_json(saved)
                        valid = (
                            existing["feature_schema_sha256"] == FEATURE_SCHEMA_SHA256
                            and existing["model_version"] == self.model_version
                            and existing["policy_version"] == self.policy_version
                            and existing["evidence_json"] == canonical
                            and existing["evidence_sha256"] ==
                            hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                            and existing["decision"] in ("ALLOW", "REVIEW")
                            and isfinite(existing["calibrated_risk"])
                            and 0 <= existing["calibrated_risk"] <= 1
                        )
                        if valid:
                            saved_vector = np.asarray(saved, dtype=np.float32)
                            if (saved_vector.shape != (len(FULL_FEATURE_NAMES),)
                                    or not np.isfinite(saved_vector).all()):
                                valid = False
                            else:
                                recomputed_risk = float(self.score_calibrated(
                                    saved_vector.reshape(1, -1)
                                ))
                                valid = (isfinite(recomputed_risk)
                                         and 0 <= recomputed_risk <= 1
                                         and abs(recomputed_risk - existing["calibrated_risk"]) <= 1e-8)
                        if not valid:
                            raise RuntimeError("Committed duplicate integrity mismatch")
                        result = self._result(
                            event_id, existing["calibrated_risk"],
                            existing["decision"], saved, True,
                        )
                    except (ValueError, KeyError, TypeError, OverflowError, RuntimeError) as exc:
                        self.healthy = False
                        raise RuntimeError(
                            "Committed duplicate integrity mismatch; rebuild required"
                        ) from exc
                    self.db.rollback()
                    return result
                latest = self.db.execute("""
                    SELECT step, source_row_index FROM decisions
                    ORDER BY step DESC, source_row_index DESC LIMIT 1
                """).fetchone()
                actual_position = (
                    (latest["step"], latest["source_row_index"])
                    if latest is not None else (-1, -1)
                )
                if actual_position != self.last_position:
                    self.healthy = False
                    raise RuntimeError("Database changed externally; rebuild required")
                position = (
                    transaction["step"], transaction["source_row_index"]
                )
                if position <= self.last_position:
                    raise ValueError("New transactions must be in replay order")
                state_touched = True
                features = self.history.features(transaction)
                risk = float(self.score_calibrated(
                    np.asarray([features], dtype=np.float32)
                ))
                if not isfinite(risk) or not 0 <= risk <= 1:
                    raise ValueError("Calibrated scorer returned invalid risk")
                decision = self.choose_action(risk)
                if decision not in ("ALLOW", "REVIEW"):
                    raise RuntimeError(
                        "Unvalidated BLOCK or other action refused; "
                        "conservative policy not yet authorized"
                    )
                evidence = canonical_evidence_json(features)
                self._insert(transaction, fingerprint, features, risk, decision, evidence)
                self.db.commit()
            except Exception as exc:
                if self.db.in_transaction:
                    self.db.rollback()
                if state_touched or isinstance(exc, sqlite3.Error):
                    self.healthy = False
                raise

            try:
                self.history.observe(transaction)
                self.last_position = position
                self.committed_events += 1
                if decision == "REVIEW":
                    self.reviewed_events += 1
            except Exception as exc:
                self.healthy = False
                raise RuntimeError(
                    "Decision committed but context update failed; rebuild required"
                ) from exc
            return self._result(event_id, risk, decision, features, False)

    def status(self):
        with self.lock:
            return {
                "healthy": self.healthy,
                "committed_events": self.committed_events,
                "reviewed_events": self.reviewed_events,
                "last_position": self.last_position,
                "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                "model_version": self.model_version,
                "policy_version": self.policy_version,
                "block_enabled": False,
            }

    def close(self):
        with self.lock:
            self.db.close()
