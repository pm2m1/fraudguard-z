"""Deterministic, keyed account pseudonyms for new FraudGuard SQLite databases.

This is pseudonymization, NOT encryption or anonymization. The key must
be securely generated, kept out of Git, and backed up for DB recovery.
"""
import hashlib
import hmac

MODE_HMAC = "hmac-sha256-v1"
MODE_PLAIN = "plain-v1"

class AccountPseudonymizer:
    def __init__(self, key):
        if not isinstance(key, str) or len(key.encode("utf-8")) < 32:
            raise ValueError("FGZ_PRIVACY_KEY must contain at least 32 UTF-8 bytes")
        self._key = key.encode("utf-8")

    def encode(self, identifier):
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Account ID cannot be blank")
        digest = hmac.new(
            self._key, b"account-v1\x00" + identifier.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return "acct_v1_" + digest

    @property
    def verifier(self):
        return hmac.new(
            self._key, b"fraudguard-sqlite-key-verifier-v1",
            hashlib.sha256
        ).hexdigest()


def verify_database_mode(db, pseudonymizer):
    """Prevent accidental mixing of raw and pseudonymized account IDs."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS privacy_config (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            mode TEXT NOT NULL,
            key_verifier TEXT NOT NULL
        )
    """)
    expected_mode = MODE_HMAC if pseudonymizer else MODE_PLAIN
    expected_verifier = pseudonymizer.verifier if pseudonymizer else ""
    saved = db.execute(
        "SELECT mode, key_verifier FROM privacy_config WHERE singleton=1"
    ).fetchone()

    if saved is not None:
        if (saved["mode"] != expected_mode
                or not hmac.compare_digest(saved["key_verifier"], expected_verifier)):
            raise RuntimeError(
                "SQLite privacy mode or key mismatch. Use the original key "
                "and a separate database for each mode. No migration was performed."
            )
        return

    existing = db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    if existing and pseudonymizer:
        raise RuntimeError(
            "Refusing to enable pseudonymization on an existing legacy database. "
            "Create a NEW SQLite database to avoid mixing account identifiers."
        )
    db.execute(
        "INSERT INTO privacy_config(singleton, mode, key_verifier) VALUES (1, ?, ?)",
        (expected_mode, expected_verifier),
    )
