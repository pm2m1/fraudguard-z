# Account privacy and deployment guardrails (experimental)

This branch does **not** modify existing SQLite files automatically or migrate prior records. It adds opt-in pseudonymization for a **new** database. Existing local demos still work without a privacy key.

## Run the privacy demo locally

1. Stop the current Flask instance with Ctrl+C.
2. In the same terminal, run:

```bash
cd ~/IBM-Z-Datathon-2026
source .venv/bin/activate
python -m pytest tests -q
read -s "FGZ_AUTH_PASSWORD?Demo password (16+ characters): "
echo
export FGZ_AUTH_PASSWORD
read -s "FGZ_PRIVACY_KEY?New private database key (32+ characters): "
echo
export FGZ_PRIVACY_KEY
FGZ_DB_PATH=data/runtime/private_demo_v1.sqlite3 python -m src.serve
```

Keep the key private. It must be saved in a secure secret manager or password manager to recover this database after a restart. Losing the key means the existing recipient history cannot be used to score new events. Do not change the key or start this mode against an earlier plaintext `.sqlite3` file.

The same account gets the same keyed HMAC-SHA256 pseudonym across transactions, so recipient-count history survives restarts. Sender and recipient account IDs are stored as `acct_v1_<hex>` values. The payload fingerprint hashes the pseudonymized normalized transaction. Duplicate events, replay order, and feature calculations remain as before.

## Important limits

- HMAC pseudonymization is **not encryption, anonymization, or a full banking privacy program**. Risk scores, amounts, transaction type, account linkability, event IDs, and timestamps remain in the database. Payload and access logs require further review. SQLite itself is not encrypted by this change.
- This only changes the persisted IDs for **new** pseudonymous databases. Do not submit sensitive real customer data to the demo.
- The `FGZ_DEPLOYMENT_MODE=production` setting deliberately refuses to launch through `python -m src.serve`; the Flask development server is not permitted as a production service. A dedicated single-worker WSGI entry point, TLS reverse proxy, and infrastructure security validation must be built and verified before production deployment.
- `FGZ_PRIVACY_KEY` is a process-wide secret. Do not add it to shell-history commands, version control, screenshots, or logs.
- This branch does not assert that the full production security gap is closed.

## Validation

Run `python -m pytest tests -q`. Then confirm an empty private DB starts; score two requests with the same recipient; restart using the same key and observe that historical N3 persists. Try a different key on the same DB and verify startup fails closed.
