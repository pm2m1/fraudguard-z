import hashlib
import json
import os
from pathlib import Path

import joblib

from .api import create_app
from .decision_engine import DecisionEngine
from .features import MODEL_B_FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load_model(filename):
    checksum_file = REPORTS / "model_sha256.txt"
    checksums = {}

    for line in checksum_file.read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        checksums[path.lstrip("*")] = digest

    relative_path = f"models/{filename}"
    model_path = ROOT / relative_path

    if relative_path not in checksums:
        raise RuntimeError(f"Missing checksum for {filename}")

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()

    if actual != checksums[relative_path]:
        raise RuntimeError(f"Checksum mismatch for {filename}")

    return joblib.load(model_path)


def main():
    deployment_mode = os.environ.get("FGZ_DEPLOYMENT_MODE", "local")
    if deployment_mode not in ("local", "production"):
        raise RuntimeError("FGZ_DEPLOYMENT_MODE must be local or production")
    privacy_key = os.environ.get("FGZ_PRIVACY_KEY")
    if deployment_mode == "production":
        if not os.environ.get("FGZ_AUTH_PASSWORD"):
            raise RuntimeError("Production mode requires FGZ_AUTH_PASSWORD")
        if not privacy_key:
            raise RuntimeError("Production mode requires FGZ_PRIVACY_KEY")
        raise RuntimeError("Production mode cannot use the Flask development server; deploy a single-worker authenticated TLS-protected WSGI service instead")
    policy_path = REPORTS / "review_policy.json"

    with policy_path.open() as file:
        policy = json.load(file)

    with (REPORTS / "split_manifest.json").open() as file:
        manifest = json.load(file)

    with (REPORTS / "final_test_results.json").open() as file:
        final_report = json.load(file)

    policy_hash = hashlib.sha256(
        policy_path.read_bytes()
    ).hexdigest()

    if policy_hash != final_report["policy_sha256"]:
        raise RuntimeError("Review policy does not match final evaluation")

    if policy["selected_model"] != "B":
        raise RuntimeError("Expected Model B")

    if policy["block_enabled"]:
        raise RuntimeError("Automatic blocking must remain disabled")

    if manifest["schema_b"] != list(MODEL_B_FEATURE_NAMES):
        raise RuntimeError("Model feature schema mismatch")

    model = load_model("model_b.joblib")
    calibrator = load_model("calibrator_b.joblib")

    if model.n_features_in_ != 8:
        raise RuntimeError("Incorrect model input size")

    if calibrator.n_features_in_ != 1:
        raise RuntimeError("Incorrect calibrator input size")

    db_override = os.environ.get("FGZ_DB_PATH")
    db_path = Path(db_override).expanduser() if db_override else (
        ROOT / "data/runtime/fraudguard.sqlite3"
    )

    engine = DecisionEngine(
        db_path,
        model,
        calibrator,
        float(policy["review_threshold"]),
        privacy_key=privacy_key,
    )

    app = create_app(engine)

    try:
        print("FraudGuard Z listening on http://127.0.0.1:5050")
        print(f"SQLite database: {db_path}")

        app.run(
            host="127.0.0.1",
            port=5050,
            debug=False,
            use_reloader=False,
            threaded=False,
        )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
