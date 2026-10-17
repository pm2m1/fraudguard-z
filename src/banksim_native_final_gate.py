"""Read-only integrity gate before BankSim final holdout is opened.

Does not load or iterate the final-test label column, and does not score rows.
Requires the known frozen BankSim source, model, calibration and policy report.
"""
import argparse
import json
from pathlib import Path

from .banksim_native_fit import FIT_OUTPUT, checksum, split_boundaries
from .banksim_native_prepare import OUTPUT, SOURCE_SHA256, SCHEMA_SHA256

DIAGNOSTIC = Path("reports/banksim_native_policy_diagnostic_v1.json")

PINS = {
    "model_sha256": "6ca486570adc58ded65ab916e01b68f802604ac0e8cd706aa9c4bd24778b80f1",
    "calibrator_sha256": "74fe5994b3e9d09ca5e1cdf7865cc3731077300b7655217fbbeb7dd004388e8b",
    "prepared_metadata_sha256": "43933a94f661b614c02f5ee1e140a7655898f4ac97b01e3777ffbe451369513d",
    "prepared_features_sha256": "edd0983d1399beadf2a9aa28286fe3c66ac982b28f7a6e8713947d942d097be3",
    "prepared_development_labels_sha256": "ca047ecbaf04766fdee19d7f8eb1c84be6016917c769f5f45781544e19e6d9b6",
}
FILENAMES = {
    "model_sha256": "model.joblib",
    "calibrator_sha256": "calibrator.joblib",
    "prepared_metadata_sha256": "materialization.json",
    "prepared_features_sha256": "features.npy",
    "prepared_development_labels_sha256": "development_labels.npy",
}
EXPECTED_POLICY = {
    "rows": 66259,
    "frauds": 720,
    "reviewed": 607,
    "would_block": 0,
    "false_blocks": 0,
}
THRESHOLD = 0.41746818058367946


def _read_json_file(path):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Missing or symlinked immutable input: " + path.name)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify(prepared=OUTPUT, fit_dir=FIT_OUTPUT, diagnostic=DIAGNOSTIC):
    prepared, fit_dir, diagnostic = (
        Path(prepared), Path(fit_dir), Path(diagnostic)
    )
    # First verify bytes, then parse manifests; never open any CSV rows.
    for key, expected in PINS.items():
        directory = fit_dir if key in (
            "model_sha256", "calibrator_sha256"
        ) else prepared
        path = directory / FILENAMES[key]
        if path.is_symlink() or not path.is_file() or checksum(path) != expected:
            raise RuntimeError("Frozen file digest mismatch: " + key)

    metadata = _read_json_file(prepared / "materialization.json")
    bounds = split_boundaries(metadata)
    if bounds != [0, 332592, 396332, 461412, 527671, 594643]:
        raise RuntimeError("Frozen partition boundary mismatch")

    fit = _read_json_file(fit_dir / "fit.json")
    if any(fit.get(key) != digest for key, digest in PINS.items()):
        raise RuntimeError("Fitting manifest digests do not match frozen pins")
    if not (
        fit.get("status") == "BANKSIM_FIT_FROZEN_NOT_EVALUATED"
        and fit.get("prepared_source_sha256") == SOURCE_SHA256
        and fit.get("feature_schema_sha256") == SCHEMA_SHA256
        and fit.get("selected") == "behavioral"
        and fit.get("selected_columns") == 24
        and fit.get("review_threshold") == THRESHOLD
        and fit.get("calibration_target_review_fraction") == 0.01
        and fit.get("calibration_slope") == 0.8271273942690252
        and fit.get("calibration_intercept") == -3.4818713337475757
        and fit.get("block_enabled") is False
        and fit.get("policy_labels_accessed") is False
        and fit.get("final_labels_accessed") is False
        and fit.get("partitions") == metadata["partition_rows"]
    ):
        raise RuntimeError("Fitted model, calibration or policy differs from freeze")

    report = _read_json_file(diagnostic)
    if not (
        report.get("status") == "BANKSIM_POLICY_DIAGNOSTIC_FROZEN_NO_FINAL_LABELS"
        and report.get("frozen_sha256") == PINS
        and report.get("policy_steps") == [144, 161]
        and report.get("model") == "behavioral"
        and report.get("feature_count") == 24
        and report.get("review_capacity") == "prefix_floor_1_percent"
        and report.get("review_threshold") == THRESHOLD
        and report.get("final_labels_accessed") is False
        and report.get("paySim_model_changed") is False
    ):
        raise RuntimeError("Diagnostic report does not match frozen experiment")
    m = report.get("metrics")
    if not isinstance(m, dict) or any(
        m.get(key) != value for key, value in EXPECTED_POLICY.items()
    ):
        raise RuntimeError("Diagnostic policy totals changed")
    if (
        m.get("precision_intervention") != 0.8253706754530478
        or m.get("recall_intervention") != 0.6958333333333333
        or m.get("pr_auc") != 0.8403405396166734
    ):
        raise RuntimeError("Diagnostic quality metrics changed")
    cats = m.get("by_type")
    if not isinstance(cats, dict) or len(cats) != 15:
        raise RuntimeError("Diagnostic category breakdown invalid")
    totals = {
        "rows": sum(x["rows"] for x in cats.values()),
        "frauds": sum(x["frauds"] for x in cats.values()),
        "reviewed": sum(x["interventions"] for x in cats.values()),
    }
    if totals != {k: EXPECTED_POLICY[k] for k in totals}:
        raise RuntimeError("Diagnostic category totals inconsistent")
    return {
        "status": "BANKSIM_FINAL_INTEGRITY_GATE_PASSED",
        "source_sha256": SOURCE_SHA256,
        "feature_schema_sha256": SCHEMA_SHA256,
        "model_sha256": PINS["model_sha256"],
        "calibrator_sha256": PINS["calibrator_sha256"],
        "policy_diagnostic_rows": EXPECTED_POLICY["rows"],
        "holdout_rows": bounds[5] - bounds[4],
        "holdout_labels_accessed": False,
        "final_evaluation_executed": False,
        "warning": "This gate does not validate the frozen PaySim model or authorize BLOCK.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=OUTPUT)
    parser.add_argument("--fit", type=Path, default=FIT_OUTPUT)
    parser.add_argument("--diagnostic", type=Path, default=DIAGNOSTIC)
    args = parser.parse_args()
    print(json.dumps(verify(args.prepared, args.fit, args.diagnostic), indent=2))


if __name__ == "__main__":
    main()
