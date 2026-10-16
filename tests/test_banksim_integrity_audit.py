import copy
from src.banksim_integrity_audit import audit


def fixture():
    models = {}
    for name in ("lr_static", "hgb_static", "hgb_history", "hgb_amount_only"):
        models[name] = {"final": {
            "transactions": 133231, "frauds": 1440, "reviews": 100,
            "true_positives": 50, "false_positives": 50,
            "false_negatives": 1390, "precision": .5,
            "recall": 50 / 1440, "pr_auc_ranking_diagnostic": .2,
            "review_counts_by_step": {
                str(i): (3 if i < 172 else 2) for i in range(144, 180)
            },
            "delayed_label_accounting": {
                "delay_steps": 7, "all_labels_released": True,
                "labels_released_during_stream": 107044,
                "labels_released_after_final_watermark": 26187,
                "true_positives_after_reporting_delay": 50,
                "max_pending_labels": 26206,
            },
        }}
    lock = {"dataset_sha256": "data", "evaluator_script_sha256": "code"}
    report = {"dataset_sha256": "data", "evaluator_script_sha256": "code",
        "train_steps": [0, 107], "tuning_steps": [108, 143],
        "final_steps": [144, 179], "block_size": 10000,
        "max_review_fraction": .01, "fraud_label_reporting_delay_steps": 7,
        "models": models,
        "primary_comparison_hgb_history_minus_hgb_static": {
            "true_positive_difference": 0, "reviews_difference": 0,
            "pr_auc_difference": 0,
        }}
    return lock, report, {"dataset_sha256": "data"}


def test_valid_archived_aggregate():
    args = fixture()
    assert audit(*args)["status"] == "PASS"


def test_reject_wrong_hash():
    lock, report, start = fixture()
    lock["dataset_sha256"] = "wrong"
    assert audit(lock, report, start)["status"] == "FAIL"


def test_reject_inconsistent_decision_accounting():
    lock, report, start = fixture()
    report["models"]["hgb_history"]["final"]["false_positives"] = 49
    assert audit(lock, report, start)["status"] == "FAIL"


def test_reject_inconsistent_review_count():
    lock, report, start = fixture()
    report["models"]["hgb_static"]["final"]["review_counts_by_step"]["144"] = 9
    assert audit(lock, report, start)["status"] == "FAIL"


def test_reject_inconsistent_delayed_labels():
    lock, report, start = fixture()
    report["models"]["lr_static"]["final"]["delayed_label_accounting"]["all_labels_released"] = False
    assert audit(lock, report, start)["status"] == "FAIL"
