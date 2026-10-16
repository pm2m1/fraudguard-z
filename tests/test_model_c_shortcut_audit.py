import csv
from src.model_c_shortcut_audit import exact_positive_balance_match, audit


def test_exact_drain_matches():
    assert exact_positive_balance_match("10.00", "10.00")


def test_zero_balance_is_excluded():
    assert not exact_positive_balance_match("0.00", "0.00")


def test_approximate_drain_is_not_exact():
    assert not exact_positive_balance_match("9.50", "10.00")


def test_audit_respects_split_and_does_not_read_later_rows(tmp_path):
    path = tmp_path / "mini.csv"
    rows = [
        [238, "TRANSFER", "10.0", "10.0", 1],
        [239, "CASH_OUT", "10.0", "10.0", 1],
        [306, "CASH_OUT", "11.0", "10.0", 0],
        [307, "TRANSFER", "10.0", "10.0", 0],
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "type", "amount", "oldbalanceOrg", "isFraud"])
        w.writerows(rows)
    result = audit(path)
    assert result["train"]["ALL"]["matching_frauds"] == 1
    assert result["validation"]["ALL"]["transactions"] == 2
    assert result["validation"]["ALL"]["matching_legitimate"] == 0
