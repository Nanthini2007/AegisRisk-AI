import pandas as pd

from src.validation.validate_dataset import (
    data_quality_checks,
    fraud_analysis,
    leakage_audit,
    schema_checks,
    temporal_analysis,
)


def create_valid_dataframe() -> pd.DataFrame:
    """Create a small valid dataset for validation tests."""

    return pd.DataFrame(
        {
            "transaction_id": [
                "txn_001",
                "txn_002",
                "txn_003",
            ],
            "timestamp": [
                "2026-01-01 10:00:00",
                "2026-01-01 11:00:00",
                "2026-01-01 12:00:00",
            ],
            "customer_id": [
                "cust_001",
                "cust_001",
                "cust_002",
            ],
            "merchant_id": [
                "merchant_001",
                "merchant_001",
                "merchant_002",
            ],
            "device_id": [
                "device_001",
                "device_001",
                "device_002",
            ],
            "amount": [
                500.0,
                750.0,
                1000.0,
            ],
            "payment_method": [
                "upi",
                "card",
                "upi",
            ],
            "account_age_days": [
                100,
                100,
                50,
            ],
            "failed_attempts": [
                0,
                1,
                0,
            ],
            "is_new_device": [
                0,
                0,
                1,
            ],
            "scenario_id": [
                "normal",
                "normal",
                "new_device",
            ],
            "fraud_label": [
                0,
                0,
                1,
            ],
        }
    )


def prepare_dataframe() -> pd.DataFrame:
    """Convert timestamps like the real dataset loader."""

    df = create_valid_dataframe()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_valid_schema_passes():
    df = prepare_dataframe()

    result = schema_checks(df)

    assert result["passed"] is True
    assert result["missing_required_columns"] == []
    assert result["unexpected_columns"] == []
    assert result["duplicate_transaction_ids"] == 0


def test_missing_required_column_is_detected():
    df = prepare_dataframe().drop(
        columns=["amount"]
    )

    result = schema_checks(df)

    assert result["passed"] is False
    assert "amount" in result["missing_required_columns"]


def test_unexpected_column_is_detected():
    df = prepare_dataframe()
    df["unexpected_feature"] = 123

    result = schema_checks(df)

    assert "unexpected_feature" in result[
        "unexpected_columns"
    ]


def test_duplicate_transaction_id_is_detected():
    df = prepare_dataframe()
    df.loc[1, "transaction_id"] = "txn_001"

    result = schema_checks(df)

    assert result["duplicate_transaction_ids"] == 1
    assert result["passed"] is False


def test_invalid_amount_is_detected():
    df = prepare_dataframe()
    df.loc[0, "amount"] = -100.0

    result = data_quality_checks(df)

    assert result["invalid_amounts"] == 1
    assert result["passed"] is False


def test_invalid_fraud_label_is_detected():
    df = prepare_dataframe()
    df.loc[0, "fraud_label"] = 2

    result = data_quality_checks(df)

    assert result["invalid_fraud_labels"] == 1
    assert result["passed"] is False


def test_duplicate_rows_are_detected():
    df = prepare_dataframe()
    df = pd.concat(
        [df, df.iloc[[0]]],
        ignore_index=True,
    )

    result = data_quality_checks(df)

    assert result["duplicate_rows"] == 1


def test_temporal_checks_pass_for_sorted_data():
    df = prepare_dataframe()

    result = temporal_analysis(df)

    assert result["passed"] is True
    assert result["invalid_timestamps"] == 0
    assert result["chronologically_sorted"] is True


def test_fraud_analysis_counts_correctly():
    df = prepare_dataframe()

    result = fraud_analysis(df)

    assert result["fraud_count"] == 1
    assert result["legitimate_count"] == 2


def test_leakage_audit_excludes_required_columns():
    df = prepare_dataframe()

    result = leakage_audit(df)

    assert "fraud_label" in result[
        "excluded_from_ml_features"
    ]
    assert "scenario_id" in result[
        "excluded_from_ml_features"
    ]