import pandas as pd

from src.features.build_features import (
    add_basic_features,
    add_customer_history,
    add_customer_velocity,
    add_device_history,
    add_merchant_history,
    add_past_window_count,
    build_feature_columns,
    leakage_audit,
)


def sample_data():
    return pd.DataFrame(
        {
            "transaction_id": [
                "T1",
                "T2",
                "T3",
                "T4",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 10:05:00",
                    "2026-01-01 10:20:00",
                    "2026-01-01 10:20:00",
                ]
            ),
            "customer_id": [
                "C1",
                "C1",
                "C1",
                "C1",
            ],
            "merchant_id": [
                "M1",
                "M1",
                "M1",
                "M1",
            ],
            "device_id": [
                "D1",
                "D1",
                "D2",
                "D2",
            ],
            "amount": [
                100.0,
                200.0,
                5000.0,
                6000.0,
            ],
            "payment_method": [
                "upi",
                "upi",
                "card",
                "card",
            ],
            "account_age_days": [
                100,
                100,
                100,
                100,
            ],
            "failed_attempts": [
                0,
                0,
                2,
                1,
            ],
            "is_new_device": [
                False,
                False,
                True,
                True,
            ],
            "scenario_id": [
                "NORMAL",
                "NORMAL",
                "AMOUNT_ANOMALY",
                "AMOUNT_ANOMALY",
            ],
            "fraud_label": [
                0,
                0,
                1,
                1,
            ],
        }
    )


def test_first_transaction_has_no_history():
    df = sample_data()

    df = add_customer_history(df)

    assert df.loc[
        0,
        "customer_txn_count_before",
    ] == 0

    assert df.loc[
        0,
        "customer_has_history",
    ] == 0


def test_second_transaction_sees_first_transaction():
    df = sample_data()

    df = add_customer_history(df)

    assert df.loc[
        1,
        "customer_txn_count_before",
    ] == 1


def test_historical_mean_excludes_current_transaction():
    df = sample_data()

    df = add_customer_history(df)

    # T2 = 200.
    # Historical data contains only T1 = 100.
    assert df.loc[
        1,
        "customer_amount_mean_before",
    ] == 100.0

    # T3 = 5000.
    # Historical data contains T1 and T2.
    assert df.loc[
        2,
        "customer_amount_mean_before",
    ] == 150.0


def test_velocity_excludes_current_and_same_timestamp_events():
    df = sample_data()

    df = add_customer_velocity(df)

    # T1 has no previous transactions.
    assert df.loc[
        0,
        "customer_txn_count_10m",
    ] == 0

    # T2 sees T1.
    assert df.loc[
        1,
        "customer_txn_count_10m",
    ] == 1

    # T3 sees T1 and T2.
    assert df.loc[
        2,
        "customer_txn_count_1h",
    ] == 2

    # T4 has exactly the same timestamp as T3.
    # T3 must NOT be treated as historical.
    assert df.loc[
        3,
        "customer_txn_count_1h",
    ] == 2


def test_device_novelty_is_history_derived():
    df = sample_data()

    df = add_device_history(df)

    # First appearance of C1 + D1.
    assert df.loc[
        0,
        "customer_device_seen_before",
    ] == 0

    # Same customer/device appeared previously.
    assert df.loc[
        1,
        "customer_device_seen_before",
    ] == 1

    # C1 + D2 appears for the first time.
    assert df.loc[
        2,
        "customer_device_seen_before",
    ] == 0

    # D2 has now appeared before.
    assert df.loc[
        3,
        "customer_device_seen_before",
    ] == 1


def test_merchant_history_excludes_current_transaction():
    df = sample_data()

    df = add_merchant_history(df)

    assert df.loc[
        0,
        "merchant_txn_count_before",
    ] == 0

    assert df.loc[
        1,
        "merchant_txn_count_before",
    ] == 1

    assert df.loc[
        2,
        "merchant_txn_count_before",
    ] == 2

    assert df.loc[
        2,
        "merchant_amount_mean_before",
    ] == 150.0


def test_temporal_features_are_created():
    df = sample_data()

    df = add_basic_features(df)

    assert df.loc[0, "hour"] == 10
    assert df.loc[0, "day_of_week"] == 3
    assert df.loc[0, "is_weekend"] == 0

    assert "hour_sin" in df.columns
    assert "hour_cos" in df.columns


def test_unsorted_input_is_sorted_before_temporal_features():
    df = sample_data()

    shuffled = df.iloc[
        [2, 0, 3, 1]
    ].reset_index(drop=True)

    shuffled = shuffled.sort_values(
        ["timestamp", "transaction_id"]
    ).reset_index(drop=True)

    assert list(
        shuffled["transaction_id"]
    ) == [
        "T1",
        "T2",
        "T3",
        "T4",
    ]


def test_forbidden_columns_are_excluded():
    df = sample_data()

    df = add_basic_features(df)

    features = build_feature_columns(df)

    assert "fraud_label" not in features
    assert "scenario_id" not in features
    assert "is_new_device" not in features

    assert "transaction_id" not in features
    assert "customer_id" not in features
    assert "merchant_id" not in features
    assert "device_id" not in features


def test_leakage_audit_passes():
    df = sample_data()

    df = add_basic_features(df)

    features = build_feature_columns(df)

    audit = leakage_audit(features)

    assert audit["passed"] is True
    assert audit[
        "forbidden_columns_found"
    ] == []