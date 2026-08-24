"""
AegisRisk AI
Day 4 - Leakage-Safe Temporal Feature Engineering

All historical features must use information available
before the current transaction.

This module does NOT train any ML model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/raw/transactions.csv")
OUTPUT_PATH = Path(
    "data/processed/transactions_features.csv"
)
REPORT_PATH = Path(
    "reports/day4_feature_report.json"
)

TARGET_COLUMN = "fraud_label"

# These columns are retained in the processed dataset for
# auditing/traceability but are NEVER model features.
EXCLUDED_COLUMNS = {
    "transaction_id",
    "customer_id",
    "merchant_id",
    "device_id",
    "fraud_label",
    "scenario_id",
    "is_new_device",
}

TEMPORAL_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
]

CUSTOMER_FEATURES = [
    "customer_txn_count_before",
    "customer_amount_mean_before",
    "customer_amount_std_before",
    "amount_vs_customer_mean",
    "customer_amount_zscore",
    "customer_seconds_since_prev",
    "customer_has_history",
]

VELOCITY_FEATURES = [
    "customer_txn_count_10m",
    "customer_txn_count_1h",
    "customer_txn_count_24h",
]

MERCHANT_FEATURES = [
    "merchant_txn_count_before",
    "merchant_amount_mean_before",
    "merchant_seconds_since_prev",
    "merchant_txn_count_1h",
]

DEVICE_FEATURES = [
    "customer_device_seen_before",
    "device_seen_before",
]


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw transaction dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    required = {
        "transaction_id",
        "timestamp",
        "customer_id",
        "merchant_id",
        "device_id",
        "amount",
        "payment_method",
        "account_age_days",
        "failed_attempts",
        "is_new_device",
        "scenario_id",
        "fraud_label",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamps detected."
        )

    if df["transaction_id"].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs detected."
        )

    # Deterministic ordering.
    df = (
        df.sort_values(
            ["timestamp", "transaction_id"]
        )
        .reset_index(drop=True)
    )

    return df


def add_basic_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create transaction and calendar features."""

    df = df.copy()

    df["amount_log"] = np.log1p(df["amount"])

    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    hour = (
        df["hour"]
        + df["timestamp"].dt.minute / 60.0
        + df["timestamp"].dt.second / 3600.0
    )

    df["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    return df


def add_customer_history(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create leakage-safe customer history features.

    Only transactions appearing BEFORE the current transaction
    contribute to historical statistics.
    """

    df = df.copy()

    grouped = df.groupby(
        "customer_id",
        sort=False,
    )

    # Number of previous customer transactions.
    df["customer_txn_count_before"] = (
        grouped.cumcount()
    )

    prior_count = (
        df["customer_txn_count_before"]
    )

    # Cumulative amount includes current transaction,
    # so subtract the current amount to obtain prior amount.
    cumulative_amount = (
        grouped["amount"].cumsum()
    )

    prior_amount_sum = (
        cumulative_amount - df["amount"]
    )

    df["customer_amount_mean_before"] = (
        prior_amount_sum
        / prior_count.replace(0, np.nan)
    )

    # Historical variance.
    cumulative_squared = (
        grouped["amount"]
        .transform(
            lambda s: s.pow(2).cumsum()
        )
    )

    prior_squared_sum = (
        cumulative_squared
        - df["amount"].pow(2)
    )

    prior_mean = (
        df["customer_amount_mean_before"]
    )

    prior_variance = (
        prior_squared_sum
        / prior_count.replace(0, np.nan)
        - prior_mean.pow(2)
    )

    prior_variance = prior_variance.clip(
        lower=0
    )

    df["customer_amount_std_before"] = (
        np.sqrt(prior_variance)
    )

    df["amount_vs_customer_mean"] = (
        df["amount"]
        / df["customer_amount_mean_before"]
    )

    df["customer_amount_zscore"] = (
        (
            df["amount"]
            - df["customer_amount_mean_before"]
        )
        / df["customer_amount_std_before"]
    )

    # Time since previous customer transaction.
    df["customer_seconds_since_prev"] = (
        grouped["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    df["customer_has_history"] = (
        prior_count > 0
    ).astype(int)

    return df


def add_past_window_count(
    df: pd.DataFrame,
    group_column: str,
    seconds: int,
    feature_name: str,
) -> pd.DataFrame:
    """
    Count historical events in a time window.

    IMPORTANT:
    - Current transaction is excluded.
    - ALL transactions with the same timestamp are excluded.
    - Only strictly earlier timestamps are considered historical.

    This prevents same-timestamp information leakage.
    """

    result = pd.Series(
        0,
        index=df.index,
        dtype="int64",
    )

    window = pd.Timedelta(
        seconds=seconds
    )

    for _, group_indices in df.groupby(
        group_column,
        sort=False,
    ).groups.items():

        group = (
            df.loc[group_indices]
            .sort_values(
                ["timestamp", "transaction_id"]
            )
        )

        timestamps = (
            group["timestamp"]
            .to_numpy()
        )

        # For each transaction timestamp, find the first
        # timestamp belonging to the historical window.
        left_positions = np.searchsorted(
            timestamps,
            timestamps - window.to_numpy(),
            side="left",
        )

        # Find the first occurrence of the current timestamp.
        # Everything from this position onward has timestamp
        # equal to the current timestamp and is excluded.
        current_timestamp_positions = (
            np.searchsorted(
                timestamps,
                timestamps,
                side="left",
            )
        )

        counts = (
            current_timestamp_positions
            - left_positions
        )

        result.loc[group.index] = counts

    df[feature_name] = result

    return df


def add_customer_velocity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create customer transaction velocity features."""

    df = add_past_window_count(
        df,
        "customer_id",
        10 * 60,
        "customer_txn_count_10m",
    )

    df = add_past_window_count(
        df,
        "customer_id",
        60 * 60,
        "customer_txn_count_1h",
    )

    df = add_past_window_count(
        df,
        "customer_id",
        24 * 60 * 60,
        "customer_txn_count_24h",
    )

    return df


def add_merchant_history(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create leakage-safe merchant history features."""

    df = df.copy()

    grouped = df.groupby(
        "merchant_id",
        sort=False,
    )

    df["merchant_txn_count_before"] = (
        grouped.cumcount()
    )

    cumulative_amount = (
        grouped["amount"].cumsum()
    )

    prior_sum = (
        cumulative_amount - df["amount"]
    )

    prior_count = (
        df["merchant_txn_count_before"]
    )

    df["merchant_amount_mean_before"] = (
        prior_sum
        / prior_count.replace(0, np.nan)
    )

    df["merchant_seconds_since_prev"] = (
        grouped["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    df = add_past_window_count(
        df,
        "merchant_id",
        60 * 60,
        "merchant_txn_count_1h",
    )

    return df


def add_device_history(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive device novelty from historical transactions.

    The simulator's is_new_device flag is deliberately NOT used.
    """

    df = df.copy()

    customer_device_count = (
        df.groupby(
            ["customer_id", "device_id"],
            sort=False,
        )
        .cumcount()
    )

    global_device_count = (
        df.groupby(
            "device_id",
            sort=False,
        )
        .cumcount()
    )

    df["customer_device_seen_before"] = (
        customer_device_count > 0
    ).astype(int)

    df["device_seen_before"] = (
        global_device_count > 0
    ).astype(int)

    return df


def clean_feature_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean undefined numerical values.

    Missing historical statistics mean that there was
    insufficient prior history.
    """

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    history_columns = [
        "customer_amount_mean_before",
        "customer_amount_std_before",
        "amount_vs_customer_mean",
        "customer_amount_zscore",
        "customer_seconds_since_prev",
        "merchant_amount_mean_before",
        "merchant_seconds_since_prev",
    ]

    for column in history_columns:
        if column in df.columns:
            df[column] = (
                df[column].fillna(0)
            )

    return df


def build_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """Return columns allowed as model inputs."""

    return [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]


def leakage_audit(
    feature_columns: list[str],
) -> dict:
    """Verify forbidden columns are absent from model inputs."""

    forbidden = (
        EXCLUDED_COLUMNS
        & set(feature_columns)
    )

    return {
        "forbidden_columns_found": sorted(
            forbidden
        ),
        "passed": len(forbidden) == 0,
    }


def quality_checks(
    original: pd.DataFrame,
    processed: pd.DataFrame,
) -> dict:
    """Run Day 4 dataset integrity checks."""

    numeric_columns = processed.select_dtypes(
        include=[np.number]
    ).columns

    missing_values = (
        processed.isna()
        .sum()
    )

    infinite_values = {}

    for column in numeric_columns:
        count = int(
            np.isinf(
                processed[column].to_numpy()
            ).sum()
        )

        if count > 0:
            infinite_values[column] = count

    timestamp_unchanged = (
        original["timestamp"].reset_index(drop=True)
        .equals(
            processed["timestamp"]
            .reset_index(drop=True)
        )
    )

    transaction_ids_unchanged = (
        original["transaction_id"]
        .reset_index(drop=True)
        .equals(
            processed["transaction_id"]
            .reset_index(drop=True)
        )
    )

    return {
        "input_rows": int(len(original)),
        "output_rows": int(len(processed)),
        "row_count_unchanged": (
            len(original) == len(processed)
        ),
        "timestamp_unchanged": (
            timestamp_unchanged
        ),
        "transaction_ids_unchanged": (
            transaction_ids_unchanged
        ),
        "transaction_ids_unique": (
            processed["transaction_id"]
            .is_unique
        ),
        "missing_values": {
            column: int(count)
            for column, count
            in missing_values.items()
            if count > 0
        },
        "infinite_values": infinite_values,
    }


def build_report(
    original: pd.DataFrame,
    processed: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    """Create the Day 4 machine-readable report."""

    audit = leakage_audit(
        feature_columns
    )

    quality = quality_checks(
        original,
        processed,
    )

    warnings = []

    if quality["missing_values"]:
        warnings.append(
            "Missing values remain in the processed dataset."
        )

    if quality["infinite_values"]:
        warnings.append(
            "Infinite values remain in the processed dataset."
        )

    if not quality["row_count_unchanged"]:
        warnings.append(
            "Input and output row counts differ."
        )

    if not quality["timestamp_unchanged"]:
        warnings.append(
            "Timestamps changed during feature engineering."
        )

    if not quality["transaction_ids_unchanged"]:
        warnings.append(
            "Transaction IDs changed during feature engineering."
        )

    limitations = [
        "Historical features depend on the available transaction history.",
        "First-ever events have no prior behavioral history.",
        "Payment method remains categorical and may require encoding during Day 5.",
        "Historical features are computed from the synthetic transaction stream.",
        "No fraud-label-derived historical features are created.",
        "No ML model is trained during Day 4.",
    ]

    all_quality_checks_pass = (
        quality["row_count_unchanged"]
        and quality["timestamp_unchanged"]
        and quality["transaction_ids_unchanged"]
        and quality["transaction_ids_unique"]
        and not quality["missing_values"]
        and not quality["infinite_values"]
    )

    verdict = (
        "PASS"
        if audit["passed"]
        and all_quality_checks_pass
        else "REVISE"
    )

    return {
        "project": "AegisRisk AI",
        "stage": (
            "Day 4 - Leakage-Safe "
            "Temporal Feature Engineering"
        ),
        "synthetic_data_notice": (
            "The dataset is synthetic and must not be "
            "represented as real Razorpay production data."
        ),
        "input_rows": int(len(original)),
        "output_rows": int(len(processed)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "excluded_columns": sorted(
            EXCLUDED_COLUMNS
        ),
        "target_column": TARGET_COLUMN,
        "temporal_features": TEMPORAL_FEATURES,
        "customer_features": CUSTOMER_FEATURES,
        "velocity_features": VELOCITY_FEATURES,
        "merchant_features": MERCHANT_FEATURES,
        "device_features": DEVICE_FEATURES,
        "quality_checks": quality,
        "leakage_audit": {
            "target": [
                "fraud_label"
            ],
            "simulation_metadata": [
                "scenario_id"
            ],
            "simulator_shortcut": [
                "is_new_device"
            ],
            "excluded_identifiers": [
                "transaction_id",
                "customer_id",
                "merchant_id",
                "device_id",
            ],
            "future_information_rule": (
                "Historical features use only "
                "information from strictly earlier "
                "timestamps."
            ),
            "forbidden_columns_found": audit[
                "forbidden_columns_found"
            ],
            "passed": audit["passed"],
        },
        "tests_executed": False,
        "test_results": None,
        "warnings": warnings,
        "limitations": limitations,
        "model_training_performed": False,
        "day4_verdict": verdict,
    }


def main() -> None:
    original = load_data(
        INPUT_PATH
    )

    df = original.copy()

    df = add_basic_features(df)
    df = add_customer_history(df)
    df = add_customer_velocity(df)
    df = add_merchant_history(df)
    df = add_device_history(df)
    df = clean_feature_values(df)

    feature_columns = build_feature_columns(
        df
    )

    report = build_report(
        original,
        df,
        feature_columns,
    )

    if not report["leakage_audit"]["passed"]:
        raise RuntimeError(
            "Leakage audit failed."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("=" * 60)
    print(
        "AegisRisk AI - Day 4 Feature Engineering"
    )
    print("=" * 60)

    print(
        f"Input rows:  {len(original):,}"
    )

    print(
        f"Output rows: {len(df):,}"
    )

    print(
        f"Feature columns: {len(feature_columns)}"
    )

    print("\nFeatures:")

    for column in feature_columns:
        print(f"  - {column}")

    print("\nLeakage audit: PASS")

    print(
        f"\nSaved dataset: {OUTPUT_PATH}"
    )

    print(
        f"Saved report: {REPORT_PATH}"
    )

    print(
        f"\nDay 4 preliminary verdict: "
        f"{report['day4_verdict']}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()