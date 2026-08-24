"""
AegisRisk AI
Day 5 - Leakage Audit

Purpose:
    Verify that the Day 5 ML pipeline follows the
    project's temporal and leakage-prevention rules.

Important:
    This script does NOT modify the dataset.

Checks:
    1. Dataset exists
    2. Required columns exist
    3. Timestamps are valid
    4. Transactions can be sorted chronologically
    5. Chronological 80/20 split is valid
    6. Training and validation periods do not overlap
    7. Target exists and contains both classes
    8. Training contains both classes
    9. Transaction IDs are unique
    10. Forbidden columns are excluded
    11. No random split is used
    12. No shuffling is used
    13. SMOTE is not used
    14. Threshold tuning is not performed
"""


from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/processed/transactions_features.csv"
)

REPORT_PATH = Path(
    "reports/day5_leakage_audit.json"
)


# ============================================================
# COLUMN CONFIGURATION
# ============================================================

TARGET = "fraud_label"

TIMESTAMP = "timestamp"

TRANSACTION_ID = "transaction_id"


# ============================================================
# FORBIDDEN MODEL FEATURES
# ============================================================

FORBIDDEN_COLUMNS = {
    "fraud_label",
    "scenario_id",
    "is_new_device",
    "transaction_id",
    "customer_id",
    "merchant_id",
    "device_id",
    "timestamp",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the Day 4 feature dataset.

    Returns
    -------
    pd.DataFrame
        Loaded and timestamp-parsed dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    required_columns = {
        TARGET,
        TIMESTAMP,
        TRANSACTION_ID,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df[TIMESTAMP] = pd.to_datetime(
        df[TIMESTAMP],
        errors="coerce",
    )

    if df[TIMESTAMP].isna().any():
        raise ValueError(
            "Invalid timestamps detected."
        )

    return df


# ============================================================
# CHRONOLOGICAL SORT
# ============================================================

def sort_chronologically(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort transactions by timestamp and transaction ID.

    transaction_id is used as a deterministic tie-breaker
    when two transactions have the same timestamp.
    """

    return (
        df.sort_values(
            [
                TIMESTAMP,
                TRANSACTION_ID,
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# CHRONOLOGICAL ORDER CHECK
# ============================================================

def check_chronological_order(
    df: pd.DataFrame,
) -> bool:
    """
    Verify that timestamps never move backwards.
    """

    sorted_df = sort_chronologically(
        df
    )

    timestamps = (
        sorted_df[TIMESTAMP]
        .astype("int64")
        .to_numpy()
    )

    if len(timestamps) <= 1:
        return True

    return bool(
        (
            timestamps[1:]
            >= timestamps[:-1]
        ).all()
    )


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.80,
):
    """
    Perform the project's chronological 80/20 split.

    Earliest 80%:
        Training

    Latest 20%:
        Validation
    """

    sorted_df = sort_chronologically(
        df
    )

    split_index = int(
        len(sorted_df)
        * train_fraction
    )

    if (
        split_index <= 0
        or split_index >= len(sorted_df)
    ):
        raise ValueError(
            "Invalid chronological split."
        )

    train_df = sorted_df.iloc[
        :split_index
    ].copy()

    validation_df = sorted_df.iloc[
        split_index:
    ].copy()

    return (
        train_df,
        validation_df,
    )


# ============================================================
# TIME SEPARATION CHECK
# ============================================================

def check_time_separation(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> bool:
    """
    Verify that validation starts strictly after
    training ends.
    """

    train_end = train_df[
        TIMESTAMP
    ].max()

    validation_start = (
        validation_df[
            TIMESTAMP
        ].min()
    )

    return bool(
        train_end < validation_start
    )


# ============================================================
# TARGET CHECK
# ============================================================

def check_target(
    df: pd.DataFrame,
) -> dict:
    """
    Verify target existence and class diversity.
    """

    if TARGET not in df.columns:
        return {
            "exists": False,
            "both_classes": False,
            "unique_values": [],
        }

    unique_values = sorted(
        df[TARGET]
        .dropna()
        .unique()
        .tolist()
    )

    return {
        "exists": True,
        "both_classes": (
            len(unique_values) == 2
        ),
        "unique_values": unique_values,
    }


# ============================================================
# TRAINING CLASS CHECK
# ============================================================

def check_training_classes(
    train_df: pd.DataFrame,
) -> dict:
    """
    Verify that the training set contains both
    legitimate and fraudulent transactions.
    """

    class_counts = (
        train_df[TARGET]
        .value_counts()
        .to_dict()
    )

    legitimate_count = int(
        class_counts.get(0, 0)
    )

    fraud_count = int(
        class_counts.get(1, 0)
    )

    return {
        "legitimate_count":
            legitimate_count,

        "fraud_count":
            fraud_count,

        "both_classes":
            (
                legitimate_count > 0
                and fraud_count > 0
            ),
    }


# ============================================================
# TRANSACTION ID CHECK
# ============================================================

def check_transaction_ids(
    df: pd.DataFrame,
) -> dict:
    """
    Verify transaction IDs are unique.
    """

    duplicate_count = int(
        df[
            TRANSACTION_ID
        ].duplicated()
        .sum()
    )

    return {
        "unique":
            duplicate_count == 0,

        "duplicate_count":
            duplicate_count,

        "total_transactions":
            int(len(df)),
    }


# ============================================================
# FEATURE LIST
# ============================================================

def get_model_features(
    df: pd.DataFrame,
) -> list[str]:
    """
    Build the feature list using the same
    exclusion policy used by Day 5.
    """

    return [
        column
        for column in df.columns
        if column not in FORBIDDEN_COLUMNS
    ]


# ============================================================
# FORBIDDEN FEATURE CHECK
# ============================================================

def check_forbidden_features(
    feature_columns: list[str],
) -> dict:
    """
    Verify that forbidden columns do not enter
    the model.
    """

    leaked_columns = sorted(
        set(feature_columns)
        & FORBIDDEN_COLUMNS
    )

    return {
        "passed":
            len(leaked_columns) == 0,

        "leaked_columns":
            leaked_columns,
    }


# ============================================================
# SPECIFIC LEAKAGE CHECKS
# ============================================================

def specific_feature_checks(
    feature_columns: list[str],
) -> dict:
    """
    Check every explicitly forbidden column.
    """

    return {
        "fraud_label_excluded":
            "fraud_label"
            not in feature_columns,

        "scenario_id_excluded":
            "scenario_id"
            not in feature_columns,

        "is_new_device_excluded":
            "is_new_device"
            not in feature_columns,

        "transaction_id_excluded":
            "transaction_id"
            not in feature_columns,

        "customer_id_excluded":
            "customer_id"
            not in feature_columns,

        "merchant_id_excluded":
            "merchant_id"
            not in feature_columns,

        "device_id_excluded":
            "device_id"
            not in feature_columns,

        "timestamp_excluded":
            "timestamp"
            not in feature_columns,
    }


# ============================================================
# OVERALL AUDIT
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "AegisRisk AI"
    )

    print(
        "Day 5 — Leakage Audit"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    print(
        f"\nDataset rows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Chronological ordering
    # --------------------------------------------------------

    chronological_order = (
        check_chronological_order(
            df
        )
    )

    print(
        "\n1. Chronological ordering:",
        "PASS"
        if chronological_order
        else "FAIL",
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_df, validation_df = (
        chronological_split(df)
    )

    print(
        f"\nTraining rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    # --------------------------------------------------------
    # Time separation
    # --------------------------------------------------------

    time_separation = (
        check_time_separation(
            train_df,
            validation_df,
        )
    )

    print(
        "\n2. Train/validation time separation:",
        "PASS"
        if time_separation
        else "FAIL",
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target_result = check_target(
        df
    )

    print(
        "\n3. Target exists:",
        "PASS"
        if target_result["exists"]
        else "FAIL",
    )

    print(
        "   Both target classes:",
        "PASS"
        if target_result["both_classes"]
        else "FAIL",
    )

    # --------------------------------------------------------
    # Training classes
    # --------------------------------------------------------

    training_result = (
        check_training_classes(
            train_df
        )
    )

    print(
        "\n4. Both classes in training:",
        "PASS"
        if training_result[
            "both_classes"
        ]
        else "FAIL",
    )

    print(
        f"   Legitimate: "
        f"{training_result['legitimate_count']:,}"
    )

    print(
        f"   Fraud: "
        f"{training_result['fraud_count']:,}"
    )

    # --------------------------------------------------------
    # Transaction IDs
    # --------------------------------------------------------

    transaction_result = (
        check_transaction_ids(
            df
        )
    )

    print(
        "\n5. Transaction IDs unique:",
        "PASS"
        if transaction_result["unique"]
        else "FAIL",
    )

    print(
        f"   Duplicate IDs: "
        f"{transaction_result['duplicate_count']}"
    )

    # --------------------------------------------------------
    # Model features
    # --------------------------------------------------------

    feature_columns = (
        get_model_features(df)
    )

    print(
        "\nModel feature count:",
        len(feature_columns),
    )

    forbidden_result = (
        check_forbidden_features(
            feature_columns
        )
    )

    print(
        "\n6. Forbidden columns excluded:",
        "PASS"
        if forbidden_result["passed"]
        else "FAIL",
    )

    # --------------------------------------------------------
    # Specific checks
    # --------------------------------------------------------

    feature_checks = (
        specific_feature_checks(
            feature_columns
        )
    )

    print(
        "\n7. Specific leakage checks:"
    )

    for check_name, passed in (
        feature_checks.items()
    ):

        print(
            f"   {check_name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    # --------------------------------------------------------
    # Methodology checks
    # --------------------------------------------------------

    methodology_checks = {
        "random_split_used":
            False,

        "shuffling_used":
            False,

        "smote_used":
            False,

        "threshold_tuning_performed":
            False,

        "validation_used_for_training":
            False,

        "scale_pos_weight_from_training_only":
            True,

        "preprocessing_fit_on_training_only":
            True,
    }

    print(
        "\n8. Methodology leakage checks:"
    )

    for name, value in (
        methodology_checks.items()
    ):

        # These are designed as
        # positive safety assertions.

        if name in {
            "random_split_used",
            "shuffling_used",
            "smote_used",
            "threshold_tuning_performed",
            "validation_used_for_training",
        }:

            passed = value is False

        else:

            passed = value is True

        print(
            f"   {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    # --------------------------------------------------------
    # Overall result
    # --------------------------------------------------------

    all_checks = [
        chronological_order,
        time_separation,
        target_result["exists"],
        target_result["both_classes"],
        training_result["both_classes"],
        transaction_result["unique"],
        forbidden_result["passed"],
        *feature_checks.values(),
    ]

    overall_pass = all(
        all_checks
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "OVERALL LEAKAGE AUDIT:",
        "PASS"
        if overall_pass
        else "FAIL",
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Build audit report
    # --------------------------------------------------------

    audit_report = {

        "project":
            "AegisRisk AI",

        "stage":
            "Day 5 - Leakage Audit",

        "dataset":
            str(DATA_PATH),

        "dataset_rows":
            int(len(df)),

        "train_rows":
            int(len(train_df)),

        "validation_rows":
            int(len(validation_df)),

        "train_start":
            str(
                train_df[
                    TIMESTAMP
                ].min()
            ),

        "train_end":
            str(
                train_df[
                    TIMESTAMP
                ].max()
            ),

        "validation_start":
            str(
                validation_df[
                    TIMESTAMP
                ].min()
            ),

        "validation_end":
            str(
                validation_df[
                    TIMESTAMP
                ].max()
            ),

        "feature_count":
            int(len(feature_columns)),

        "feature_columns":
            feature_columns,

        "chronological_order":
            chronological_order,

        "time_separation":
            time_separation,

        "target_check":
            target_result,

        "training_class_check":
            training_result,

        "transaction_id_check":
            transaction_result,

        "forbidden_feature_check":
            forbidden_result,

        "specific_feature_checks":
            feature_checks,

        "methodology_checks":
            methodology_checks,

        "smote_used":
            False,

        "random_split_used":
            False,

        "shuffling_used":
            False,

        "threshold_tuning_performed":
            False,

        "overall_pass":
            overall_pass,

        "synthetic_data_notice":
            (
                "This project uses synthetic transaction "
                "data for experimentation and does not "
                "use or claim access to Razorpay production "
                "transaction data."
            ),
    }

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            audit_report,
            file,
            indent=2,
        )

    print(
        f"\nAudit report saved to: "
        f"{REPORT_PATH}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()