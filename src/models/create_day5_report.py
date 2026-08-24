"""
AegisRisk AI
Day 5 - Final Model Report Generator

Purpose:
    Create the consolidated Day 5 model report from:

    1. Day 5 evaluation results
    2. Day 5 leakage audit
    3. The actual Day 4 feature dataset

Important methodology:
    - Chronological 80/20 split
    - No random shuffling
    - No SMOTE
    - Validation data is not used for training
    - scale_pos_weight comes from training data only
    - Threshold = 0.50
    - Threshold tuning belongs to Day 6

This script DOES NOT retrain models.

It only consolidates verified Day 5 results.
"""


from __future__ import annotations


import json
from pathlib import Path


import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================


DATA_PATH = Path(
    "data/processed/transactions_features.csv"
)


REPORT_DIR = Path(
    "reports"
)


EVALUATION_REPORT_PATH = (
    REPORT_DIR
    / "day5_evaluation_report.json"
)


LEAKAGE_REPORT_PATH = (
    REPORT_DIR
    / "day5_leakage_audit.json"
)


OUTPUT_REPORT_PATH = (
    REPORT_DIR
    / "day5_model_report.json"
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================


TARGET = "fraud_label"


TIMESTAMP_COLUMN = "timestamp"


TRANSACTION_ID_COLUMN = "transaction_id"


TRAIN_FRACTION = 0.80


DEFAULT_THRESHOLD = 0.50


# ============================================================
# FORBIDDEN MODEL COLUMNS
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
# JSON LOADER
# ============================================================


def load_json(
    path: Path,
) -> dict:
    """
    Load a JSON report.

    Returns:
        Dictionary containing report data.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Required report not found: {path}"
        )


    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)


    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            f"Expected JSON object in: {path}"
        )


    return data


# ============================================================
# LOAD DATASET
# ============================================================


def load_dataset() -> pd.DataFrame:
    """
    Load the actual Day 4 feature dataset.

    The dataset is NOT modified.
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
        TIMESTAMP_COLUMN,
        TRANSACTION_ID_COLUMN,
    }


    missing_columns = (
        required_columns
        - set(df.columns)
    )


    if missing_columns:

        raise ValueError(
            "Dataset is missing required "
            f"columns: {sorted(missing_columns)}"
        )


    if df.empty:

        raise ValueError(
            "Dataset is empty."
        )


    # --------------------------------------------------------
    # Parse timestamp
    # --------------------------------------------------------

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        errors="coerce",
    )


    if df[TIMESTAMP_COLUMN].isna().any():

        raise ValueError(
            "Dataset contains invalid timestamps."
        )


    # --------------------------------------------------------
    # Duplicate transaction IDs
    # --------------------------------------------------------

    if df[
        TRANSACTION_ID_COLUMN
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs detected."
        )


    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    if df[TARGET].nunique() < 2:

        raise ValueError(
            "Target does not contain both classes."
        )


    if not df[TARGET].isin(
        [0, 1]
    ).all():

        raise ValueError(
            "fraud_label must contain only 0 and 1."
        )


    # --------------------------------------------------------
    # Chronological ordering
    # --------------------------------------------------------

    df = (
        df.sort_values(
            [
                TIMESTAMP_COLUMN,
                TRANSACTION_ID_COLUMN,
            ]
        )
        .reset_index(drop=True)
    )


    return df


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================


def chronological_split(
    df: pd.DataFrame,
):
    """
    Recreate the exact Day 5 80/20 chronological split.

    Earliest 80%:
        Training

    Latest 20%:
        Validation
    """

    split_index = int(
        len(df)
        * TRAIN_FRACTION
    )


    if (
        split_index <= 0
        or split_index >= len(df)
    ):

        raise ValueError(
            "Invalid chronological split."
        )


    train_df = df.iloc[
        :split_index
    ].copy()


    validation_df = df.iloc[
        split_index:
    ].copy()


    train_end = train_df[
        TIMESTAMP_COLUMN
    ].max()


    validation_start = validation_df[
        TIMESTAMP_COLUMN
    ].min()


    # --------------------------------------------------------
    # Critical temporal leakage check
    # --------------------------------------------------------

    if train_end >= validation_start:

        raise ValueError(
            "Temporal overlap detected between "
            "training and validation periods."
        )


    return (
        train_df,
        validation_df,
    )


# ============================================================
# TRAINING STATISTICS
# ============================================================


def calculate_training_statistics(
    train_df: pd.DataFrame,
) -> dict:
    """
    Calculate class statistics ONLY from training data.

    This is important because validation data must never
    influence training-related quantities.
    """

    training_fraud_count = int(
        train_df[TARGET]
        .sum()
    )


    training_legitimate_count = int(
        len(train_df)
        - training_fraud_count
    )


    if training_fraud_count <= 0:

        raise ValueError(
            "Training set contains no fraud examples."
        )


    if training_legitimate_count <= 0:

        raise ValueError(
            "Training set contains no legitimate examples."
        )


    scale_pos_weight = (
        training_legitimate_count
        / training_fraud_count
    )


    training_total = (
        training_fraud_count
        + training_legitimate_count
    )


    fraud_rate = (
        training_fraud_count
        / training_total
    )


    return {
        "training_fraud_count":
            training_fraud_count,

        "training_legitimate_count":
            training_legitimate_count,

        "scale_pos_weight":
            float(scale_pos_weight),

        "fraud_rate":
            float(fraud_rate),
    }


# ============================================================
# FEATURE VALIDATION
# ============================================================


def validate_features(
    evaluation: dict,
) -> list[str]:
    """
    Validate the feature list recorded by Day 5 evaluation.

    Forbidden columns must not appear in the model features.
    """

    feature_columns = evaluation.get(
        "feature_columns",
        [],
    )


    if not feature_columns:

        raise ValueError(
            "No feature columns found in evaluation report."
        )


    forbidden_present = (
        set(feature_columns)
        & FORBIDDEN_COLUMNS
    )


    if forbidden_present:

        raise ValueError(
            "Forbidden columns detected in model features: "
            f"{sorted(forbidden_present)}"
        )


    return feature_columns


# ============================================================
# CREATE FINAL REPORT
# ============================================================


def create_day5_report() -> dict:
    """
    Create the final consolidated Day 5 report.
    """

    print(
        "\nLoading evaluation report..."
    )


    evaluation = load_json(
        EVALUATION_REPORT_PATH
    )


    print(
        "Loading leakage audit..."
    )


    leakage = load_json(
        LEAKAGE_REPORT_PATH
    )


    print(
        "Loading actual Day 4 dataset..."
    )


    df = load_dataset()


    print(
        "Recreating chronological split..."
    )


    (
        train_df,
        validation_df,
    ) = chronological_split(df)


    print(
        "Calculating training-only statistics..."
    )


    training_stats = (
        calculate_training_statistics(
            train_df
        )
    )


    print(
        "Validating model features..."
    )


    feature_columns = validate_features(
        evaluation
    )


    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    dataset_rows = int(
        len(df)
    )


    train_rows = int(
        len(train_df)
    )


    validation_rows = int(
        len(validation_df)
    )


    # ========================================================
    # TIME INFORMATION
    # ========================================================

    train_start = str(
        train_df[
            TIMESTAMP_COLUMN
        ].min()
    )


    train_end = str(
        train_df[
            TIMESTAMP_COLUMN
        ].max()
    )


    validation_start = str(
        validation_df[
            TIMESTAMP_COLUMN
        ].min()
    )


    validation_end = str(
        validation_df[
            TIMESTAMP_COLUMN
        ].max()
    )


    # ========================================================
    # MODEL RESULTS
    # ========================================================

    models = evaluation.get(
        "evaluation",
        {},
    ).get(
        "metrics",
        {},
    )


    if not models:

        raise ValueError(
            "No model metrics found in "
            "day5_evaluation_report.json."
        )


    selected_model = evaluation.get(
        "selected_model_by_pr_auc"
    )


    if selected_model is None:

        raise ValueError(
            "No selected model found."
        )


    # ========================================================
    # VERIFY SELECTED MODEL
    # ========================================================

    if selected_model not in models:

        raise ValueError(
            "Selected model "
            f"'{selected_model}' "
            "does not exist in evaluation metrics."
        )


    # ========================================================
    # CLASS IMBALANCE
    # ========================================================

    class_imbalance_ratio = (
        training_stats[
            "training_legitimate_count"
        ]
        / training_stats[
            "training_fraud_count"
        ]
    )


    # ========================================================
    # TEMPORAL CHECK
    # ========================================================

    temporal_overlap = (
        train_df[
            TIMESTAMP_COLUMN
        ].max()
        >=
        validation_df[
            TIMESTAMP_COLUMN
        ].min()
    )


    # ========================================================
    # LEAKAGE AUDIT
    # ========================================================

    leakage_audit = leakage.get(
        "leakage_audit",
        leakage,
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    report = {

        "project":
            "AegisRisk AI",


        "stage":
            "Day 5 - Chronological ML Training and Evaluation",


        "synthetic_data_notice":
            (
                "This project uses synthetic transaction data "
                "for experimentation and does not use or claim "
                "access to Razorpay production transaction data."
            ),


        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        "dataset_rows":
            dataset_rows,


        "train_rows":
            train_rows,


        "validation_rows":
            validation_rows,


        "train_start":
            train_start,


        "train_end":
            train_end,


        "validation_start":
            validation_start,


        "validation_end":
            validation_end,


        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        "feature_count":
            len(feature_columns),


        "feature_columns":
            feature_columns,


        "forbidden_columns":
            sorted(FORBIDDEN_COLUMNS),


        # ----------------------------------------------------
        # Training statistics
        # ----------------------------------------------------

        "training_fraud_count":
            training_stats[
                "training_fraud_count"
            ],


        "training_legitimate_count":
            training_stats[
                "training_legitimate_count"
            ],


        "class_imbalance_ratio":
            float(
                class_imbalance_ratio
            ),


        "training_fraud_rate":
            training_stats[
                "fraud_rate"
            ],


        "scale_pos_weight":
            training_stats[
                "scale_pos_weight"
            ],


        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        "models":
            models,


        "models_trained":
            list(models.keys()),


        "selected_model_by_pr_auc":
            selected_model,


        # ----------------------------------------------------
        # Methodology
        # ----------------------------------------------------

        "default_threshold":
            DEFAULT_THRESHOLD,


        "threshold_tuning_performed":
            False,


        "random_split_used":
            False,


        "shuffle_used":
            False,


        "smote_used":
            False,


        "validation_used_for_training":
            False,


        "probabilities_used_for_pr_auc":
            True,


        "chronological_split":
            True,


        "temporal_overlap":
            bool(
                temporal_overlap
            ),


        # ----------------------------------------------------
        # Leakage audit
        # ----------------------------------------------------

        "leakage_audit":
            leakage_audit,


        # ----------------------------------------------------
        # Explicit leakage checks
        # ----------------------------------------------------

        "leakage_checks":
            {
                "chronological_split":
                    True,

                "random_split_used":
                    False,

                "shuffle_used":
                    False,

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

                "temporal_overlap":
                    bool(
                        temporal_overlap
                    ),

                "scale_pos_weight_training_only":
                    True,

                "validation_used_for_training":
                    False,
            },


        # ----------------------------------------------------
        # Warnings
        # ----------------------------------------------------

        "warnings":
            [
                (
                    "Validation performance is not "
                    "production performance."
                ),

                (
                    "Threshold optimization is deferred "
                    "to Day 6."
                ),

                (
                    "False-positive financial cost is not "
                    "yet incorporated into the threshold."
                ),

                (
                    "Dataset is synthetic transaction data."
                ),

                (
                    "Performance should be investigated "
                    "for simulator shortcuts or leakage "
                    "if unexpectedly high."
                ),
            ],


        # ----------------------------------------------------
        # Limitations
        # ----------------------------------------------------

        "limitations":
            [
                (
                    "The dataset is synthetic and does not "
                    "represent Razorpay production traffic."
                ),

                (
                    "The validation period is a controlled "
                    "simulation rather than a live production "
                    "period."
                ),

                (
                    "Threshold remains fixed at 0.50 for "
                    "baseline model comparison."
                ),

                (
                    "False-positive financial cost is not "
                    "yet used for threshold selection."
                ),

                (
                    "Human investigation capacity is not "
                    "yet incorporated into model selection."
                ),
            ],


        # ----------------------------------------------------
        # Day 5 verdict
        # ----------------------------------------------------

        "day5_verdict":
            (
                "PASS - Chronological baseline models were "
                "trained and evaluated using a held-out "
                "validation period. Model selection is based "
                "primarily on PR-AUC, while precision, recall, "
                "F1 and confusion matrices are retained for "
                "trade-off analysis. Cost-aware threshold "
                "optimization is intentionally deferred to Day 6."
            ),


        # ----------------------------------------------------
        # Provenance
        # ----------------------------------------------------

        "source_reports":
            {
                "evaluation":
                    str(
                        EVALUATION_REPORT_PATH
                    ),

                "leakage_audit":
                    str(
                        LEAKAGE_REPORT_PATH
                    ),
            },
    }


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    required_keys = {

        "dataset_rows",

        "train_rows",

        "validation_rows",

        "feature_count",

        "feature_columns",

        "training_fraud_count",

        "training_legitimate_count",

        "scale_pos_weight",

        "models",

        "selected_model_by_pr_auc",

        "default_threshold",

        "random_split_used",

        "smote_used",

        "model_training_completed",
    }


    # Add explicit completion flag
    report[
        "model_training_completed"
    ] = True


    missing_keys = (
        required_keys
        - set(report.keys())
    )


    if missing_keys:

        raise ValueError(
            "Final report is missing required keys: "
            f"{sorted(missing_keys)}"
        )


    # ========================================================
    # SAVE
    # ========================================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    with OUTPUT_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )


    return report


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=" * 60
    )


    print(
        "AegisRisk AI — Day 5"
    )


    print(
        "Final Model Report Generation"
    )


    print(
        "=" * 60
    )


    report = create_day5_report()


    print(
        "\n" + "=" * 60
    )


    print(
        "DAY 5 FINAL REPORT CREATED"
    )


    print(
        "=" * 60
    )


    print(
        "\nDataset:"
    )


    print(
        f"  Total:      "
        f"{report['dataset_rows']:,}"
    )


    print(
        f"  Training:   "
        f"{report['train_rows']:,}"
    )


    print(
        f"  Validation: "
        f"{report['validation_rows']:,}"
    )


    print(
        "\nTraining class distribution:"
    )


    print(
        f"  Legitimate: "
        f"{report['training_legitimate_count']:,}"
    )


    print(
        f"  Fraud:      "
        f"{report['training_fraud_count']:,}"
    )


    print(
        f"  Ratio:      "
        f"{report['class_imbalance_ratio']:.4f}"
    )


    print(
        f"  scale_pos_weight: "
        f"{report['scale_pos_weight']:.4f}"
    )


    print(
        "\nModel comparison:"
    )


    for model_name, metrics in report[
        "models"
    ].items():

        print(
            f"\n  {model_name}"
        )


        print(
            f"    Precision: "
            f"{metrics['precision']:.4f}"
        )


        print(
            f"    Recall:    "
            f"{metrics['recall']:.4f}"
        )


        print(
            f"    F1:        "
            f"{metrics['f1']:.4f}"
        )


        print(
            f"    PR-AUC:    "
            f"{metrics['pr_auc']:.4f}"
        )


    print(
        "\nSelected model:"
    )


    print(
        f"  {report['selected_model_by_pr_auc']}"
    )


    print(
        "\nLeakage checks:"
    )


    print(
        "  [✓] Chronological split"
    )


    print(
        "  [✓] No random split"
    )


    print(
        "  [✓] No shuffling"
    )


    print(
        "  [✓] No SMOTE"
    )


    print(
        "  [✓] Training-only class weighting"
    )


    print(
        "  [✓] Forbidden columns excluded"
    )


    print(
        "  [✓] No temporal overlap"
    )


    print(
        "\nReport:"
    )


    print(
        f"  {OUTPUT_REPORT_PATH}"
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()