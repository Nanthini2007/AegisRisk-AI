"""
AegisRisk AI
Day 5 - Model Evaluation

Purpose:
    Evaluate the three Day 5 models on the
    chronological validation period.

Models:
    1. Logistic Regression
    2. Random Forest
    3. XGBoost

Evaluation metrics:
    - Precision
    - Recall
    - F1-score
    - PR-AUC
    - Confusion matrix

Important:
    This module does NOT retrain the models.

    It loads the saved models and evaluates them
    on the held-out chronological validation data.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/transactions_features.csv"
)

MODEL_DIR = Path(
    "models"
)

REPORT_DIR = Path(
    "reports"
)

REPORT_PATH = (
    REPORT_DIR
    / "day5_evaluation_report.json"
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

TARGET = "fraud_label"

TIMESTAMP_COLUMN = "timestamp"

TRANSACTION_ID_COLUMN = "transaction_id"


# ============================================================
# FORBIDDEN COLUMNS
# ============================================================

DROP_COLUMNS = {
    TARGET,
    "transaction_id",
    "customer_id",
    "merchant_id",
    "device_id",
    "scenario_id",
    "is_new_device",
    "timestamp",
}


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILES = {
    "logistic_regression":
        MODEL_DIR
        / "logistic_regression.joblib",

    "random_forest":
        MODEL_DIR
        / "random_forest.joblib",

    "xgboost":
        MODEL_DIR
        / "xgboost.joblib",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Load the Day 4 feature dataset.

    The dataset is sorted chronologically before
    creating the validation period.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        TARGET,
        TIMESTAMP_COLUMN,
        TRANSACTION_ID_COLUMN,
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    if df[TIMESTAMP_COLUMN].isna().any():
        raise ValueError(
            "Invalid timestamps detected."
        )

    if df[
        TRANSACTION_ID_COLUMN
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs detected."
        )

    if df[TARGET].nunique() < 2:
        raise ValueError(
            "Target does not contain both classes."
        )

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
    train_fraction: float = 0.80,
):
    """
    Recreate the exact chronological 80/20 split.

    No shuffling.
    No random splitting.
    """

    split_index = int(
        len(df) * train_fraction
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

    validation_start = (
        validation_df[
            TIMESTAMP_COLUMN
        ].min()
    )

    if train_end >= validation_start:
        raise ValueError(
            "Temporal overlap detected."
        )

    return train_df, validation_df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_validation_data(
    validation_df: pd.DataFrame,
):
    """
    Prepare validation features.

    The saved model contains its own preprocessing
    pipeline, so we only remove forbidden columns here.
    """

    feature_columns = [
        column
        for column in validation_df.columns
        if column not in DROP_COLUMNS
    ]

    if not feature_columns:
        raise ValueError(
            "No model features remain."
        )

    forbidden_present = (
        DROP_COLUMNS
        & set(feature_columns)
    )

    if forbidden_present:
        raise ValueError(
            "Forbidden columns found in model input: "
            f"{sorted(forbidden_present)}"
        )

    X_validation = validation_df[
        feature_columns
    ].copy()

    y_validation = validation_df[
        TARGET
    ].astype(int)

    return (
        X_validation,
        y_validation,
        feature_columns,
    )


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    model,
    X_validation,
    y_validation,
):
    """
    Evaluate one saved model.

    Threshold:
        0.50

    PR-AUC:
        Uses probability predictions.
    """

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # --------------------------------------------------------
    # Probability validation
    # --------------------------------------------------------

    if probabilities.min() < 0:
        raise ValueError(
            "Probability below 0 detected."
        )

    if probabilities.max() > 1:
        raise ValueError(
            "Probability above 1 detected."
        )

    # --------------------------------------------------------
    # Initial threshold
    # --------------------------------------------------------

    threshold = 0.50

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_validation,
        predictions,
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    return {
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
        "pr_auc": float(
            pr_auc
        ),
        "threshold": threshold,
        "confusion_matrix": (
            cm.tolist()
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "validation_predictions": int(
            predictions.sum()
        ),
    }


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    model_name: str,
    metrics: dict,
):
    """
    Print evaluation results clearly.
    """

    print(
        "\n" + "-" * 60
    )

    print(
        model_name.upper()
    )

    print(
        "-" * 60
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"Threshold : "
        f"{metrics['threshold']:.2f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        metrics["confusion_matrix"]
    )

    print(
        "\nConfusion matrix interpretation:"
    )

    print(
        f"  True Negatives : "
        f"{metrics['true_negatives']:,}"
    )

    print(
        f"  False Positives: "
        f"{metrics['false_positives']:,}"
    )

    print(
        f"  False Negatives: "
        f"{metrics['false_negatives']:,}"
    )

    print(
        f"  True Positives : "
        f"{metrics['true_positives']:,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "=" * 70
    )

    print(
        "AegisRisk AI"
    )

    print(
        "Day 5 — Step 8: Full Model Evaluation"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_data()

    print(
        f"\nDataset rows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    train_df, validation_df = (
        chronological_split(df)
    )

    print(
        "\nChronological validation period:"
    )

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Training end: "
        f"{train_df[TIMESTAMP_COLUMN].max()}"
    )

    print(
        f"Validation start: "
        f"{validation_df[TIMESTAMP_COLUMN].min()}"
    )

    # --------------------------------------------------------
    # Prepare validation features
    # --------------------------------------------------------

    (
        X_validation,
        y_validation,
        feature_columns,
    ) = prepare_validation_data(
        validation_df
    )

    print(
        f"\nValidation feature count: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Check validation classes
    # --------------------------------------------------------

    validation_fraud = int(
        y_validation.sum()
    )

    validation_legitimate = int(
        len(y_validation)
        - validation_fraud
    )

    print(
        "\nValidation target distribution:"
    )

    print(
        f"Legitimate: "
        f"{validation_legitimate:,}"
    )

    print(
        f"Fraud: "
        f"{validation_fraud:,}"
    )

    if validation_fraud == 0:
        raise ValueError(
            "Validation set contains no fraud examples."
        )

    if validation_legitimate == 0:
        raise ValueError(
            "Validation set contains no legitimate examples."
        )

    # --------------------------------------------------------
    # Evaluate all models
    # --------------------------------------------------------

    results = {}

    for model_name, model_path in (
        MODEL_FILES.items()
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            f"Loading {model_name}..."
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact missing: "
                f"{model_path}"
            )

        model = joblib.load(
            model_path
        )

        print(
            f"Model loaded: "
            f"{model_path}"
        )

        metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
        )

        results[
            model_name
        ] = metrics

        print_metrics(
            model_name,
            metrics,
        )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    comparison = []

    for model_name, metrics in (
        results.items()
    ):

        comparison.append(
            {
                "model": model_name,
                "precision": metrics[
                    "precision"
                ],
                "recall": metrics[
                    "recall"
                ],
                "f1": metrics[
                    "f1"
                ],
                "pr_auc": metrics[
                    "pr_auc"
                ],
            }
        )

    # --------------------------------------------------------
    # Select candidate by PR-AUC
    # --------------------------------------------------------

    best_model = max(
        results,
        key=lambda name:
            results[name]["pr_auc"],
    )

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    report = {
        "project": "AegisRisk AI",

        "stage": (
            "Day 5 - Step 8 "
            "Full Model Evaluation"
        ),

        "dataset": {
            "path": str(DATA_PATH),
            "rows": int(len(df)),
            "validation_rows": int(
                len(validation_df)
            ),
        },

        "chronological_split": {
            "method": "80/20 chronological",
            "random_split_used": False,
            "shuffle_used": False,
            "train_end": str(
                train_df[
                    TIMESTAMP_COLUMN
                ].max()
            ),
            "validation_start": str(
                validation_df[
                    TIMESTAMP_COLUMN
                ].min()
            ),
            "temporal_overlap": False,
        },

        "validation_target_distribution": {
            "legitimate": validation_legitimate,
            "fraud": validation_fraud,
        },

        "feature_count": len(
            feature_columns
        ),

        "feature_columns": (
            feature_columns
        ),

        "forbidden_columns": sorted(
            DROP_COLUMNS
        ),

        "evaluation": {
            "threshold": 0.50,
            "threshold_tuning_performed": False,
            "metrics": results,
            "comparison": comparison,
        },

        "selected_model_by_pr_auc": (
            best_model
        ),

        "methodology": {
            "smote_used": False,
            "validation_used_for_training": False,
            "probabilities_used_for_pr_auc": True,
        },

        "warnings": [
            "Validation performance is not production performance.",
            "Threshold optimization is deferred to Day 6.",
            "False-positive financial cost is not yet incorporated into the threshold.",
            "Dataset is synthetic transaction data.",
        ],

        "synthetic_data_notice": (
            "This project uses synthetic transaction "
            "data for experimentation and does not use "
            "or claim access to Razorpay production "
            "transaction data."
        ),
    }

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        "\n"
    )

    print(
        f"{'Model':<25}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'PR-AUC':>12}"
    )

    print(
        "-" * 73
    )

    for row in comparison:

        print(
            f"{row['model']:<25}"
            f"{row['precision']:>12.4f}"
            f"{row['recall']:>12.4f}"
            f"{row['f1']:>12.4f}"
            f"{row['pr_auc']:>12.4f}"
        )

    print(
        "\nCandidate by PR-AUC:"
    )

    print(
        best_model
    )

    print(
        "\nReport created:"
    )

    print(
        REPORT_PATH
    )

    print(
        "\nDAY 5 — STEP 8 COMPLETE"
    )


if __name__ == "__main__":
    main()