"""
AegisRisk AI
Day 5 - Evaluation Figures

Purpose:
    Generate visual evaluation artifacts for the
    three fraud detection models.

Figures:
    1. Precision-Recall curve
    2. Confusion matrix comparison

Important:
    This script does not retrain models.

    It evaluates the already-trained models on the
    same chronological validation period.

    Threshold tuning is NOT performed.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/processed/transactions_features.csv"
)

MODEL_DIR = Path(
    "models"
)

FIGURE_DIR = Path(
    "figures"
)


# ============================================================
# DATA CONFIGURATION
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
# MODELS
# ============================================================

MODEL_FILES = {
    "Logistic Regression":
        MODEL_DIR
        / "logistic_regression.joblib",

    "Random Forest":
        MODEL_DIR
        / "random_forest.joblib",

    "XGBoost":
        MODEL_DIR
        / "xgboost.joblib",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Load and validate the feature dataset.
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
            "Invalid timestamps found."
        )

    if df[
        TRANSACTION_ID_COLUMN
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs found."
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
    Recreate the Day 5 chronological split.

    Earliest 80%:
        training

    Latest 20%:
        validation
    """

    split_index = int(
        len(df)
        * train_fraction
    )

    if (
        split_index <= 0
        or split_index >= len(df)
    ):
        raise ValueError(
            "Invalid split index."
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
            "Chronological boundary invalid."
        )

    return (
        train_df,
        validation_df,
    )


# ============================================================
# PREPARE VALIDATION FEATURES
# ============================================================

def prepare_validation_data(
    validation_df: pd.DataFrame,
):
    """
    Remove target, identifiers, metadata and raw timestamp.

    The saved model pipelines contain their own
    preprocessing logic.
    """

    feature_columns = [
        column
        for column in validation_df.columns
        if column not in DROP_COLUMNS
    ]

    if not feature_columns:
        raise ValueError(
            "No model features available."
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
# LOAD MODELS AND PREDICT
# ============================================================

def generate_predictions(
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
):
    """
    Load every saved model and generate
    validation probabilities and predictions.
    """

    model_results = {}

    for model_name, model_path in (
        MODEL_FILES.items()
    ):

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model artifact: "
                f"{model_path}"
            )

        print(
            f"Loading {model_name}..."
        )

        model = joblib.load(
            model_path
        )

        probabilities = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        if (
            probabilities.min() < 0
            or probabilities.max() > 1
        ):
            raise ValueError(
                f"Invalid probabilities "
                f"from {model_name}."
            )

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        model_results[
            model_name
        ] = {
            "probabilities":
                probabilities,

            "predictions":
                predictions,
        }

    return model_results


# ============================================================
# PR CURVE
# ============================================================

def create_pr_curve(
    y_validation: pd.Series,
    model_results: dict,
):
    """
    Create a Precision-Recall curve for all models.

    Important:
        This visualizes different thresholds.

        It does NOT select a threshold.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 7)
    )

    for model_name, values in (
        model_results.items()
    ):

        probabilities = values[
            "probabilities"
        ]

        precision, recall, _ = (
            precision_recall_curve(
                y_validation,
                probabilities,
            )
        )

        pr_auc = (
            average_precision_score(
                y_validation,
                probabilities,
            )
        )

        plt.plot(
            recall,
            precision,
            label=(
                f"{model_name} "
                f"(AP={pr_auc:.4f})"
            ),
        )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    fraud_rate = (
        y_validation.mean()
    )

    plt.axhline(
        fraud_rate,
        linestyle="--",
        label=(
            f"Fraud prevalence "
            f"({fraud_rate:.4f})"
        ),
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "AegisRisk AI — Precision-Recall Curve"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "day5_pr_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"PR curve saved: "
        f"{output_path}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix_figure(
    y_validation: pd.Series,
    model_results: dict,
):
    """
    Create a comparison figure containing the confusion
    matrices for all three models.
    """

    model_names = list(
        model_results.keys()
    )

    # --------------------------------------------------------
    # Create separate figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(model_names),
        figsize=(15, 5),
    )

    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name in zip(
        axes,
        model_names,
    ):

        predictions = (
            model_results[
                model_name
            ]["predictions"]
        )

        cm = confusion_matrix(
            y_validation,
            predictions,
        )

        # ----------------------------------------------------
        # Display matrix
        # ----------------------------------------------------

        image = ax.imshow(
            cm,
            interpolation="nearest",
        )

        ax.set_title(
            model_name
        )

        ax.set_xlabel(
            "Predicted label"
        )

        ax.set_ylabel(
            "Actual label"
        )

        ax.set_xticks(
            [0, 1]
        )

        ax.set_yticks(
            [0, 1]
        )

        ax.set_xticklabels(
            ["Legitimate", "Fraud"]
        )

        ax.set_yticklabels(
            ["Legitimate", "Fraud"]
        )

        # ----------------------------------------------------
        # Write values inside cells
        # ----------------------------------------------------

        for row in range(
            cm.shape[0]
        ):

            for column in range(
                cm.shape[1]
            ):

                ax.text(
                    column,
                    row,
                    f"{cm[row, column]:,}",
                    ha="center",
                    va="center",
                )

    fig.suptitle(
        "AegisRisk AI — Confusion Matrix Comparison"
    )

    fig.tight_layout()

    output_path = (
        FIGURE_DIR
        / "day5_confusion_matrix.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Confusion matrix saved: "
        f"{output_path}"
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
        "Day 5 — Step 10: Evaluation Figures"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Chronological split
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
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    model_results = (
        generate_predictions(
            X_validation,
            y_validation,
        )
    )

    # --------------------------------------------------------
    # PR curve
    # --------------------------------------------------------

    create_pr_curve(
        y_validation,
        model_results,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    create_confusion_matrix_figure(
        y_validation,
        model_results,
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DAY 5 — STEP 10 COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()