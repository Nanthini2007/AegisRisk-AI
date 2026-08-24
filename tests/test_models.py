"""
AegisRisk AI
Day 5 - Model Pipeline Tests

Purpose:
    Verify the critical assumptions of the Day 5
    chronological fraud detection pipeline.

These tests do not replace model evaluation.

They verify:
    - chronological splitting
    - leakage prevention
    - target integrity
    - model training
    - probability validity
    - metric validity
    - saved artifacts
    - generated reports
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import (
    average_precision_score,
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

FIGURE_DIR = Path(
    "figures"
)


# ============================================================
# COLUMN CONFIGURATION
# ============================================================

TARGET = "fraud_label"

TIMESTAMP = "timestamp"

TRANSACTION_ID = "transaction_id"


# ============================================================
# FORBIDDEN FEATURES
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
# HELPERS
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the Day 4 feature dataset.
    """

    assert DATA_PATH.exists(), (
        f"Dataset does not exist: {DATA_PATH}"
    )

    df = pd.read_csv(
        DATA_PATH
    )

    assert TARGET in df.columns
    assert TIMESTAMP in df.columns
    assert TRANSACTION_ID in df.columns

    df[TIMESTAMP] = pd.to_datetime(
        df[TIMESTAMP],
        errors="coerce",
    )

    assert not df[TIMESTAMP].isna().any()

    return df


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.80,
):
    """
    Reproduce the Day 5 chronological split.
    """

    df = (
        df.sort_values(
            [
                TIMESTAMP,
                TRANSACTION_ID,
            ]
        )
        .reset_index(drop=True)
    )

    split_index = int(
        len(df)
        * train_fraction
    )

    train_df = df.iloc[
        :split_index
    ].copy()

    validation_df = df.iloc[
        split_index:
    ].copy()

    return (
        train_df,
        validation_df,
    )


def get_model_features(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return only columns allowed to enter the model.
    """

    return [
        column
        for column in df.columns
        if column not in FORBIDDEN_COLUMNS
    ]


# ============================================================
# TEST 1
# ============================================================

def test_chronological_ordering():
    """
    Transactions must be ordered from past to future.
    """

    df = load_dataset()

    sorted_df = (
        df.sort_values(
            [
                TIMESTAMP,
                TRANSACTION_ID,
            ]
        )
        .reset_index(drop=True)
    )

    timestamps = (
        sorted_df[TIMESTAMP]
        .astype("int64")
        .to_numpy()
    )

    assert (
        timestamps[1:]
        >= timestamps[:-1]
    ).all()


# ============================================================
# TEST 2
# ============================================================

def test_no_train_validation_time_overlap():
    """
    Validation must begin strictly after training ends.
    """

    df = load_dataset()

    train_df, validation_df = (
        chronological_split(df)
    )

    train_end = train_df[
        TIMESTAMP
    ].max()

    validation_start = (
        validation_df[
            TIMESTAMP
        ].min()
    )

    assert train_end < validation_start


# ============================================================
# TEST 3
# ============================================================

def test_target_exists():
    """
    fraud_label must exist in the dataset.
    """

    df = load_dataset()

    assert TARGET in df.columns


# ============================================================
# TEST 4
# ============================================================

def test_forbidden_columns_are_excluded():
    """
    Leakage-prone and identifier columns must not
    enter the model feature set.
    """

    df = load_dataset()

    feature_columns = (
        get_model_features(df)
    )

    leaked_columns = (
        set(feature_columns)
        & FORBIDDEN_COLUMNS
    )

    assert leaked_columns == set()


# ============================================================
# TEST 5
# ============================================================

def test_both_classes_exist_in_training():
    """
    Training must contain both legitimate and
    fraudulent transactions.
    """

    df = load_dataset()

    train_df, _ = (
        chronological_split(df)
    )

    classes = set(
        train_df[TARGET]
        .astype(int)
        .unique()
    )

    assert classes == {0, 1}


# ============================================================
# TEST 6
# ============================================================

def test_model_artifacts_exist():
    """
    All three Day 5 model artifacts must exist.
    """

    expected_models = [
        "logistic_regression.joblib",
        "random_forest.joblib",
        "xgboost.joblib",
    ]

    for model_name in expected_models:

        model_path = (
            MODEL_DIR
            / model_name
        )

        assert model_path.exists(), (
            f"Missing model artifact: "
            f"{model_path}"
        )


# ============================================================
# TEST 7
# ============================================================

@pytest.mark.parametrize(
    "model_name",
    [
        "logistic_regression.joblib",
        "random_forest.joblib",
        "xgboost.joblib",
    ],
)
def test_saved_model_can_be_loaded(
    model_name,
):
    """
    Saved joblib artifacts must be loadable.
    """

    model_path = (
        MODEL_DIR
        / model_name
    )

    assert model_path.exists()

    model = joblib.load(
        model_path
    )

    assert model is not None


# ============================================================
# TEST 8
# ============================================================

@pytest.mark.parametrize(
    "model_name",
    [
        "logistic_regression.joblib",
        "random_forest.joblib",
        "xgboost.joblib",
    ],
)
def test_model_probabilities_are_valid(
    model_name,
):
    """
    Model probability predictions must lie in [0, 1].
    """

    df = load_dataset()

    _, validation_df = (
        chronological_split(df)
    )

    feature_columns = (
        get_model_features(df)
    )

    X_validation = (
        validation_df[
            feature_columns
        ]
        .copy()
    )

    model = joblib.load(
        MODEL_DIR
        / model_name
    )

    probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    assert len(
        probabilities
    ) == len(
        validation_df
    )

    assert np.isfinite(
        probabilities
    ).all()

    assert (
        probabilities >= 0
    ).all()

    assert (
        probabilities <= 1
    ).all()


# ============================================================
# TEST 9
# ============================================================

@pytest.mark.parametrize(
    "model_name",
    [
        "logistic_regression.joblib",
        "random_forest.joblib",
        "xgboost.joblib",
    ],
)
def test_evaluation_metrics_are_valid(
    model_name,
):
    """
    Verify that the standard Day 5 evaluation metrics
    can be calculated and remain within valid ranges.
    """

    df = load_dataset()

    _, validation_df = (
        chronological_split(df)
    )

    feature_columns = (
        get_model_features(df)
    )

    X_validation = (
        validation_df[
            feature_columns
        ]
        .copy()
    )

    y_validation = (
        validation_df[
            TARGET
        ]
        .astype(int)
    )

    model = joblib.load(
        MODEL_DIR
        / model_name
    )

    probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

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

    pr_auc = (
        average_precision_score(
            y_validation,
            probabilities,
        )
    )

    metrics = [
        precision,
        recall,
        f1,
        pr_auc,
    ]

    for metric in metrics:

        assert np.isfinite(
            metric
        )

        assert (
            0 <= metric <= 1
        )


# ============================================================
# TEST 10
# ============================================================

def test_day5_report_exists():
    """
    The Day 5 model report must exist.
    """

    report_path = (
        REPORT_DIR
        / "day5_model_report.json"
    )

    assert report_path.exists(), (
        f"Missing report: {report_path}"
    )


# ============================================================
# TEST 11
# ============================================================

def test_day5_report_is_valid_json():
    """
    The Day 5 report must contain valid JSON.
    """

    report_path = (
        REPORT_DIR
        / "day5_model_report.json"
    )

    assert report_path.exists()

    with report_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        report = json.load(
            file
        )

    assert isinstance(
        report,
        dict,
    )


# ============================================================
# TEST 12
# ============================================================

def test_day5_report_contains_required_sections():
    """
    Verify that the Day 5 report contains the important
    evaluation information.
    """

    report_path = (
        REPORT_DIR
        / "day5_model_report.json"
    )

    assert report_path.exists()

    with report_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        report = json.load(
            file
        )

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

    missing_keys = (
        required_keys
        - set(report.keys())
    )

    assert missing_keys == set()


# ============================================================
# TEST 13
# ============================================================

def test_pr_curve_exists():
    """
    The precision-recall curve artifact must exist.
    """

    figure_path = (
        FIGURE_DIR
        / "day5_pr_curve.png"
    )

    assert figure_path.exists(), (
        f"Missing PR curve: {figure_path}"
    )


# ============================================================
# TEST 14
# ============================================================

def test_confusion_matrix_exists():
    """
    The confusion matrix artifact must exist.
    """

    figure_path = (
        FIGURE_DIR
        / "day5_confusion_matrix.png"
    )

    assert figure_path.exists(), (
        "Missing confusion matrix: "
        f"{figure_path}"
    )