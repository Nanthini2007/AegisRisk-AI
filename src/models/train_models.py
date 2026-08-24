"""
AegisRisk AI
Day 5 - Chronological ML Training and Evaluation

Current stage:
Step 7 - XGBoost

Purpose:
    Train and compare:
        1. Logistic Regression
        2. Random Forest
        3. XGBoost

Critical methodology:
    - Chronological 80/20 split
    - No random train/test split
    - No shuffling
    - No SMOTE
    - Validation data never used for fitting
    - scale_pos_weight calculated from training data only
    - Threshold = 0.50
    - Threshold tuning belongs to Day 6
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "XGBoost is not installed.\n"
        "Install it using:\n"
        "python -m pip install xgboost"
    ) from exc


# ============================================================
# PROJECT PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/transactions_features.csv"
)

MODEL_DIR = Path("models")


# ============================================================
# DATASET CONFIGURATION
# ============================================================

TARGET = "fraud_label"

TIMESTAMP_COLUMN = "timestamp"

TRANSACTION_ID_COLUMN = "transaction_id"


# ============================================================
# FORBIDDEN MODEL COLUMNS
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
# STEP 1 — LOAD AND VALIDATE DATA
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Load and validate the Day 4 feature dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError(
            "Dataset is empty."
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
            "Required columns are missing: "
            f"{sorted(missing_columns)}"
        )

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    if df[TIMESTAMP_COLUMN].isna().any():
        raise ValueError(
            "Invalid timestamps found."
        )

    duplicate_ids = df[
        TRANSACTION_ID_COLUMN
    ].duplicated()

    if duplicate_ids.any():
        raise ValueError(
            "Duplicate transaction IDs detected."
        )

    if df[TARGET].nunique() < 2:
        raise ValueError(
            "Target contains fewer than two classes."
        )

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(
            "fraud_label must contain only 0 and 1."
        )

    return df


# ============================================================
# STEP 2 — SORT CHRONOLOGICALLY
# ============================================================

def sort_chronologically(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort transactions by timestamp.

    transaction_id provides a deterministic
    tie-breaker for equal timestamps.
    """

    sorted_df = (
        df.sort_values(
            [
                TIMESTAMP_COLUMN,
                TRANSACTION_ID_COLUMN,
            ]
        )
        .reset_index(drop=True)
    )

    if not sorted_df[
        TIMESTAMP_COLUMN
    ].is_monotonic_increasing:

        raise ValueError(
            "Dataset is not chronologically ordered."
        )

    return sorted_df


# ============================================================
# STEP 3 — CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.80,
):
    """
    Create an 80/20 chronological split.

    Earliest 80%:
        Training

    Latest 20%:
        Validation
    """

    if not 0 < train_fraction < 1:
        raise ValueError(
            "train_fraction must be between 0 and 1."
        )

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

    validation_start = validation_df[
        TIMESTAMP_COLUMN
    ].min()

    if train_end >= validation_start:
        raise ValueError(
            "Temporal leakage detected: "
            "training period overlaps validation period."
        )

    if (
        len(train_df)
        + len(validation_df)
        != len(df)
    ):
        raise ValueError(
            "Train + validation rows do not "
            "equal original dataset size."
        )

    return train_df, validation_df


# ============================================================
# STEP 4 — FEATURE SELECTION
# ============================================================

def prepare_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
):
    """
    Prepare leakage-safe model features.

    Excluded:
        fraud_label
        transaction_id
        customer_id
        merchant_id
        device_id
        scenario_id
        is_new_device
        timestamp
    """

    feature_columns = [
        column
        for column in train_df.columns
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
            "Forbidden columns detected: "
            f"{sorted(forbidden_present)}"
        )

    X_train = train_df[
        feature_columns
    ].copy()

    X_validation = validation_df[
        feature_columns
    ].copy()

    y_train = train_df[
        TARGET
    ].astype(int)

    y_validation = validation_df[
        TARGET
    ].astype(int)

    categorical_columns = (
        X_train
        .select_dtypes(
            include=[
                "object",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    numerical_columns = [
        column
        for column in feature_columns
        if column not in categorical_columns
    ]

    return (
        X_train,
        X_validation,
        y_train,
        y_validation,
        numerical_columns,
        categorical_columns,
        feature_columns,
    )


# ============================================================
# LOGISTIC REGRESSION PREPROCESSOR
# ============================================================

def make_logistic_preprocessor(
    numerical_columns,
    categorical_columns,
):
    """
    Logistic Regression preprocessing.

    Numerical:
        median imputation
        StandardScaler

    Categorical:
        most-frequent imputation
        OneHotEncoder
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# TREE PREPROCESSOR
# ============================================================

def make_tree_preprocessor(
    numerical_columns,
    categorical_columns,
):
    """
    Tree-model preprocessing.

    Numerical:
        median imputation

    Categorical:
        most-frequent imputation
        OneHotEncoder

    No numerical scaling.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# STEP 5 — LOGISTIC REGRESSION
# ============================================================

def make_logistic_model(
    numerical_columns,
    categorical_columns,
):
    """
    Create Logistic Regression pipeline.
    """

    preprocessor = (
        make_logistic_preprocessor(
            numerical_columns,
            categorical_columns,
        )
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# STEP 6 — RANDOM FOREST
# ============================================================

def make_random_forest_model(
    numerical_columns,
    categorical_columns,
):
    """
    Create Random Forest pipeline.
    """

    preprocessor = (
        make_tree_preprocessor(
            numerical_columns,
            categorical_columns,
        )
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# STEP 7 — XGBOOST
# ============================================================

def make_xgboost_model(
    numerical_columns,
    categorical_columns,
    scale_pos_weight: float,
):
    """
    Create XGBoost pipeline.

    scale_pos_weight is calculated ONLY
    from the training data.
    """

    preprocessor = (
        make_tree_preprocessor(
            numerical_columns,
            categorical_columns,
        )
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_validation,
    y_validation,
):
    """
    Evaluate a trained model.

    Threshold:
        0.50

    PR-AUC:
        calculated from probabilities.
    """

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    if (
        probabilities.min() < 0
        or probabilities.max() > 1
    ):
        raise ValueError(
            "Predicted probabilities are outside [0, 1]."
        )

    threshold = 0.50

    predictions = (
        probabilities >= threshold
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

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    cm = confusion_matrix(
        y_validation,
        predictions,
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

def train_logistic_regression(
    X_train,
    y_train,
    X_validation,
    y_validation,
    numerical_columns,
    categorical_columns,
):

    print(
        "\nTraining Logistic Regression..."
    )

    model = make_logistic_model(
        numerical_columns,
        categorical_columns,
    )

    model.fit(
        X_train,
        y_train,
    )

    metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_DIR
        / "logistic_regression.joblib",
    )

    return model, metrics


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_random_forest(
    X_train,
    y_train,
    X_validation,
    y_validation,
    numerical_columns,
    categorical_columns,
):

    print(
        "\nTraining Random Forest..."
    )

    model = make_random_forest_model(
        numerical_columns,
        categorical_columns,
    )

    model.fit(
        X_train,
        y_train,
    )

    metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_DIR
        / "random_forest.joblib",
    )

    return model, metrics


# ============================================================
# STEP 7 — TRAIN XGBOOST
# ============================================================

def train_xgboost(
    X_train,
    y_train,
    X_validation,
    y_validation,
    numerical_columns,
    categorical_columns,
):
    """
    Train XGBoost.

    IMPORTANT:
        scale_pos_weight is derived only
        from y_train.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "STEP 7 — XGBoost"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Calculate class imbalance from TRAINING DATA ONLY
    # --------------------------------------------------------

    positive_count = int(
        y_train.sum()
    )

    negative_count = int(
        len(y_train) - positive_count
    )

    if positive_count == 0:
        raise ValueError(
            "Training set contains no fraud examples."
        )

    if negative_count == 0:
        raise ValueError(
            "Training set contains no legitimate examples."
        )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print(
        "\nTraining class distribution:"
    )

    print(
        f"  Legitimate: "
        f"{negative_count:,}"
    )

    print(
        f"  Fraud:      "
        f"{positive_count:,}"
    )

    print(
        "\nXGBoost class weighting:"
    )

    print(
        f"  scale_pos_weight = "
        f"{scale_pos_weight:.4f}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "scale_pos_weight was calculated "
        "from training data only."
    )

    # --------------------------------------------------------
    # Create XGBoost pipeline
    # --------------------------------------------------------

    model = make_xgboost_model(
        numerical_columns,
        categorical_columns,
        scale_pos_weight,
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print(
        "\nModel configuration:"
    )

    print(
        "  n_estimators = 300"
    )

    print(
        "  max_depth = 5"
    )

    print(
        "  learning_rate = 0.05"
    )

    print(
        "  subsample = 0.8"
    )

    print(
        "  colsample_bytree = 0.8"
    )

    print(
        "  objective = binary:logistic"
    )

    print(
        "  eval_metric = aucpr"
    )

    print(
        "  random_state = 42"
    )

    print(
        "  tree_method = hist"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\nTraining XGBoost..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "XGBoost training complete."
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / "xgboost.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"Model saved: {model_path}"
    )

    print(
        "\nXGBoost validation metrics:"
    )

    print(
        f"  Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"  Recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"  F1: "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"  PR-AUC: "
        f"{metrics['pr_auc']:.4f}"
    )

    return model, metrics, scale_pos_weight


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # ========================================================
    # STEP 1
    # ========================================================

    df = load_data()

    print("=" * 60)
    print("AegisRisk AI — Day 5")
    print("Chronological ML Training")
    print("=" * 60)

    print(
        f"Dataset rows: {len(df):,}"
    )

    print(
        f"Dataset columns: {len(df.columns)}"
    )

    print(
        "\nDataset validation: PASSED"
    )

    # ========================================================
    # STEP 2
    # ========================================================

    df = sort_chronologically(df)

    print(
        "\nChronological ordering: PASSED"
    )

    # ========================================================
    # STEP 3
    # ========================================================

    train_df, validation_df = (
        chronological_split(df)
    )

    print(
        "\nChronological split:"
    )

    print(
        f"  Training: "
        f"{len(train_df):,}"
    )

    print(
        f"  Validation: "
        f"{len(validation_df):,}"
    )

    print(
        "\nTraining end:"
    )

    print(
        train_df[
            TIMESTAMP_COLUMN
        ].max()
    )

    print(
        "\nValidation start:"
    )

    print(
        validation_df[
            TIMESTAMP_COLUMN
        ].min()
    )

    # ========================================================
    # STEP 4
    # ========================================================

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
        numerical_columns,
        categorical_columns,
        feature_columns,
    ) = prepare_features(
        train_df,
        validation_df,
    )

    print(
        "\nFeature preparation: PASSED"
    )

    print(
        f"Model feature count: "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # STEP 5
    # ========================================================

    (
        logistic_model,
        logistic_metrics,
    ) = train_logistic_regression(
        X_train,
        y_train,
        X_validation,
        y_validation,
        numerical_columns,
        categorical_columns,
    )

    # ========================================================
    # STEP 6
    # ========================================================

    (
        random_forest_model,
        random_forest_metrics,
    ) = train_random_forest(
        X_train,
        y_train,
        X_validation,
        y_validation,
        numerical_columns,
        categorical_columns,
    )

    # ========================================================
    # STEP 7
    # ========================================================

    (
        xgboost_model,
        xgboost_metrics,
        scale_pos_weight,
    ) = train_xgboost(
        X_train,
        y_train,
        X_validation,
        y_validation,
        numerical_columns,
        categorical_columns,
    )

    # ========================================================
    # STEP 7 COMPLETION
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "DAY 5 — STEP 7 COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nModels trained:"
    )

    print(
        "  [✓] Logistic Regression"
    )

    print(
        "  [✓] Random Forest"
    )

    print(
        "  [✓] XGBoost"
    )

    print(
        "\nArtifacts:"
    )

    print(
        "  [✓] models/logistic_regression.joblib"
    )

    print(
        "  [✓] models/random_forest.joblib"
    )

    print(
        "  [✓] models/xgboost.joblib"
    )

    print(
        "\nMethodology:"
    )

    print(
        "  [✓] Chronological split"
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
        "  [✓] Validation used only for evaluation"
    )

    print(
        "  [✓] Threshold = 0.50"
    )

    print(
        "  [✓] Threshold tuning deferred to Day 6"
    )

    print(
        "\nNext:"
    )

    print(
        "Step 8 — Full model evaluation"
    )


if __name__ == "__main__":
    main()