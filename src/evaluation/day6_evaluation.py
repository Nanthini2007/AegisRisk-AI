"""
AegisRisk AI
Day 6 - Cost-Aware Fraud Evaluation

Methodology:
    - 70/15/15 chronological split
    - Logistic Regression architecture frozen from Day 5
    - Validation data used for threshold selection only
    - Held-out test data not used for threshold selection
    - Experimental cost assumptions are configurable
    - Three-level merchant decision policy:
        LOW RISK -> ALLOW
        MEDIUM RISK -> REVIEW
        HIGH RISK -> HOLD FOR VERIFICATION
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
)

from src.evaluation.cost_model import (
    CostModel,
    cost_model_to_dict,
)
from src.evaluation.decision_policy import (
    DecisionPolicy,
    decide_many,
    policy_to_dict,
)
from src.evaluation.splits import (
    assert_temporal_separation,
    chronological_train_validation_test_split,
)
from src.evaluation.threshold_selection import (
    evaluate_threshold,
    evaluate_thresholds,
    select_lowest_cost_threshold,
)
from src.models.train_models import (
    TARGET,
    TIMESTAMP_COLUMN,
    load_data,
    make_logistic_model,
    prepare_features,
    sort_chronologically,
)


# ============================================================
# PROJECT PATHS
# ============================================================

REPORT_DIR = Path("reports")

MODEL_DIR = Path("models")


# ============================================================
# DAY 6 EXPERIMENT CONFIGURATION
# ============================================================

# Multiple probability thresholds evaluated on VALIDATION ONLY.
THRESHOLDS = [
    round(value, 2)
    for value in np.arange(
        0.05,
        0.96,
        0.05,
    )
]


# Experimental assumption only.
# This does NOT represent Razorpay production economics.
FRAUD_MISS_COST = 1000.0


# Sensitivity analysis only.
FALSE_POSITIVE_COSTS = [
    25.0,
    50.0,
    100.0,
]


# Primary documented experimental assumption.
PRIMARY_FALSE_POSITIVE_COST = 50.0


# ============================================================
# JSON UTILITIES
# ============================================================

def make_json_safe(value):
    """
    Convert NumPy and Path values into JSON-safe
    Python values.
    """

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, Path):
        return str(value)

    return value


def save_json(
    path: Path,
    data: dict,
) -> None:
    """Save a formatted JSON report."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_safe(data),
            file,
            indent=2,
        )


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_day6_features(
    train_df,
    validation_df,
    test_df,
):
    """
    Prepare Day 6 model features.

    The feature schema is derived from the
    training data using the existing Day 5
    leakage-safe feature preparation logic.

    The exact resulting feature schema is then
    applied to the validation and held-out test
    periods.
    """

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

    missing_features = [
        column
        for column in feature_columns
        if column not in test_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Held-out test data is missing "
            "model features: "
            f"{missing_features}"
        )

    X_test = test_df[
        feature_columns
    ].copy()

    y_test = test_df[
        TARGET
    ].astype(int)

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        numerical_columns,
        categorical_columns,
        feature_columns,
    )


# ============================================================
# FINAL HELD-OUT TEST EVALUATION
# ============================================================

def evaluate_final_test(
    y_true,
    probabilities,
    threshold: float,
    cost_model: CostModel,
) -> dict:
    """
    Evaluate the held-out test set using an
    already frozen threshold.

    IMPORTANT:
    This function does NOT select or optimize
    the threshold.
    """

    threshold_metrics = evaluate_threshold(
        y_true=y_true,
        probabilities=probabilities,
        threshold=threshold,
        cost_model=cost_model,
    )

    predictions = (
        np.asarray(probabilities)
        >= threshold
    ).astype(int)

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    return {
        **threshold_metrics,
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# MAIN DAY 6 WORKFLOW
# ============================================================

def main() -> None:

    print("=" * 70)
    print("AegisRisk AI - DAY 6")
    print("Cost-Aware Fraud Evaluation")
    print("=" * 70)

    # ========================================================
    # STEP 1 - LOAD AND SORT DATA
    # ========================================================

    df = load_data()

    df = sort_chronologically(df)

    print(
        f"\nDataset rows: {len(df):,}"
    )

    # ========================================================
    # STEP 2 - 70/15/15 CHRONOLOGICAL SPLIT
    # ========================================================

    (
        train_df,
        validation_df,
        test_df,
    ) = chronological_train_validation_test_split(
        df
    )

    assert_temporal_separation(
        train_df,
        validation_df,
        test_df,
        timestamp_column=TIMESTAMP_COLUMN,
    )

    print(
        "\nChronological split: PASSED"
    )

    print(
        f"  Train:      "
        f"{len(train_df):,}"
    )

    print(
        f"  Validation: "
        f"{len(validation_df):,}"
    )

    print(
        f"  Test:       "
        f"{len(test_df):,}"
    )

    print(
        "\nTime boundaries:"
    )

    print(
        f"  Train:      "
        f"{train_df[TIMESTAMP_COLUMN].min()} -> "
        f"{train_df[TIMESTAMP_COLUMN].max()}"
    )

    print(
        f"  Validation: "
        f"{validation_df[TIMESTAMP_COLUMN].min()} -> "
        f"{validation_df[TIMESTAMP_COLUMN].max()}"
    )

    print(
        f"  Test:       "
        f"{test_df[TIMESTAMP_COLUMN].min()} -> "
        f"{test_df[TIMESTAMP_COLUMN].max()}"
    )

    # ========================================================
    # STEP 3 - PREPARE FROZEN FEATURE SCHEMA
    # ========================================================

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        numerical_columns,
        categorical_columns,
        feature_columns,
    ) = prepare_day6_features(
        train_df,
        validation_df,
        test_df,
    )

    print(
        "\nFeature schema: PASSED"
    )

    print(
        f"  Feature count: "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # STEP 4 - TRAIN FROZEN LOGISTIC REGRESSION
    # ========================================================

    print(
        "\nTraining frozen Logistic "
        "Regression architecture..."
    )

    model = make_logistic_model(
        numerical_columns,
        categorical_columns,
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # ========================================================
    # STEP 5 - GENERATE VALIDATION PROBABILITIES
    # ========================================================

    print(
        "\nGenerating validation probabilities..."
    )

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    if (
        validation_probabilities.min() < 0
        or validation_probabilities.max() > 1
    ):
        raise ValueError(
            "Validation probabilities are "
            "outside [0, 1]."
        )

    # ========================================================
    # STEP 6 - VALIDATION-ONLY COST SENSITIVITY ANALYSIS
    # ========================================================

    sensitivity_results = {}

    for false_positive_cost in (
        FALSE_POSITIVE_COSTS
    ):
        sensitivity_cost_model = CostModel(
            fraud_miss_cost=FRAUD_MISS_COST,
            false_positive_cost=(
                false_positive_cost
            ),
        )

        threshold_results = (
            evaluate_thresholds(
                y_true=y_validation,
                probabilities=(
                    validation_probabilities
                ),
                thresholds=THRESHOLDS,
                cost_model=(
                    sensitivity_cost_model
                ),
            )
        )

        selected_result = (
            select_lowest_cost_threshold(
                threshold_results
            )
        )

        sensitivity_results[
            str(false_positive_cost)
        ] = {
            "cost_assumptions": dict(
                cost_model_to_dict(
                    sensitivity_cost_model
                )
            ),
            "threshold_results": (
                threshold_results
            ),
            "selected_threshold": (
                selected_result
            ),
        }

    # ========================================================
    # STEP 7 - PRIMARY VALIDATION-ONLY THRESHOLD SELECTION
    # ========================================================

    primary_cost_model = CostModel(
        fraud_miss_cost=FRAUD_MISS_COST,
        false_positive_cost=(
            PRIMARY_FALSE_POSITIVE_COST
        ),
    )

    primary_results = evaluate_thresholds(
        y_true=y_validation,
        probabilities=validation_probabilities,
        thresholds=THRESHOLDS,
        cost_model=primary_cost_model,
    )

    selected_validation_result = (
        select_lowest_cost_threshold(
            primary_results
        )
    )

    frozen_threshold = float(
        selected_validation_result[
            "threshold"
        ]
    )

    print(
        "\nVALIDATION-ONLY threshold "
        "selection complete."
    )

    print(
        f"  Frozen threshold: "
        f"{frozen_threshold:.2f}"
    )

    print(
        f"  Validation total experimental "
        f"cost: "
        f"{selected_validation_result['total_experimental_cost']:.2f}"
    )

    # ========================================================
    # STEP 8 - FREEZE POLICY BEFORE TEST EVALUATION
    # ========================================================

    frozen_policy = {
        "model_name": (
            "Logistic Regression"
        ),
        "threshold": frozen_threshold,
        "cost_assumptions": dict(
            cost_model_to_dict(
                primary_cost_model
            )
        ),
        "selection_data": (
            "validation_only"
        ),
        "test_set_used_for_selection": False,
    }

    save_json(
        REPORT_DIR
        / "day6_frozen_policy.json",
        frozen_policy,
    )

    # ========================================================
    # STEP 9 - HELD-OUT TEST EVALUATION
    # ========================================================

    print(
        "\nAccessing held-out test for "
        "final evaluation..."
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    final_test_metrics = (
        evaluate_final_test(
            y_true=y_test,
            probabilities=test_probabilities,
            threshold=frozen_threshold,
            cost_model=primary_cost_model,
        )
    )

    print(
        "Held-out test evaluation complete."
    )

    # ========================================================
    # STEP 10 - THREE-LEVEL MERCHANT DECISION POLICY
    # ========================================================

    # REVIEW boundary is an explicit configurable
    # experimental operational policy assumption.
    #
    # HOLD boundary is the validation-selected
    # frozen fraud operating threshold.
    decision_policy = DecisionPolicy(
        review_threshold=(
            frozen_threshold / 2
        ),
        hold_threshold=frozen_threshold,
    )

    test_decisions = decide_many(
        test_probabilities,
        decision_policy,
    )

    decision_counts = {
        "ALLOW": test_decisions.count(
            "ALLOW"
        ),
        "REVIEW": test_decisions.count(
            "REVIEW"
        ),
        "HOLD FOR VERIFICATION": (
            test_decisions.count(
                "HOLD FOR VERIFICATION"
            )
        ),
    }

    if (
        sum(decision_counts.values())
        != len(test_df)
    ):
        raise ValueError(
            "Decision counts do not match "
            "held-out test row count."
        )

    print(
        "\nDecision policy:"
    )

    print(
        f"  Review threshold: "
        f"{decision_policy.review_threshold:.2f}"
    )

    print(
        f"  Hold threshold: "
        f"{decision_policy.hold_threshold:.2f}"
    )

    print(
        f"  ALLOW: "
        f"{decision_counts['ALLOW']:,}"
    )

    print(
        f"  REVIEW: "
        f"{decision_counts['REVIEW']:,}"
    )

    print(
        f"  HOLD FOR VERIFICATION: "
        f"{decision_counts['HOLD FOR VERIFICATION']:,}"
    )

    # ========================================================
    # STEP 11 - SAVE DAY 6 MODEL
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / "logistic_regression_day6.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    # ========================================================
    # STEP 12 - BUILD REPORTS
    # ========================================================

    split_report = {
        "methodology": (
            "chronological_70_15_15"
        ),
        "total_rows": len(df),
        "train": {
            "rows": len(train_df),
            "start": str(
                train_df[
                    TIMESTAMP_COLUMN
                ].min()
            ),
            "end": str(
                train_df[
                    TIMESTAMP_COLUMN
                ].max()
            ),
        },
        "validation": {
            "rows": len(validation_df),
            "start": str(
                validation_df[
                    TIMESTAMP_COLUMN
                ].min()
            ),
            "end": str(
                validation_df[
                    TIMESTAMP_COLUMN
                ].max()
            ),
        },
        "held_out_test": {
            "rows": len(test_df),
            "start": str(
                test_df[
                    TIMESTAMP_COLUMN
                ].min()
            ),
            "end": str(
                test_df[
                    TIMESTAMP_COLUMN
                ].max()
            ),
            "used_for_threshold_selection": False,
        },
        "temporal_separation_verified": True,
    }

    validation_report = {
        "stage": "validation_only",
        "model_name": (
            "Logistic Regression"
        ),
        "thresholds_evaluated": THRESHOLDS,
        "primary_cost_assumptions": dict(
            cost_model_to_dict(
                primary_cost_model
            )
        ),
        "selected_threshold": (
            frozen_threshold
        ),
        "selected_result": (
            selected_validation_result
        ),
        "sensitivity_analysis": (
            sensitivity_results
        ),
        "held_out_test_used": False,
    }

    final_report = {
        "stage": (
            "final_held_out_test_evaluation"
        ),
        "model_name": (
            "Logistic Regression"
        ),
        "feature_count": len(
            feature_columns
        ),
        "feature_columns": (
            feature_columns
        ),
        "frozen_threshold": (
            frozen_threshold
        ),
        "threshold_selected_on": (
            "validation_only"
        ),
        "held_out_test_used_for_threshold_selection": (
            False
        ),
        "cost_assumptions": dict(
            cost_model_to_dict(
                primary_cost_model
            )
        ),
        "metrics": (
            final_test_metrics
        ),
        "decision_policy": (
            policy_to_dict(
                decision_policy
            )
        ),
        "held_out_test_decision_counts": (
            decision_counts
        ),
        "limitations": [
            (
                "Dataset is synthetic."
            ),
            (
                "Cost values are experimental "
                "assumptions."
            ),
            (
                "The review threshold is an "
                "experimental operational policy "
                "assumption."
            ),
            (
                "Results do not represent "
                "Razorpay production performance."
            ),
            (
                "No production fraud or merchant "
                "data was used."
            ),
            (
                "This evaluation does not establish "
                "production readiness."
            ),
        ],
    }

    # ========================================================
    # STEP 13 - SAVE REPORTS
    # ========================================================

    save_json(
        REPORT_DIR
        / "day6_split_report.json",
        split_report,
    )

    save_json(
        REPORT_DIR
        / "day6_validation_threshold_report.json",
        validation_report,
    )

    save_json(
        REPORT_DIR
        / "day6_final_test_report.json",
        final_report,
    )

    # ========================================================
    # FINAL CONSOLE SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "DAY 6 CORE EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nFrozen threshold: "
        f"{frozen_threshold:.2f}"
    )

    print(
        "\nHeld-out test metrics:"
    )

    print(
        f"  Precision: "
        f"{final_test_metrics['precision']:.4f}"
    )

    print(
        f"  Recall:    "
        f"{final_test_metrics['recall']:.4f}"
    )

    print(
        f"  F1:        "
        f"{final_test_metrics['f1']:.4f}"
    )

    print(
        f"  PR-AUC:    "
        f"{final_test_metrics['pr_auc']:.4f}"
    )

    print(
        f"  TP: "
        f"{final_test_metrics['true_positives']}"
    )

    print(
        f"  FP: "
        f"{final_test_metrics['false_positives']}"
    )

    print(
        f"  TN: "
        f"{final_test_metrics['true_negatives']}"
    )

    print(
        f"  FN: "
        f"{final_test_metrics['false_negatives']}"
    )

    print(
        f"  Fraud miss cost: "
        f"{final_test_metrics['fraud_miss_cost']:.2f}"
    )

    print(
        f"  False-positive cost: "
        f"{final_test_metrics['false_positive_cost']:.2f}"
    )

    print(
        f"  Total experimental cost: "
        f"{final_test_metrics['total_experimental_cost']:.2f}"
    )

    print(
        "\nHeld-out test decision counts:"
    )

    print(
        f"  ALLOW: "
        f"{decision_counts['ALLOW']:,}"
    )

    print(
        f"  REVIEW: "
        f"{decision_counts['REVIEW']:,}"
    )

    print(
        f"  HOLD FOR VERIFICATION: "
        f"{decision_counts['HOLD FOR VERIFICATION']:,}"
    )

    print(
        "\nReports saved:"
    )

    print(
        "  reports/day6_split_report.json"
    )

    print(
        "  reports/day6_validation_threshold_report.json"
    )

    print(
        "  reports/day6_frozen_policy.json"
    )

    print(
        "  reports/day6_final_test_report.json"
    )

    print(
        "\nModel saved:"
    )

    print(
        "  models/logistic_regression_day6.joblib"
    )


if __name__ == "__main__":
    main()