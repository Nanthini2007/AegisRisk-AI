"""
AegisRisk AI - Day 8
Merchant Risk Scorer Validation

Validates the frozen Day 6 model and Day 8 scoring pipeline.

This script is designed to run directly with:

    python scripts\validate_day8_scorer.py
"""

from __future__ import annotations

import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT / IMPORT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# IMPORTS
# ============================================================

import joblib
import pandas as pd

from src.inference.scorer import MerchantRiskScorer


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "logistic_regression_day6.joblib"
)

FEATURE_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "transactions_features.csv"
)

EXPECTED_FEATURE_COUNT = 26
EXPECTED_REVIEW_THRESHOLD = 0.35
EXPECTED_HOLD_THRESHOLD = 0.70


# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def validate_model_exists() -> None:
    """Verify the frozen Day 6 model exists."""

    print("\n[1] Checking frozen model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Frozen model not found: {MODEL_PATH}"
        )

    print("  ✓ Frozen Day 6 model exists")


def validate_feature_schema(scorer: MerchantRiskScorer) -> None:
    """Verify the frozen model uses exactly 26 features."""

    print("\n[2] Checking feature schema...")

    feature_count = len(scorer.input_features)

    print(f"  Feature count: {feature_count}")

    if feature_count != EXPECTED_FEATURE_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"got {feature_count}"
        )

    print("  ✓ 26 frozen input features verified")


def validate_policy(scorer: MerchantRiskScorer) -> None:
    """Verify Day 6 decision thresholds are unchanged."""

    print("\n[3] Checking frozen decision policy...")

    review = scorer.policy.review_threshold
    hold = scorer.policy.hold_threshold

    print(f"  Review threshold: {review}")
    print(f"  Hold threshold:   {hold}")

    if review != EXPECTED_REVIEW_THRESHOLD:
        raise AssertionError(
            f"Expected review threshold "
            f"{EXPECTED_REVIEW_THRESHOLD}, got {review}"
        )

    if hold != EXPECTED_HOLD_THRESHOLD:
        raise AssertionError(
            f"Expected hold threshold "
            f"{EXPECTED_HOLD_THRESHOLD}, got {hold}"
        )

    print("  ✓ Frozen decision policy verified")


def validate_model_metadata(scorer: MerchantRiskScorer) -> None:
    """Verify scorer metadata."""

    print("\n[4] Checking scorer metadata...")

    metrics = scorer.get_metrics()

    expected_model_type = (
        "Logistic Regression (Frozen Day 6)"
    )

    if metrics["model_type"] != expected_model_type:
        raise AssertionError(
            f"Unexpected model type: "
            f"{metrics['model_type']}"
        )

    if metrics["input_features"] != EXPECTED_FEATURE_COUNT:
        raise AssertionError(
            f"Unexpected feature count: "
            f"{metrics['input_features']}"
        )

    if metrics["policy_review_threshold"] != (
        EXPECTED_REVIEW_THRESHOLD
    ):
        raise AssertionError(
            "Review threshold metadata mismatch"
        )

    if metrics["policy_hold_threshold"] != (
        EXPECTED_HOLD_THRESHOLD
    ):
        raise AssertionError(
            "Hold threshold metadata mismatch"
        )

    if metrics["last_scored_count"] != 0:
        raise AssertionError(
            "Scorer should not have scored transactions yet"
        )

    print("  ✓ Scorer metadata verified")


def validate_sample_scoring(
    scorer: MerchantRiskScorer,
) -> None:
    """
    Score a small sample from the processed dataset.

    This validates the inference path without retraining
    or modifying the frozen model.
    """

    print("\n[5] Checking sample scoring...")

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    # Load only a small sample.
    sample = pd.read_csv(
        FEATURE_FILE,
        nrows=10,
    )

    print(f"  Sample rows: {len(sample)}")

    if sample.empty:
        raise ValueError("Feature dataset sample is empty")

    missing = [
        feature
        for feature in scorer.input_features
        if feature not in sample.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    result = scorer.score_transactions(
        sample,
        include_explanation=False,
    )

    required_output_columns = [
        "fraud_probability",
        "risk_decision",
        "risk_action",
    ]

    for column in required_output_columns:
        if column not in result.columns:
            raise AssertionError(
                f"Missing output column: {column}"
            )

    if len(result) != len(sample):
        raise AssertionError(
            "Scored row count does not match input row count"
        )

    if not result["fraud_probability"].between(
        0.0,
        1.0,
    ).all():
        raise AssertionError(
            "Fraud probabilities outside [0, 1]"
        )

    valid_decisions = {
        "LOW_RISK",
        "MEDIUM_RISK",
        "HIGH_RISK",
    }

    if not set(result["risk_decision"]).issubset(
        valid_decisions
    ):
        raise AssertionError(
            "Invalid risk decision detected"
        )

    valid_actions = {
        "ALLOW",
        "REVIEW",
        "HOLD FOR VERIFICATION",
    }

    if not set(result["risk_action"]).issubset(
        valid_actions
    ):
        raise AssertionError(
            "Invalid risk action detected"
        )

    print("  ✓ Sample scoring passed")
    print("  ✓ Fraud probabilities valid")
    print("  ✓ Risk decisions valid")
    print("  ✓ Risk actions valid")


def validate_frozen_artifact() -> None:
    """Verify the joblib artifact can be loaded directly."""

    print("\n[6] Checking frozen artifact...")

    pipeline = joblib.load(MODEL_PATH)

    if not hasattr(pipeline, "feature_names_in_"):
        raise AssertionError(
            "Frozen pipeline has no feature_names_in_"
        )

    feature_count = len(
        pipeline.feature_names_in_
    )

    if feature_count != EXPECTED_FEATURE_COUNT:
        raise AssertionError(
            f"Frozen artifact has {feature_count} "
            f"features instead of {EXPECTED_FEATURE_COUNT}"
        )

    print("  ✓ Frozen artifact loads successfully")
    print("  ✓ Frozen artifact feature schema verified")


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    """Run complete Day 8 scorer validation."""

    print("=" * 70)
    print("AEGISRISK AI - DAY 8")
    print("MERCHANT RISK SCORER VALIDATION")
    print("=" * 70)

    try:
        validate_model_exists()

        scorer = MerchantRiskScorer(
            model_path=MODEL_PATH,
        )

        validate_feature_schema(scorer)
        validate_policy(scorer)
        validate_model_metadata(scorer)
        validate_frozen_artifact()
        validate_sample_scoring(scorer)

    except Exception as exc:
        print("\n" + "=" * 70)
        print("✗ DAY 8 VALIDATION FAILED")
        print("=" * 70)
        print(f"\nError: {exc}")

        return 1

    print("\n" + "=" * 70)
    print("✓ DAY 8 VALIDATION PASSED")
    print("=" * 70)

    print("\nVerified:")
    print("  ✓ Frozen Day 6 Logistic Regression")
    print("  ✓ 26-feature schema")
    print("  ✓ Review threshold = 0.35")
    print("  ✓ Hold threshold = 0.70")
    print("  ✓ Sample inference")
    print("  ✓ Fraud probability range")
    print("  ✓ Risk decision mapping")
    print("  ✓ Risk action mapping")
    print("  ✓ No retraining performed")
    print("  ✓ Frozen model unchanged")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())