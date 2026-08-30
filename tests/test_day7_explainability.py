from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# PROJECT ROOT / IMPORT PATH
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.risk.explainability import FraudExplainer


# ---------------------------------------------------------------------------
# FROZEN DAY 6 ARTIFACTS
# ---------------------------------------------------------------------------

MODEL_PATH = ROOT / "models" / "logistic_regression_day6.joblib"
DATA_PATH = ROOT / "data" / "processed" / "transactions_features.csv"


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def explainer() -> FraudExplainer:
    """
    Load the frozen Day 6 Logistic Regression model through
    the existing FraudExplainer implementation.
    """
    return FraudExplainer(MODEL_PATH)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """
    Load the actual processed repository dataset.
    """
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def sample(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Use a deterministic real sample from the repository dataset.
    """
    return dataset.head(5).copy()


# ---------------------------------------------------------------------------
# A. FROZEN MODEL FEATURE ORDER
# ---------------------------------------------------------------------------

def test_frozen_model_feature_order(
    explainer: FraudExplainer,
):
    """
    Verify that FraudExplainer loads the frozen Day 6 model and
    preserves the exact 26-feature input schema and order.
    """
    assert len(explainer.input_features) == 26

    assert list(
        explainer.pipeline.feature_names_in_
    ) == explainer.input_features


# ---------------------------------------------------------------------------
# B. TRANSFORMED FEATURE ALIGNMENT
# ---------------------------------------------------------------------------

def test_transformed_feature_alignment(
    explainer: FraudExplainer,
):
    """
    Verify that the frozen preprocessing pipeline produces
    exactly 29 transformed features and that their names align.
    """
    assert len(explainer.transformed_feature_names) == 29

    transformed = explainer.preprocessor.get_feature_names_out()

    assert len(transformed) == 29

    assert list(transformed) == (
        explainer.transformed_feature_names
    )


# ---------------------------------------------------------------------------
# C. SHAP EXPLANATION SHAPE
# ---------------------------------------------------------------------------

def test_shap_explanation_shape(
    explainer: FraudExplainer,
    sample: pd.DataFrame,
):
    """
    Generate actual explanations from repository data and verify
    that every explanation corresponds to all 29 transformed features.
    """
    explanations = explainer.explain_transaction(
        sample,
        top_k=29,
    )

    assert len(explanations) == len(sample)

    for explanation in explanations:
        assert explanation["feature_count_explained"] == 29

        assert len(
            explanation["top_contributors"]
        ) == 29

        assert explanation["top_k"] == 29


# ---------------------------------------------------------------------------
# D. CONTRIBUTION RANKING
# ---------------------------------------------------------------------------

def test_contribution_ranking(
    explainer: FraudExplainer,
    sample: pd.DataFrame,
):
    """
    Verify that contributors are sorted by descending absolute
    contribution magnitude.
    """
    explanations = explainer.explain_transaction(
        sample,
        top_k=29,
    )

    for explanation in explanations:
        contributors = explanation["top_contributors"]

        magnitudes = [
            item["magnitude"]
            for item in contributors
        ]

        assert magnitudes == sorted(
            magnitudes,
            reverse=True,
        )


# ---------------------------------------------------------------------------
# E. CONTRIBUTION DIRECTION
# ---------------------------------------------------------------------------

def test_contribution_direction():
    """
    Verify the existing contribution-direction mapping.
    """

    # Positive contribution
    assert (
        FraudExplainer._direction(1.0)
        == "INCREASES_FRAUD_RISK"
    )

    assert (
        FraudExplainer._direction(0.000001)
        == "INCREASES_FRAUD_RISK"
    )

    # Negative contribution
    assert (
        FraudExplainer._direction(-1.0)
        == "DECREASES_FRAUD_RISK"
    )

    assert (
        FraudExplainer._direction(-0.000001)
        == "DECREASES_FRAUD_RISK"
    )

    # Zero contribution
    assert (
        FraudExplainer._direction(0.0)
        == "NEUTRAL"
    )


# ---------------------------------------------------------------------------
# F. top_k VALIDATION
# ---------------------------------------------------------------------------

def test_top_k_zero_raises(
    explainer: FraudExplainer,
    sample: pd.DataFrame,
):
    """
    top_k=0 must be rejected.
    """
    with pytest.raises(ValueError):
        explainer.explain_transaction(
            sample,
            top_k=0,
        )


def test_negative_top_k_raises(
    explainer: FraudExplainer,
    sample: pd.DataFrame,
):
    """
    Negative top_k must be rejected.
    """
    with pytest.raises(ValueError):
        explainer.explain_transaction(
            sample,
            top_k=-1,
        )


# ---------------------------------------------------------------------------
# G. MISSING FEATURE VALIDATION
# ---------------------------------------------------------------------------

def test_missing_feature_validation(
    explainer: FraudExplainer,
    sample: pd.DataFrame,
):
    """
    Removing a required frozen model feature must raise ValueError.
    """
    required_feature = explainer.input_features[0]

    broken = sample.drop(
        columns=[required_feature]
    )

    with pytest.raises(
        ValueError,
        match="Missing required frozen model features",
    ):
        explainer.explain_transaction(broken)


# ---------------------------------------------------------------------------
# H. FEATURE VALUE MAPPING
# ---------------------------------------------------------------------------

def test_feature_value_mapping(
    explainer: FraudExplainer,
    sample: pd.DataFrame,
):
    """
    Verify that transformed feature names map back to the
    original source values.

    Numeric features should return their original source value.

    One-hot payment-method features should return the original
    payment_method string rather than a fabricated numeric value.
    """
    transaction = sample.iloc[0]

    explanations = explainer.explain_transaction(
        sample.head(1),
        top_k=29,
    )[0]["top_contributors"]

    by_name = {
        item["internal_feature"]: item
        for item in explanations
    }

    # -----------------------------------------------------------------------
    # Numeric transformed feature -> original source feature value
    # -----------------------------------------------------------------------

    numeric_feature = "numeric__amount"

    assert numeric_feature in by_name

    assert (
        by_name[numeric_feature]["feature_value"]
        == transaction["amount"]
    )

    # -----------------------------------------------------------------------
    # One-hot payment method -> original payment_method value
    # -----------------------------------------------------------------------

    payment_features = [
        item
        for item in explanations
        if item["internal_feature"].startswith(
            "categorical__payment_method_"
        )
    ]

    # The frozen preprocessing currently produces four payment-method
    # transformed features.
    assert len(payment_features) == 4

    for item in payment_features:
        assert (
            item["feature_value"]
            == transaction["payment_method"]
        )