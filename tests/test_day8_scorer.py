"""
AegisRisk AI - Day 8
Merchant Risk Scorer Tests

Tests the inference pipeline using frozen Day 6 model
and Day 7 explainability system.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.inference.scorer import MerchantRiskScorer


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def data_path() -> Path:
    """Path to processed dataset."""
    return Path("data/processed/transactions_features.csv")


@pytest.fixture(scope="module")
def sample_data(data_path: Path) -> pd.DataFrame:
    """Load small sample for testing."""
    if not data_path.exists():
        pytest.skip(f"Dataset not found: {data_path}")

    return pd.read_csv(data_path, nrows=50)


@pytest.fixture(scope="module")
def scorer() -> MerchantRiskScorer:
    """Initialize the scorer."""
    model_path = Path("models/logistic_regression_day6.joblib")

    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")

    return MerchantRiskScorer(model_path)


# ============================================================
# INITIALIZATION TESTS
# ============================================================

class TestScorerInitialization:
    """Test scorer initialization."""

    def test_scorer_loads_successfully(self, scorer: MerchantRiskScorer):
        """Verify scorer initializes."""
        assert scorer is not None
        assert scorer.model_path.exists()
        assert len(scorer.input_features) == 26

    def test_frozen_policy_loaded(self, scorer: MerchantRiskScorer):
        """Verify frozen policy is correct."""
        assert scorer.policy.review_threshold == 0.35
        assert scorer.policy.hold_threshold == 0.70

    def test_explainer_initialized(self, scorer: MerchantRiskScorer):
        """Verify SHAP explainer is ready."""
        assert scorer.explainer is not None
        assert len(scorer.explainer.input_features) == 26

    def test_metrics_available(self, scorer: MerchantRiskScorer):
        """Verify metrics can be retrieved."""
        metrics = scorer.get_metrics()
        assert "model_type" in metrics
        assert "input_features" in metrics
        assert metrics["input_features"] == 26


# ============================================================
# SCORING TESTS
# ============================================================

class TestScoring:
    """Test transaction scoring."""

    def test_score_empty_dataframe_raises_error(self, scorer: MerchantRiskScorer):
        """Empty DataFrame should raise ValueError."""
        with pytest.raises(ValueError):
            scorer.score_transactions(pd.DataFrame())

    def test_missing_features_raises_error(
        self,
        scorer: MerchantRiskScorer,
        sample_data: pd.DataFrame,
    ):
        """Missing required features should raise ValueError."""
        incomplete = sample_data.drop(columns=["amount"], errors="ignore")

        with pytest.raises(ValueError, match="Missing required features"):
            scorer.score_transactions(incomplete)

    def test_score_single_transaction(
        self,
        scorer: MerchantRiskScorer,
        sample_data: pd.DataFrame,
    ):
        """Score a single transaction."""
        single = sample_data.iloc[[0]]
        scored = scorer.score_transactions(single, include_explanation=False)

        assert len(scored) == 1
        assert "fraud_probability" in scored.columns
        assert "risk_decision" in scored.columns
        assert "risk_action" in scored.columns

    def test_score_multiple_transactions(
        self,
        scorer: MerchantRiskScorer,
        sample_data: pd.DataFrame,
    ):
        """Score multiple transactions."""
        scored = scorer.score_transactions(
            sample_data,
            include_explanation=False,
        )

        assert len(scored) == len(sample_data)
        assert scored["fraud_probability"].notna().all()
        assert scored["risk_decision"].notna().all()
        assert scored["risk_action"].notna().all()


# ============================================================
# OUTPUT VALIDATION TESTS
# ============================================================

class TestOutputValidation:
    """Test output correctness."""

    def test_fraud_probability_range(
        self,
        scorer: MerchantRiskScorer,
        sample_data: pd.DataFrame,
    ):
        """Fraud probabilities must be in [0, 1]."""
        scored = scorer.score_transactions(sample_data, include_explanation=False)

        assert (scored["fraud_probability"] >= 0).all()
        assert (scored["fraud_probability"] <= 1).all()

    def test_risk_decision_valid_values(
        self,
        scorer: MerchantRiskScorer,
        sample_data: pd.DataFrame,
    ):
        """Risk decisions must be valid."""
        scored = scorer.score_transactions(sample_data, include_explanation=False)

        valid_decisions = {"LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"}
        assert scored["risk_decision"].isin(valid_decisions).all()

    def test_risk_action_valid_values(
        self,
        scorer: MerchantRiskScorer,
        sample_data: pd.DataFrame,
    ):
        """Risk actions must be valid."""
        scored = scorer.score_transactions(sample_data, include_explanation=False)

        valid_actions = {"ALLOW", "REVIEW", "HOLD_FOR_VERIFICATION"}
        assert scored["risk_action"].isin(valid_actions).all()

    def test_threshold_alignment(
        self,
        scorer: MerchantRiskScorer,
        sample_data: pd.DataFrame,
    ):
        """Verify thresholds are correctly applied."""
        scored = scorer.score_transactions(sample_data, include_explanation=False)

        # LOW_RISK should have prob < 0.35
        low_risk = scored[scored["risk_decision"] == "LOW_RISK"]
        if len(low_risk) > 0:
            assert (low_risk["fraud_probability"] < 0.35).all()

        # MEDIUM_RISK should have 0.35 <= prob < 0.70
        med_risk = scored[scored["risk_decision"] == "MEDIUM_RISK"]
        if len(med_risk) > 0:
            assert (med_risk["fraud_probability"] >= 0.35).all()
            assert (med_risk["fraud_probability"] < 0.70).all()

        # HIGH_RISK should have prob >= 0.70
        high_risk = scored[scored["risk_decision"] == "HIGH_RISK"]
        if len(high_risk) > 0:
            assert (high_risk["fraud_probability"] >= 0.70).all()

    def test_decision_action_correspondence(
        self,
        scorer: MerchantRiskScorer,
        sample_data: pd.DataFrame,
    ):
        """Verify decision levels correspond to actions."""
        scored = scorer.score_transactions(sample_data, include_explanation=False)

        # LOW_RISK -> ALLOW
        low_risk = scored[scored["risk_decision"] == "LOW_RISK"]
        if len(low_risk) > 0:
            assert (low_risk["risk_action"] == "ALLOW").all()

        # MEDIUM_RISK -> REVIEW
        med_risk = scored[scored["risk_decision"] == "MEDIUM_RISK"]
        if len(med_risk) > 0:
            assert (med_risk["risk_action"] == "REVIEW").all()

        # HIGH_RISK -> HOLD_FOR_VERIFICATION
        high_risk = scored[scored["risk_decision"] == "HIGH_RISK"]
        if len(high_risk) > 0:
            assert (
                high_risk["risk_action"] == "HOLD_FOR_VERIFICATION"
            ).all()


# ============================================================
# EXPLANATION TESTS
# ============================================================

class TestExplanations:
    """Test SHAP explanations."""

    def test_explanations_generated(
        self,
        scorer: MerchantRiskScorer,
        sample_data: pd.DataFrame,
    ):
        """Verify explanations are generated."""
        scored = scorer.score_transactions(
            sample_data.head(5),
            include_explanation=True,
            top_k=5,
        )

        assert "top_contributors" in scored.columns
        assert "explanation_summary" in scored.columns

        # Each transaction should have contributors
        for contributors in scored["top_contributors"]:
            assert isinstance(contributors, list)
            assert len(contributors) <= 5

    def test_top_k_parameter(
        self,
        scorer: MerchantRiskScorer,
        sample_data: pd.DataFrame,
    ):
        """Verify top_k limits explanation features."""
        scored_k5 = scorer.score_transactions(
            sample_data.head(1),
            include_explanation=True,
            top_k=5,
        )

        scored_k3 = scorer.score_transactions(
            sample_data.head(1),
            include_explanation=True,
            top_k=3,
        )

        assert len(scored_k5.iloc[0]["top_contributors"]) <= 5
        assert len(scored_k3.iloc[0]["top_contributors"]) <= 3

    def test_contributor_structure(
        self,
        scorer: MerchantRiskScorer,
        sample_data: pd.DataFrame,
    ):
        """Verify contributor dict has required fields."""
        scored = scorer.score_transactions(
            sample_data.head(1),
            include_explanation=True,
            top_k=5,
        )

        contributors = scored.iloc[0]["top_contributors"]
        if contributors:
            for contrib in contributors:
                assert "internal_feature" in contrib
                assert "display_name" in contrib
                assert "direction" in contrib
                assert "contribution" in contrib
                assert "magnitude" in contrib


# ============================================================
# METADATA TESTS
# ============================================================

class TestMetadata:
    """Test metadata and tracking."""

    def test_last_scored_count_updated(
        self,
        scorer: MerchantRiskScorer,
        sample_data: pd.DataFrame,
    ):
        """Verify scoring count is tracked."""
        # Score a batch and verify count updates
        scored = scorer.score_transactions(sample_data.head(10), include_explanation=False)
        assert scorer.last_scored_count == 10

        # Score another batch and verify count updates to new value
        scored = scorer.score_transactions(sample_data.head(5), include_explanation=False)
        assert scorer.last_scored_count == 5

        # Score full batch and verify
        scored = scorer.score_transactions(sample_data, include_explanation=False)
        assert scorer.last_scored_count == len(sample_data)

    def test_original_data_preserved(
        self,
        scorer: MerchantRiskScorer,
        sample_data: pd.DataFrame,
    ):
        """Verify original columns are preserved."""
        original_cols = set(sample_data.columns)
        scored = scorer.score_transactions(sample_data.head(5), include_explanation=False)

        # All original columns should still be present
        for col in original_cols:
            assert col in scored.columns


# ============================================================
# FROZEN COMPONENT TESTS
# ============================================================

class TestFrozenComponents:
    """Verify frozen components are not modified."""

    def test_model_26_features_frozen(self, scorer: MerchantRiskScorer):
        """Verify model input features are exactly 26."""
        assert len(scorer.input_features) == 26
        assert scorer.input_features[0] == "amount"  # First feature check

    def test_policy_thresholds_frozen(self, scorer: MerchantRiskScorer):
        """Verify policy thresholds match Day 6 frozen values."""
        assert scorer.policy.review_threshold == 0.35
        assert scorer.policy.hold_threshold == 0.70

    def test_explainer_29_features(self, scorer: MerchantRiskScorer):
        """Verify explainer has exactly 29 transformed features."""
        assert len(scorer.explainer.transformed_feature_names) == 29
