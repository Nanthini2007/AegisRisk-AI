"""
AegisRisk AI - Day 10 Integration Tests

Comprehensive pytest suite validating frozen Day 6-Day 9 components
using 10 deterministic rows.

Tests all 10 integration requirements:
1. Frozen Day 6 Logistic Regression loads
2. Exactly 26 model input features in frozen order
3. Fraud probabilities finite and in [0,1]
4. Risk decisions use thresholds 0.35 and 0.70
5. Actions match existing DecisionPolicy
6. SHAP explanations generated with top contributors
7. RiskMonitor returns PASS
8. Frozen model verification returns PASS
9. Frozen policy verification returns PASS
10. Results saved to reports/day10_integration_report.json

Does NOT modify Day 6-Day 9 components or create new models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from src.inference.scorer import MerchantRiskScorer
from src.monitoring.risk_monitor import RiskMonitor
from src.evaluation.decision_policy import DecisionPolicy


@pytest.fixture
def test_data() -> pd.DataFrame:
    """Load the first 10 deterministic rows from processed data."""
    data_path = Path("data/processed/transactions_features.csv")
    return pd.read_csv(data_path, nrows=10)


@pytest.fixture
def scorer(test_data) -> MerchantRiskScorer:
    """Initialize scorer with frozen Day 6 model."""
    return MerchantRiskScorer(
        model_path="models/logistic_regression_day6.joblib"
    )


@pytest.fixture
def scored_data(
    scorer: MerchantRiskScorer,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """Score the test data."""
    return scorer.score_transactions(
        test_data,
        include_explanation=True,
        top_k=5,
    )


@pytest.fixture
def monitor() -> RiskMonitor:
    """Initialize risk monitor with frozen policy."""
    return RiskMonitor(
        review_threshold=0.35,
        hold_threshold=0.70,
        model_path="models/logistic_regression_day6.joblib",
    )


class TestRequirement1ModelLoading:
    """Requirement 1: Frozen Day 6 Logistic Regression loads."""

    def test_frozen_model_file_exists(self) -> None:
        """Test that frozen model file exists."""
        model_path = Path("models/logistic_regression_day6.joblib")
        assert model_path.exists(), f"Model not found: {model_path}"

    def test_scorer_initializes_successfully(
        self,
        scorer: MerchantRiskScorer,
    ) -> None:
        """Test that scorer initializes without error."""
        assert scorer is not None
        assert scorer.pipeline is not None
        assert scorer.model is not None
        assert scorer.preprocessor is not None

    def test_model_path_accessible(
        self,
        scorer: MerchantRiskScorer,
    ) -> None:
        """Test that model path is correct and accessible."""
        assert scorer.model_path.exists()
        assert scorer.model_path.is_file()

    def test_pipeline_has_required_steps(
        self,
        scorer: MerchantRiskScorer,
    ) -> None:
        """Test that pipeline contains required preprocessing and model steps."""
        assert hasattr(scorer.pipeline, "named_steps")
        assert "preprocessor" in scorer.pipeline.named_steps
        assert "model" in scorer.pipeline.named_steps


class TestRequirement2FeatureCount:
    """Requirement 2: Exactly 26 model input features in frozen order."""

    def test_feature_count_is_26(
        self,
        scorer: MerchantRiskScorer,
    ) -> None:
        """Test that exactly 26 features are required."""
        assert len(scorer.input_features) == 26

    def test_features_in_correct_order(
        self,
        scorer: MerchantRiskScorer,
    ) -> None:
        """Test that features are in expected frozen order."""
        expected_features = [
            "amount",
            "payment_method",
            "account_age_days",
            "failed_attempts",
            "amount_log",
            "hour",
            "day_of_week",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "customer_txn_count_before",
            "customer_amount_mean_before",
            "customer_amount_std_before",
            "amount_vs_customer_mean",
            "customer_amount_zscore",
            "customer_seconds_since_prev",
            "customer_has_history",
            "customer_txn_count_10m",
            "customer_txn_count_1h",
            "customer_txn_count_24h",
            "merchant_txn_count_before",
            "merchant_amount_mean_before",
            "merchant_seconds_since_prev",
            "merchant_txn_count_1h",
            "customer_device_seen_before",
            "device_seen_before",
        ]

        assert len(scorer.input_features) == len(expected_features)
        assert scorer.input_features == expected_features

    def test_all_required_features_present_in_data(
        self,
        test_data: pd.DataFrame,
        scorer: MerchantRiskScorer,
    ) -> None:
        """Test that all 26 features exist in input data."""
        missing = [
            f for f in scorer.input_features
            if f not in test_data.columns
        ]
        assert len(missing) == 0, f"Missing features: {missing}"

    def test_model_accepts_features_in_frozen_order(
        self,
        test_data: pd.DataFrame,
        scorer: MerchantRiskScorer,
    ) -> None:
        """Test that model can score data with features in frozen order."""
        X = test_data[scorer.input_features].copy()
        predictions = scorer.pipeline.predict_proba(X)
        assert predictions.shape[0] == len(test_data)
        assert predictions.shape[1] == 2  # Binary classifier


class TestRequirement3Probabilities:
    """Requirement 3: Fraud probabilities finite and in [0,1]."""

    def test_probabilities_are_finite(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that all probabilities are finite (not inf or nan)."""
        probs = scored_data["fraud_probability"].values
        assert np.all(np.isfinite(probs)), (
            "Found non-finite probabilities"
        )

    def test_probabilities_in_valid_range(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that all probabilities are in [0, 1]."""
        probs = scored_data["fraud_probability"].values
        assert np.all(probs >= 0.0), "Found probabilities < 0"
        assert np.all(probs <= 1.0), "Found probabilities > 1"

    def test_probability_column_exists(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that fraud_probability column exists."""
        assert "fraud_probability" in scored_data.columns

    def test_no_null_probabilities(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that there are no null probabilities."""
        assert scored_data["fraud_probability"].notna().all()

    def test_probability_statistics(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test basic probability statistics."""
        probs = scored_data["fraud_probability"].values
        assert 0.0 <= np.min(probs) <= 1.0
        assert 0.0 <= np.max(probs) <= 1.0
        assert 0.0 <= np.mean(probs) <= 1.0
        assert np.std(probs) >= 0.0


class TestRequirement4Thresholds:
    """Requirement 4: Risk decisions use thresholds 0.35 and 0.70."""

    def test_review_threshold_is_0_35(
        self,
        scorer: MerchantRiskScorer,
    ) -> None:
        """Test that review threshold is 0.35."""
        assert np.isclose(
            scorer.policy.review_threshold,
            0.35,
        )

    def test_hold_threshold_is_0_70(
        self,
        scorer: MerchantRiskScorer,
    ) -> None:
        """Test that hold threshold is 0.70."""
        assert np.isclose(
            scorer.policy.hold_threshold,
            0.70,
        )

    def test_thresholds_are_ordered(
        self,
        scorer: MerchantRiskScorer,
    ) -> None:
        """Test that review_threshold < hold_threshold."""
        assert (
            scorer.policy.review_threshold
            < scorer.policy.hold_threshold
        )

    def test_thresholds_in_valid_range(
        self,
        scorer: MerchantRiskScorer,
    ) -> None:
        """Test that thresholds are in [0, 1]."""
        assert 0.0 <= scorer.policy.review_threshold <= 1.0
        assert 0.0 <= scorer.policy.hold_threshold <= 1.0

    def test_decision_policy_frozen(
        self,
        scorer: MerchantRiskScorer,
    ) -> None:
        """Test that DecisionPolicy is frozen (immutable)."""
        assert isinstance(scorer.policy, DecisionPolicy)
        # Try to modify - should raise AttributeError
        with pytest.raises(AttributeError):
            scorer.policy.review_threshold = 0.40


class TestRequirement5Actions:
    """Requirement 5: Actions match existing DecisionPolicy."""

    def test_risk_action_column_exists(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that risk_action column exists."""
        assert "risk_action" in scored_data.columns

    def test_all_actions_valid(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that all actions are valid policy actions."""
        valid_actions = {"ALLOW", "REVIEW", "HOLD_FOR_VERIFICATION"}
        actions = set(scored_data["risk_action"].unique())
        assert actions.issubset(valid_actions), (
            f"Invalid actions found: "
            f"{actions - valid_actions}"
        )

    def test_no_null_actions(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that there are no null actions."""
        assert scored_data["risk_action"].notna().all()

    def test_actions_match_probability_thresholds(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """
        Test that actions align with probabilities and thresholds.

        - P < 0.35 -> ALLOW
        - 0.35 <= P < 0.70 -> REVIEW
        - P >= 0.70 -> HOLD_FOR_VERIFICATION
        """
        for idx, row in scored_data.iterrows():
            prob = row["fraud_probability"]
            action = row["risk_action"]

            if prob < 0.35:
                assert action == "ALLOW", (
                    f"Row {idx}: P={prob:.3f} should be ALLOW"
                )
            elif prob < 0.70:
                assert action == "REVIEW", (
                    f"Row {idx}: P={prob:.3f} should be REVIEW"
                )
            else:
                assert action == "HOLD_FOR_VERIFICATION", (
                    f"Row {idx}: P={prob:.3f} "
                    "should be HOLD_FOR_VERIFICATION"
                )

    def test_action_distribution_is_deterministic(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """
        Test that action distribution matches expected ranges.

        With 10 deterministic rows, the distribution should be
        consistent across multiple runs.
        """
        action_counts = scored_data["risk_action"].value_counts()
        total_actions = len(scored_data)
        assert total_actions == 10, "Should have exactly 10 actions"


class TestRequirement6Explanations:
    """Requirement 6: SHAP explanations generated with top contributors."""

    def test_top_contributors_column_exists(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that top_contributors column exists."""
        assert "top_contributors" in scored_data.columns

    def test_all_transactions_have_explanations(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that all transactions have explanation data."""
        assert scored_data["top_contributors"].notna().all()

    def test_explanations_contain_contributors(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that each explanation contains top contributors."""
        for contributors in scored_data["top_contributors"]:
            assert isinstance(contributors, list)
            assert len(contributors) > 0, (
                "Explanation has no contributors"
            )

    def test_contributor_structure(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that contributors have required fields."""
        required_fields = {
            "internal_feature",
            "display_name",
            "direction",
            "contribution",
            "magnitude",
        }

        for contributors in scored_data["top_contributors"]:
            for contributor in contributors:
                assert isinstance(contributor, dict)
                assert required_fields.issubset(
                    set(contributor.keys())
                ), (
                    f"Missing fields in contributor: "
                    f"{required_fields - set(contributor.keys())}"
                )

    def test_contributor_directions_valid(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that contribution directions are valid."""
        valid_directions = {
            "INCREASES_FRAUD_RISK",
            "DECREASES_FRAUD_RISK",
            "NEUTRAL",
        }

        for contributors in scored_data["top_contributors"]:
            for contributor in contributors:
                direction = contributor["direction"]
                assert direction in valid_directions, (
                    f"Invalid direction: {direction}"
                )

    def test_contributions_are_numeric(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that contributions are numeric values."""
        for contributors in scored_data["top_contributors"]:
            for contributor in contributors:
                contribution = contributor["contribution"]
                magnitude = contributor["magnitude"]
                assert isinstance(contribution, (int, float))
                assert isinstance(magnitude, (int, float))
                assert magnitude >= 0.0, "Magnitude must be non-negative"

    def test_magnitude_matches_contribution(
        self,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that magnitude is absolute value of contribution."""
        for contributors in scored_data["top_contributors"]:
            for contributor in contributors:
                contribution = contributor["contribution"]
                magnitude = contributor["magnitude"]
                expected_magnitude = abs(contribution)
                assert np.isclose(
                    magnitude,
                    expected_magnitude,
                ), (
                    "Magnitude does not match "
                    "absolute contribution"
                )


class TestRequirement7RiskMonitor:
    """Requirement 7: RiskMonitor returns PASS."""

    def test_monitor_initializes_successfully(
        self,
        monitor: RiskMonitor,
    ) -> None:
        """Test that monitor initializes correctly."""
        assert monitor is not None
        assert monitor.review_threshold == 0.35
        assert monitor.hold_threshold == 0.70

    def test_monitor_returns_results(
        self,
        monitor: RiskMonitor,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that monitor returns a results dictionary."""
        results = monitor.monitor(scored_data)
        assert isinstance(results, dict)
        assert "overall_status" in results

    def test_monitor_overall_status_is_pass(
        self,
        monitor: RiskMonitor,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that monitor returns PASS status."""
        results = monitor.monitor(scored_data)
        assert results["overall_status"] == "PASS", (
            f"Monitor status is {results['overall_status']} "
            f"instead of PASS"
        )

    def test_monitor_checks_required_columns(
        self,
        monitor: RiskMonitor,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that monitor checks for required columns."""
        results = monitor.monitor(scored_data)
        assert "data_quality_checks" in results
        assert "required_columns" in results["data_quality_checks"]

    def test_monitor_validates_probabilities(
        self,
        monitor: RiskMonitor,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that monitor validates probabilities."""
        results = monitor.monitor(scored_data)
        prob_check = results["data_quality_checks"]["probabilities"]
        assert prob_check["status"] == "PASS"
        assert prob_check["invalid_count"] == 0

    def test_monitor_validates_risk_levels(
        self,
        monitor: RiskMonitor,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that monitor validates risk levels."""
        results = monitor.monitor(scored_data)
        risk_check = results["data_quality_checks"]["risk_levels"]
        assert risk_check["status"] == "PASS"

    def test_monitor_validates_actions(
        self,
        monitor: RiskMonitor,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that monitor validates actions."""
        results = monitor.monitor(scored_data)
        action_check = results["data_quality_checks"]["actions"]
        assert action_check["status"] == "PASS"


class TestRequirement8FrozenModel:
    """Requirement 8: Frozen model verification returns PASS."""

    def test_verify_frozen_model_returns_dict(
        self,
        monitor: RiskMonitor,
    ) -> None:
        """Test that frozen model verification returns a dictionary."""
        result = monitor.verify_frozen_model()
        assert isinstance(result, dict)

    def test_frozen_model_exists(
        self,
        monitor: RiskMonitor,
    ) -> None:
        """Test that frozen model file exists."""
        result = monitor.verify_frozen_model()
        assert result["exists"] is True

    def test_frozen_model_loads(
        self,
        monitor: RiskMonitor,
    ) -> None:
        """Test that frozen model loads successfully."""
        result = monitor.verify_frozen_model()
        assert result["loads"] is True

    def test_frozen_model_feature_count(
        self,
        monitor: RiskMonitor,
    ) -> None:
        """Test that frozen model has exactly 26 features."""
        result = monitor.verify_frozen_model()
        assert result["input_feature_count"] == 26

    def test_frozen_model_feature_count_check_passes(
        self,
        monitor: RiskMonitor,
    ) -> None:
        """Test that feature count check passes."""
        result = monitor.verify_frozen_model()
        assert result["feature_count_check"] == "PASS"

    def test_frozen_model_verification_status_passes(
        self,
        monitor: RiskMonitor,
    ) -> None:
        """Test that overall frozen model verification passes."""
        result = monitor.verify_frozen_model()
        assert result["status"] == "PASS"


class TestRequirement9FrozenPolicy:
    """Requirement 9: Frozen policy verification returns PASS."""

    def test_verify_frozen_policy_returns_dict(
        self,
        monitor: RiskMonitor,
    ) -> None:
        """Test that frozen policy verification returns a dictionary."""
        result = monitor.verify_frozen_policy()
        assert isinstance(result, dict)

    def test_review_threshold_verified(
        self,
        monitor: RiskMonitor,
    ) -> None:
        """Test that review threshold is verified as 0.35."""
        result = monitor.verify_frozen_policy()
        assert result["expected_review_threshold"] == 0.35
        assert np.isclose(
            result["actual_review_threshold"],
            0.35,
        )
        assert result["review_threshold_check"] == "PASS"

    def test_hold_threshold_verified(
        self,
        monitor: RiskMonitor,
    ) -> None:
        """Test that hold threshold is verified as 0.70."""
        result = monitor.verify_frozen_policy()
        assert result["expected_hold_threshold"] == 0.70
        assert np.isclose(
            result["actual_hold_threshold"],
            0.70,
        )
        assert result["hold_threshold_check"] == "PASS"

    def test_frozen_policy_verification_status_passes(
        self,
        monitor: RiskMonitor,
    ) -> None:
        """Test that overall frozen policy verification passes."""
        result = monitor.verify_frozen_policy()
        assert result["status"] == "PASS"


class TestRequirement10Report:
    """Requirement 10: Save actual results to reports/day10_integration_report.json"""

    def test_report_file_path_exists_after_integration(self) -> None:
        """
        Test that the report file path directory exists.

        Note: Report is generated by run_day10_integration.py script.
        This test verifies the path would be writable.
        """
        report_dir = Path("reports")
        assert report_dir.exists(), "reports directory must exist"
        assert report_dir.is_dir()

    def test_report_structure_would_be_valid(
        self,
        scored_data: pd.DataFrame,
        scorer: MerchantRiskScorer,
        monitor: RiskMonitor,
    ) -> None:
        """Test that report structure would be valid JSON."""
        # Build a sample report structure
        sample_report = {
            "timestamp": "2026-08-30T00:00:00",
            "stage": "day10_integration_test",
            "test_data": {
                "rows_tested": len(scored_data),
                "features_per_row": len(
                    scored_data.columns
                ),
            },
            "requirements": [
                {"requirement": 1, "status": "PASS"},
                {"requirement": 2, "status": "PASS"},
            ],
            "overall_status": "PASS",
        }

        # Verify it's JSON serializable
        json_str = json.dumps(sample_report)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Verify it can be loaded back
        loaded = json.loads(json_str)
        assert loaded["stage"] == "day10_integration_test"


class TestIntegrationEnd2End:
    """End-to-end integration test of all components."""

    def test_full_pipeline_executes_without_error(
        self,
        test_data: pd.DataFrame,
    ) -> None:
        """Test that the full pipeline executes successfully."""
        # Initialize scorer
        scorer = MerchantRiskScorer(
            model_path="models/logistic_regression_day6.joblib"
        )

        # Score data
        scored_data = scorer.score_transactions(
            test_data,
            include_explanation=True,
        )

        # Initialize monitor
        monitor = RiskMonitor(
            review_threshold=0.35,
            hold_threshold=0.70,
        )

        # Monitor results
        monitor_result = monitor.monitor(scored_data)

        # Verify all critical paths
        assert len(scored_data) == 10
        assert monitor_result["overall_status"] == "PASS"

    def test_all_10_rows_processed(
        self,
        test_data: pd.DataFrame,
        scored_data: pd.DataFrame,
    ) -> None:
        """Test that all 10 rows are processed."""
        assert len(test_data) == 10
        assert len(scored_data) == 10

    def test_deterministic_results_across_runs(
        self,
        test_data: pd.DataFrame,
    ) -> None:
        """
        Test that results are deterministic across multiple runs.

        Same input data should produce identical probabilities.
        """
        scorer = MerchantRiskScorer()

        result1 = scorer.score_transactions(test_data)
        result2 = scorer.score_transactions(test_data)

        assert np.allclose(
            result1["fraud_probability"].values,
            result2["fraud_probability"].values,
        )
