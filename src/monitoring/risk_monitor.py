"""
AegisRisk AI - Day 9
Fraud-Risk Monitoring & Model Health

Reusable monitoring component for the frozen AegisRisk scoring system.

This module:
- monitors fraud probability statistics
- monitors risk-level and action distributions
- verifies probability -> risk -> action consistency
- validates scored transaction data
- verifies frozen model configuration

It does NOT:
- retrain models
- modify model artifacts
- modify preprocessing
- modify feature schema
- modify decision thresholds
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.evaluation.decision_policy import (
    ALLOW,
    HOLD_FOR_VERIFICATION,
    REVIEW,
    DecisionPolicy,
    decide,
)


class RiskMonitor:
    """
    Reusable monitoring system for scored AegisRisk transactions.

    The monitor observes the output of the existing frozen scoring
    pipeline without changing the model or decision policy.
    """

    REQUIRED_COLUMNS = {
        "fraud_probability",
        "risk_decision",
        "risk_action",
    }

    VALID_RISK_LEVELS = {
        "LOW_RISK",
        "MEDIUM_RISK",
        "HIGH_RISK",
    }

    VALID_ACTIONS = {
        ALLOW,
        REVIEW,
        HOLD_FOR_VERIFICATION,
    }

    def __init__(
        self,
        review_threshold: float = 0.35,
        hold_threshold: float = 0.70,
        model_path: str | Path = (
            "models/logistic_regression_day6.joblib"
        ),
    ) -> None:
        """
        Initialize the monitoring configuration.

        Thresholds default to the frozen Day 6 policy and are validated
        against the existing decision-policy implementation.

        They are configuration values for verification, not newly tuned
        business thresholds.
        """
        self.review_threshold = float(review_threshold)
        self.hold_threshold = float(hold_threshold)
        self.model_path = Path(model_path)

        self.policy = DecisionPolicy(
            review_threshold=self.review_threshold,
            hold_threshold=self.hold_threshold,
        )

    def monitor(
        self,
        scored_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Monitor a scored transaction DataFrame.

        Parameters
        ----------
        scored_data:
            DataFrame containing:
            - fraud_probability
            - risk_decision
            - risk_action

        Returns
        -------
        dict
            Structured monitoring result.
        """
        column_check = self._validate_required_columns(scored_data)

        probability_check = self._validate_probabilities(
            scored_data
        )

        risk_check = self._validate_risk_levels(
            scored_data
        )

        action_check = self._validate_actions(
            scored_data
        )

        policy_check = self._verify_policy_consistency(
            scored_data
        )

        probability_statistics = self._probability_statistics(
            scored_data
        )

        risk_distribution = self._risk_distribution(
            scored_data
        )

        action_distribution = self._action_distribution(
            scored_data
        )

        data_quality_checks = {
            "required_columns": column_check,
            "probabilities": probability_check,
            "risk_levels": risk_check,
            "actions": action_check,
            "policy_consistency": policy_check,
        }

        status = self._determine_status(
            data_quality_checks
        )

        return {
            "transaction_count": int(len(scored_data)),
            "probability_statistics": probability_statistics,
            "risk_distribution": risk_distribution,
            "action_distribution": action_distribution,
            "policy_verification": {
                "review_threshold": self.review_threshold,
                "hold_threshold": self.hold_threshold,
                "policy_consistency": policy_check,
            },
            "data_quality_checks": data_quality_checks,
            "overall_status": status,
        }

    def verify_frozen_model(self) -> dict[str, Any]:
        """
        Verify the existing frozen Day 6 model artifact.

        This method only loads and inspects the artifact.
        It never trains or modifies it.
        """
        result: dict[str, Any] = {
            "model_path": str(self.model_path),
            "exists": False,
            "loads": False,
            "input_feature_count": None,
            "expected_input_feature_count": 26,
            "feature_count_check": "UNAVAILABLE",
            "status": "ALERT",
        }

        if not self.model_path.exists():
            return result

        result["exists"] = True

        try:
            pipeline = joblib.load(self.model_path)
            result["loads"] = True

            feature_names = getattr(
                pipeline,
                "feature_names_in_",
                None,
            )

            if feature_names is None:
                result["feature_count_check"] = "UNAVAILABLE"
                result["status"] = "WARNING"
                return result

            feature_count = len(feature_names)
            result["input_feature_count"] = int(feature_count)

            if feature_count == 26:
                result["feature_count_check"] = "PASS"
                result["status"] = "PASS"
            else:
                result["feature_count_check"] = "ALERT"
                result["status"] = "ALERT"

        except Exception as exc:
            result["error"] = str(exc)
            result["status"] = "ALERT"

        return result

    def verify_frozen_policy(self) -> dict[str, Any]:
        """
        Verify that monitoring configuration matches the frozen
        Day 6 thresholds.
        """
        review_matches = np.isclose(
            self.review_threshold,
            0.35,
        )

        hold_matches = np.isclose(
            self.hold_threshold,
            0.70,
        )

        return {
            "expected_review_threshold": 0.35,
            "actual_review_threshold": self.review_threshold,
            "review_threshold_check": (
                "PASS" if review_matches else "ALERT"
            ),
            "expected_hold_threshold": 0.70,
            "actual_hold_threshold": self.hold_threshold,
            "hold_threshold_check": (
                "PASS" if hold_matches else "ALERT"
            ),
            "status": (
                "PASS"
                if review_matches and hold_matches
                else "ALERT"
            ),
        }

    def _validate_required_columns(
        self,
        data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Check that all required scored-data columns exist."""
        missing = sorted(
            self.REQUIRED_COLUMNS - set(data.columns)
        )

        return {
            "status": "PASS" if not missing else "ALERT",
            "missing_columns": missing,
        }

    def _validate_probabilities(
        self,
        data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Validate fraud probabilities."""
        if "fraud_probability" not in data.columns:
            return {
                "status": "ALERT",
                "invalid_count": None,
                "reason": "Required column is missing.",
            }

        probabilities = pd.to_numeric(
            data["fraud_probability"],
            errors="coerce",
        )

        invalid_mask = (
            probabilities.isna()
            | ~np.isfinite(probabilities)
            | (probabilities < 0.0)
            | (probabilities > 1.0)
        )

        invalid_count = int(invalid_mask.sum())

        return {
            "status": (
                "PASS"
                if invalid_count == 0
                else "ALERT"
            ),
            "invalid_count": invalid_count,
        }

    def _validate_risk_levels(
        self,
        data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Validate risk-level values."""
        if "risk_decision" not in data.columns:
            return {
                "status": "ALERT",
                "invalid_count": None,
                "reason": "Required column is missing.",
            }

        invalid_mask = ~data["risk_decision"].isin(
            self.VALID_RISK_LEVELS
        )

        invalid_count = int(invalid_mask.sum())

        return {
            "status": (
                "PASS"
                if invalid_count == 0
                else "ALERT"
            ),
            "invalid_count": invalid_count,
        }

    def _validate_actions(
        self,
        data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Validate operational action values."""
        if "risk_action" not in data.columns:
            return {
                "status": "ALERT",
                "invalid_count": None,
                "reason": "Required column is missing.",
            }

        invalid_mask = ~data["risk_action"].isin(
            self.VALID_ACTIONS
        )

        invalid_count = int(invalid_mask.sum())

        return {
            "status": (
                "PASS"
                if invalid_count == 0
                else "ALERT"
            ),
            "invalid_count": invalid_count,
        }

    def _verify_policy_consistency(
        self,
        data: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Verify probability -> risk level -> action consistency.

        This independently reconstructs the expected policy outcome
        using the frozen thresholds.
        """
        required = self.REQUIRED_COLUMNS

        if not required.issubset(data.columns):
            return {
                "status": "ALERT",
                "inconsistent_count": None,
                "reason": "Required columns are missing.",
            }

        probabilities = pd.to_numeric(
            data["fraud_probability"],
            errors="coerce",
        )

        if (
            probabilities.isna().any()
            or (~np.isfinite(probabilities)).any()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            return {
                "status": "ALERT",
                "inconsistent_count": None,
                "reason": "Invalid probabilities detected.",
            }

        expected_risk_levels = [
            (
                "LOW_RISK"
                if probability < self.review_threshold
                else (
                    "MEDIUM_RISK"
                    if probability < self.hold_threshold
                    else "HIGH_RISK"
                )
            )
            for probability in probabilities
        ]

        expected_actions = [
            decide(
                float(probability),
                self.policy,
            )
            for probability in probabilities
        ]

        risk_mismatch = (
            data["risk_decision"].to_numpy()
            != np.asarray(expected_risk_levels)
        )

        action_mismatch = (
            data["risk_action"].to_numpy()
            != np.asarray(expected_actions)
        )

        inconsistent_count = int(
            np.logical_or(
                risk_mismatch,
                action_mismatch,
            ).sum()
        )

        return {
            "status": (
                "PASS"
                if inconsistent_count == 0
                else "ALERT"
            ),
            "inconsistent_count": inconsistent_count,
            "checked_count": int(len(data)),
        }

    @staticmethod
    def _probability_statistics(
        data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Calculate descriptive statistics for fraud probability."""
        if "fraud_probability" not in data.columns:
            return {
                "status": "UNAVAILABLE",
            }

        probabilities = pd.to_numeric(
            data["fraud_probability"],
            errors="coerce",
        ).dropna()

        if probabilities.empty:
            return {
                "status": "UNAVAILABLE",
                "count": 0,
            }

        return {
            "status": "PASS",
            "count": int(len(probabilities)),
            "minimum": float(probabilities.min()),
            "maximum": float(probabilities.max()),
            "mean": float(probabilities.mean()),
            "median": float(probabilities.median()),
            "standard_deviation": float(
                probabilities.std(ddof=0)
            ),
            "percentiles": {
                "p01": float(probabilities.quantile(0.01)),
                "p05": float(probabilities.quantile(0.05)),
                "p25": float(probabilities.quantile(0.25)),
                "p50": float(probabilities.quantile(0.50)),
                "p75": float(probabilities.quantile(0.75)),
                "p95": float(probabilities.quantile(0.95)),
                "p99": float(probabilities.quantile(0.99)),
            },
        }

    def _risk_distribution(
        self,
        data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Calculate risk-level counts and percentages."""
        total = len(data)

        result: dict[str, Any] = {}

        for risk_level in (
            "LOW_RISK",
            "MEDIUM_RISK",
            "HIGH_RISK",
        ):
            count = int(
                (
                    data["risk_decision"] == risk_level
                ).sum()
            ) if "risk_decision" in data.columns else 0

            percentage = (
                (count / total) * 100
                if total > 0
                else 0.0
            )

            result[risk_level] = {
                "count": count,
                "percentage": float(percentage),
            }

        return result

    def _action_distribution(
        self,
        data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Calculate action counts and percentages."""
        total = len(data)

        result: dict[str, Any] = {}

        for action in (
            ALLOW,
            REVIEW,
            HOLD_FOR_VERIFICATION,
        ):
            count = int(
                (
                    data["risk_action"] == action
                ).sum()
            ) if "risk_action" in data.columns else 0

            percentage = (
                (count / total) * 100
                if total > 0
                else 0.0
            )

            result[action] = {
                "count": count,
                "percentage": float(percentage),
            }

        return result

    @staticmethod
    def _determine_status(
        checks: dict[str, dict[str, Any]],
    ) -> str:
        """Determine overall monitoring status."""
        statuses = [
            check.get("status")
            for check in checks.values()
        ]

        if "ALERT" in statuses:
            return "ALERT"

        if "WARNING" in statuses:
            return "WARNING"

        return "PASS"