"""
AegisRisk AI - Day 8
Merchant Risk Scorer

Uses frozen Day 6 model, Day 6 decision policy, and Day 7 SHAP explanations
to score transactions in a unified pipeline.

Does not retrain, modify features, or change thresholds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import numpy as np

from src.evaluation.decision_policy import (
    DecisionPolicy,
    decide_many,
)
from src.risk.explainability import FraudExplainer


class MerchantRiskScorer:
    """
    Prototype risk scorer for merchant transactions.

    Combines:
    1. Frozen Day 6 Logistic Regression model
    2. Frozen Day 6 decision policy (thresholds + actions)
    3. Day 7 SHAP explainability system

    Input: DataFrame with 26 required features
    Output: DataFrame with fraud_probability, risk_decision, risk_action, explanation
    """

    def __init__(
        self,
        model_path: str | Path = "models/logistic_regression_day6.joblib",
    ):
        """
        Initialize the scorer with frozen Day 6 artifacts.

        Parameters
        ----------
        model_path : str or Path
            Path to frozen Logistic Regression pipeline.
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Frozen model not found: {self.model_path}"
            )

        # Load frozen model
        self.pipeline = joblib.load(self.model_path)
        self.model = self.pipeline.named_steps["model"]
        self.preprocessor = self.pipeline.named_steps["preprocessor"]

        # Get input features from frozen model
        self.input_features = list(self.pipeline.feature_names_in_)

        if len(self.input_features) != 26:
            raise ValueError(
                f"Expected 26 input features, got {len(self.input_features)}"
            )

        # Frozen decision policy from Day 6
        self.policy = DecisionPolicy(
            review_threshold=0.35,
            hold_threshold=0.70,
        )

        # Explainability engine
        self.explainer = FraudExplainer(self.model_path)

        # Track scoring metadata
        self.last_scored_count = 0

    def score_transactions(
        self,
        transactions: pd.DataFrame,
        include_explanation: bool = True,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """
        Score transactions and assign risk decisions.

        Parameters
        ----------
        transactions : pd.DataFrame
            DataFrame containing required features + transaction_id + timestamp.
        include_explanation : bool, default=True
            Whether to generate SHAP explanations.
        top_k : int, default=5
            Number of top feature contributions to include in explanation.

        Returns
        -------
        pd.DataFrame
            Original data plus:
            - fraud_probability (float): P(fraud) from model
            - risk_decision (str): LOW_RISK, MEDIUM_RISK, HIGH_RISK
            - risk_action (str): ALLOW, REVIEW, HOLD_FOR_VERIFICATION
            - top_contributors (list): Top k SHAP feature contributions (if requested)
            - explanation_summary (str): Human-readable risk explanation (if requested)
        """

        if transactions.empty:
            raise ValueError("Input DataFrame is empty.")

        # Validate required features
        required_features = self.input_features
        missing_features = [
            f for f in required_features if f not in transactions.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}"
            )

        # Extract features in exact frozen order
        X = transactions[required_features].copy()

        # Generate predictions
        probabilities = self.pipeline.predict_proba(X)[:, 1]

        # Assign risk decisions
        risk_decisions = self._assign_risk_decisions(probabilities)
        risk_actions = decide_many(probabilities, self.policy)

        # Build result dataframe
        result = transactions.copy()
        result["fraud_probability"] = probabilities
        result["risk_decision"] = risk_decisions
        result["risk_action"] = risk_actions

        # Generate explanations if requested
        if include_explanation:
            explanations = self.explainer.explain_transaction(
                X,
                top_k=top_k,
            )

            result["top_contributors"] = [
                exp.get("top_contributors", []) for exp in explanations
            ]

            # Create human-readable summary
            summaries = []
            for idx, exp in enumerate(explanations):
                summary = self._create_explanation_summary(
                    probabilities[idx],
                    risk_decisions[idx],
                    exp.get("top_contributors", []),
                )
                summaries.append(summary)

            result["explanation_summary"] = summaries

        # Track
        self.last_scored_count = len(result)

        return result

    @staticmethod
    def _assign_risk_decisions(probabilities) -> list[str]:
        """
        Assign risk decision level based on probability.

        Decision levels:
        - LOW_RISK: [0.0, 0.35)
        - MEDIUM_RISK: [0.35, 0.70)
        - HIGH_RISK: [0.70, 1.0]
        """
        decisions = []
        for prob in probabilities:
            if prob < 0.35:
                decisions.append("LOW_RISK")
            elif prob < 0.70:
                decisions.append("MEDIUM_RISK")
            else:
                decisions.append("HIGH_RISK")

        return decisions

    @staticmethod
    def _create_explanation_summary(
        probability: float,
        risk_decision: str,
        top_contributors: list[dict],
    ) -> str:
        """
        Create a human-readable explanation summary.

        Parameters
        ----------
        probability : float
            Fraud probability from model.
        risk_decision : str
            Risk level (LOW/MEDIUM/HIGH).
        top_contributors : list[dict]
            Top SHAP feature contributions.

        Returns
        -------
        str
            Formatted explanation.
        """

        summary = f"Fraud Risk: {probability:.1%} ({risk_decision}) | "

        if not top_contributors:
            summary += "No contributors."
            return summary

        # Get top contributors that increase risk
        risk_increasing = [
            c for c in top_contributors
            if c.get("direction") == "INCREASES_FRAUD_RISK"
        ]

        risk_decreasing = [
            c for c in top_contributors
            if c.get("direction") == "DECREASES_FRAUD_RISK"
        ]

        if risk_increasing:
            top_risk = risk_increasing[0]
            summary += f"Top risk factor: {top_risk.get('display_name', 'N/A')}"
        elif risk_decreasing:
            top_safe = risk_decreasing[0]
            summary += f"Top protective factor: {top_safe.get('display_name', 'N/A')}"
        else:
            summary += f"Primary factor: {top_contributors[0].get('display_name', 'N/A')}"

        return summary

    def get_metrics(self) -> dict[str, Any]:
        """Return metadata about last scoring run."""
        return {
            "model_path": str(self.model_path),
            "model_type": "Logistic Regression (Frozen Day 6)",
            "input_features": len(self.input_features),
            "policy_review_threshold": self.policy.review_threshold,
            "policy_hold_threshold": self.policy.hold_threshold,
            "last_scored_count": self.last_scored_count,
        }
