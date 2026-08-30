from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap


@dataclass
class FeatureContribution:
    internal_feature: str
    display_name: str
    direction: str
    contribution: float
    magnitude: float
    feature_value: Any


class FraudExplainer:
    """
    Reusable local explanation engine for the frozen Day 6 AegisRisk model.

    The explainer:
    1. Preserves the exact original model input feature order.
    2. Uses the frozen Day 6 preprocessing pipeline.
    3. Explains the actual trained Logistic Regression representation.
    4. Returns local contributions sorted by absolute magnitude.

    This component does not retrain or modify the model or policy.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)

        self.pipeline = joblib.load(self.model_path)

        if "preprocessor" not in self.pipeline.named_steps:
            raise ValueError(
                "Frozen model pipeline does not contain a 'preprocessor' step."
            )

        if "model" not in self.pipeline.named_steps:
            raise ValueError(
                "Frozen model pipeline does not contain a 'model' step."
            )

        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model = self.pipeline.named_steps["model"]

        self.input_features = list(self.pipeline.feature_names_in_)
        self.transformed_feature_names = list(
            self.preprocessor.get_feature_names_out()
        )

        self._explainer = None

    def _validate_input(self, transaction: pd.DataFrame) -> pd.DataFrame:
        """
        Validate required frozen features and preserve exact model order.
        """
        missing = [
            feature
            for feature in self.input_features
            if feature not in transaction.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required frozen model features: {missing}"
            )

        X = transaction.loc[:, self.input_features].copy()

        if list(X.columns) != self.input_features:
            raise ValueError(
                "Transaction feature order does not match frozen model order."
            )

        return X

    def _get_explainer(self, X_transformed):
        """
        Create the SHAP explainer lazily.

        The exact transformed representation used by the frozen model
        is supplied to SHAP.
        """
        if self._explainer is None:
            self._explainer = shap.LinearExplainer(
                self.model,
                X_transformed,
            )

        return self._explainer

    @staticmethod
    def _display_name(transformed_feature: str) -> str:
        """
        Convert internal transformed feature names into readable labels.

        This does not modify model features.
        """
        display_mapping = {
            "numeric__amount": "Transaction amount",
            "numeric__account_age_days": "Account age",
            "numeric__failed_attempts": "Failed payment attempts",
            "numeric__amount_log": "Log-transformed transaction amount",
            "numeric__hour": "Transaction hour",
            "numeric__day_of_week": "Day of week",
            "numeric__is_weekend": "Weekend indicator",
            "numeric__hour_sin": "Transaction time cycle (sine)",
            "numeric__hour_cos": "Transaction time cycle (cosine)",
            "numeric__customer_txn_count_before":
                "Customer transactions before this transaction",
            "numeric__customer_amount_mean_before":
                "Customer historical average amount",
            "numeric__customer_amount_std_before":
                "Customer historical amount variability",
            "numeric__amount_vs_customer_mean":
                "Amount compared with the customer's historical average",
            "numeric__customer_amount_zscore":
                "Amount deviation from the customer's historical pattern",
            "numeric__customer_seconds_since_prev":
                "Time since the customer's previous transaction",
            "numeric__customer_has_history":
                "Whether the customer has transaction history",
            "numeric__customer_txn_count_10m":
                "Transaction activity in the last 10 minutes",
            "numeric__customer_txn_count_1h":
                "Transaction activity in the last hour",
            "numeric__customer_txn_count_24h":
                "Transaction activity in the last 24 hours",
            "numeric__merchant_txn_count_before":
                "Merchant transaction history before this transaction",
            "numeric__merchant_amount_mean_before":
                "Merchant historical average amount",
            "numeric__merchant_seconds_since_prev":
                "Time since the merchant's previous transaction",
            "numeric__merchant_txn_count_1h":
                "Merchant transaction activity in the last hour",
            "numeric__customer_device_seen_before":
                "Whether this customer has used this device before",
            "numeric__device_seen_before":
                "Whether this device has been seen before",
            "categorical__payment_method_card":
                "Payment method: card",
            "categorical__payment_method_netbanking":
                "Payment method: netbanking",
            "categorical__payment_method_upi":
                "Payment method: UPI",
            "categorical__payment_method_wallet":
                "Payment method: wallet",
        }

        return display_mapping.get(
            transformed_feature,
            transformed_feature,
        )

    @staticmethod
    def _direction(value: float) -> str:
        if value > 0:
            return "INCREASES_FRAUD_RISK"
        if value < 0:
            return "DECREASES_FRAUD_RISK"
        return "NEUTRAL"

    @staticmethod
    def _original_feature_value(
        transformed_feature: str,
        transaction_row: pd.Series,
    ) -> Any:
        """
        Return the actual source value when it can be safely mapped.

        For one-hot encoded payment methods, return the original
        payment method rather than a fabricated numeric interpretation.
        """
        if transformed_feature.startswith("numeric__"):
            original_feature = transformed_feature.replace(
                "numeric__",
                "",
                1,
            )
            return transaction_row.get(original_feature)

        if transformed_feature.startswith("categorical__payment_method_"):
            return transaction_row.get("payment_method")

        return None

    def explain_transaction(
        self,
        transaction: pd.DataFrame,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Generate local SHAP explanations.

        Parameters
        ----------
        transaction:
            DataFrame containing one or more transactions.
        top_k:
            Number of largest absolute contributions to return.

        Returns
        -------
        List of dictionaries, one explanation list per transaction.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        X = self._validate_input(transaction)

        X_transformed = self.preprocessor.transform(X)

        if X_transformed.shape[1] != len(
            self.transformed_feature_names
        ):
            raise ValueError(
                "Transformed feature count does not match frozen feature names."
            )

        explainer = self._get_explainer(X_transformed)
        shap_values = explainer.shap_values(X_transformed)

        if shap_values.ndim != 2:
            raise ValueError(
                "Expected 2-dimensional SHAP values "
                "for the frozen binary Logistic Regression model."
            )

        if shap_values.shape[1] != len(
            self.transformed_feature_names
        ):
            raise ValueError(
                "SHAP values are not aligned with transformed feature names."
            )

        results: list[dict[str, Any]] = []

        for row_index in range(len(X)):
            transaction_row = X.iloc[row_index]
            row_shap = shap_values[row_index]

            contributions: list[FeatureContribution] = []

            for feature_name, contribution_value in zip(
                self.transformed_feature_names,
                row_shap,
            ):
                contribution = float(contribution_value)

                contributions.append(
                    FeatureContribution(
                        internal_feature=feature_name,
                        display_name=self._display_name(feature_name),
                        direction=self._direction(contribution),
                        contribution=contribution,
                        magnitude=abs(contribution),
                        feature_value=self._original_feature_value(
                            feature_name,
                            transaction_row,
                        ),
                    )
                )

            contributions.sort(
                key=lambda item: item.magnitude,
                reverse=True,
            )

            top_contributions = [
                asdict(item)
                for item in contributions[:top_k]
            ]

            results.append(
                {
                    "top_contributors": top_contributions,
                    "feature_count_explained": len(contributions),
                    "top_k": top_k,
                }
            )

        return results