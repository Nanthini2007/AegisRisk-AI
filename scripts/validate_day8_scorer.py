"""
AegisRisk AI - Day 8
Scored Transaction Validation

Validates that the scoring pipeline produces correct outputs
using the frozen model and existing decision policy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from src.inference.scorer import MerchantRiskScorer


def load_sample_data(
    data_path: Path = Path("data/processed/transactions_features.csv"),
    sample_size: int = 100,
) -> pd.DataFrame:
    """Load sample transactions for validation."""
    print(f"\nLoading data: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path, nrows=sample_size)
    print(f"Loaded {len(df)} transactions")

    return df


def validate_scored_output(
    scored: pd.DataFrame,
) -> dict:
    """Validate the scored transaction output."""

    validation = {
        "total_transactions": len(scored),
        "fraud_probability_range": (
            float(scored["fraud_probability"].min()),
            float(scored["fraud_probability"].max()),
        ),
        "fraud_probability_mean": float(
            scored["fraud_probability"].mean()
        ),
        "risk_decision_distribution": (
            scored["risk_decision"].value_counts().to_dict()
        ),
        "risk_action_distribution": (
            scored["risk_action"].value_counts().to_dict()
        ),
        "has_transaction_ids": "transaction_id" in scored.columns,
        "has_timestamps": "timestamp" in scored.columns,
        "has_fraud_probability": "fraud_probability" in scored.columns,
        "has_risk_decision": "risk_decision" in scored.columns,
        "has_risk_action": "risk_action" in scored.columns,
        "has_explanations": "top_contributors" in scored.columns,
        "has_summaries": "explanation_summary" in scored.columns,
    }

    # Additional checks
    prob_valid = (scored["fraud_probability"] >= 0) & (
        scored["fraud_probability"] <= 1
    )
    validation["all_probabilities_valid"] = bool(prob_valid.all())

    decision_valid_values = {"LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"}
    validation["all_decisions_valid"] = bool(
        scored["risk_decision"].isin(decision_valid_values).all()
    )

    action_valid_values = {"ALLOW", "REVIEW", "HOLD_FOR_VERIFICATION"}
    validation["all_actions_valid"] = bool(
        scored["risk_action"].isin(action_valid_values).all()
    )

    # Threshold alignment check
    alignment = (
        (scored[scored["risk_decision"] == "LOW_RISK"]["fraud_probability"] < 0.35).all()
        and (
            scored[scored["risk_decision"] == "MEDIUM_RISK"][
                "fraud_probability"
            ]
            >= 0.35
        ).all()
        and (
            scored[scored["risk_decision"] == "MEDIUM_RISK"][
                "fraud_probability"
            ]
            < 0.70
        ).all()
        and (
            scored[scored["risk_decision"] == "HIGH_RISK"][
                "fraud_probability"
            ]
            >= 0.70
        ).all()
    )

    validation["threshold_alignment"] = alignment

    return validation


def main() -> None:
    print("=" * 70)
    print("AEGISRISK AI - DAY 8")
    print("SCORED TRANSACTION VALIDATION")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    data = load_sample_data(sample_size=100)

    # ========================================================
    # INITIALIZE SCORER
    # ========================================================

    print("\nInitializing scorer...")
    scorer = MerchantRiskScorer()
    print(f"✓ Scorer initialized")
    print(f"  - Model: {scorer.model_path}")
    print(f"  - Input features: {len(scorer.input_features)}")
    print(f"  - Review threshold: {scorer.policy.review_threshold}")
    print(f"  - Hold threshold: {scorer.policy.hold_threshold}")

    # ========================================================
    # SCORE TRANSACTIONS
    # ========================================================

    print("\nScoring transactions...")
    scored = scorer.score_transactions(data, include_explanation=True, top_k=5)
    print(f"✓ Scored {len(scored)} transactions")

    # ========================================================
    # VALIDATE OUTPUT
    # ========================================================

    print("\nValidating scored output...")
    validation = validate_scored_output(scored)

    print("\n✓ Validation Results:")
    for key, value in validation.items():
        print(f"  {key}: {value}")

    # ========================================================
    # SAMPLE OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("SAMPLE SCORED TRANSACTION (First Row)")
    print("=" * 70)

    sample_row = scored.iloc[0]

    print(f"\nTransaction ID: {sample_row.get('transaction_id', 'N/A')}")
    print(f"Amount: ${sample_row.get('amount', 0):.2f}")
    print(f"Payment Method: {sample_row.get('payment_method', 'N/A')}")
    print(f"\nFraud Probability: {sample_row['fraud_probability']:.1%}")
    print(f"Risk Decision: {sample_row['risk_decision']}")
    print(f"Risk Action: {sample_row['risk_action']}")
    print(f"Explanation Summary: {sample_row.get('explanation_summary', 'N/A')}")

    if sample_row.get("top_contributors"):
        print(f"\nTop Contributing Factors:")
        for i, contrib in enumerate(
            sample_row["top_contributors"][:3], 1
        ):
            print(
                f"  {i}. {contrib.get('display_name', 'N/A')} "
                f"({contrib.get('direction', 'N/A')}, "
                f"magnitude: {contrib.get('magnitude', 0):.4f})"
            )

    # ========================================================
    # VERDICT
    # ========================================================

    all_valid = all(
        v for k, v in validation.items() if isinstance(v, bool)
    )

    if all_valid:
        print("\n" + "=" * 70)
        print("✓ VALIDATION PASSED")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("✗ VALIDATION FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit(main())
