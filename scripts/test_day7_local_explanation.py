from pathlib import Path
import json

import pandas as pd

from src.risk.explainability import FraudExplainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_day6.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "transactions_features.csv"


def main():
    print("=" * 70)
    print("AEGISRISK AI - DAY 7")
    print("LOCAL EXPLANATION TEST")
    print("=" * 70)

    # Load actual processed repository data
    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset rows: {len(df):,}")
    print(f"Dataset columns: {len(df.columns)}")

    # Create the reusable frozen-model explainer
    explainer = FraudExplainer(MODEL_PATH)

    print("\nFrozen model feature count:", len(explainer.input_features))
    print(
        "Transformed feature count:",
        len(explainer.transformed_feature_names),
    )

    # Select one actual transaction.
    # Keep transaction_id only for identification; it is NOT sent to the model.
    sample = df.iloc[[0]].copy()

    transaction_id = (
        sample["transaction_id"].iloc[0]
        if "transaction_id" in sample.columns
        else "UNAVAILABLE"
    )

    print(f"\nTransaction ID: {transaction_id}")

    # Generate a local explanation using the actual frozen pipeline
    explanation = explainer.explain_transaction(
        sample,
        top_k=5,
    )

    print("\nExplanation output:")
    print(json.dumps(explanation, indent=2, default=str))

    # Structural verification
    if len(explanation) != 1:
        raise ValueError(
            "Expected exactly one explanation for one transaction."
        )

    result = explanation[0]

    required_keys = {
        "top_contributors",
        "feature_count_explained",
        "top_k",
    }

    missing_keys = required_keys - set(result.keys())

    if missing_keys:
        raise ValueError(
            f"Explanation output missing keys: {missing_keys}"
        )

    if result["feature_count_explained"] != 29:
        raise ValueError(
            "Expected all 29 transformed model features to be explained."
        )

    if len(result["top_contributors"]) != 5:
        raise ValueError(
            "Expected exactly 5 top contributors."
        )

    required_contributor_keys = {
        "internal_feature",
        "display_name",
        "direction",
        "contribution",
        "magnitude",
        "feature_value",
    }

    for contributor in result["top_contributors"]:
        missing = required_contributor_keys - set(contributor.keys())

        if missing:
            raise ValueError(
                f"Contributor missing keys: {missing}"
            )

        if contributor["direction"] not in {
            "INCREASES_FRAUD_RISK",
            "DECREASES_FRAUD_RISK",
            "NEUTRAL",
        }:
            raise ValueError(
                "Invalid contribution direction."
            )

    # Verify deterministic magnitude ordering
    magnitudes = [
        item["magnitude"]
        for item in result["top_contributors"]
    ]

    if magnitudes != sorted(magnitudes, reverse=True):
        raise ValueError(
            "Top contributors are not sorted by magnitude."
        )

    print("\n" + "=" * 70)
    print("LOCAL EXPLANATION STRUCTURE: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()