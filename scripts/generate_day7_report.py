from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import shap

from src.risk.explainability import FraudExplainer


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "logistic_regression_day6.joblib"
DATA_PATH = ROOT / "data" / "processed" / "transactions_features.csv"
REPORT_PATH = ROOT / "reports" / "day7_explainability_report.json"

TOP_K = 5
FULL_FEATURE_COUNT = 29


def verify_feature_order(explainer: FraudExplainer) -> bool:
    return (
        len(explainer.input_features) == 26
        and list(explainer.pipeline.feature_names_in_)
        == explainer.input_features
    )


def verify_transformed_alignment(explainer: FraudExplainer) -> bool:
    transformed = list(
        explainer.preprocessor.get_feature_names_out()
    )

    return (
        len(transformed) == FULL_FEATURE_COUNT
        and transformed == explainer.transformed_feature_names
    )


def verify_ranking(contributors: list[dict]) -> bool:
    magnitudes = [
        float(item["magnitude"])
        for item in contributors
    ]

    return magnitudes == sorted(
        magnitudes,
        reverse=True,
    )


def make_json_safe(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def main() -> None:
    print("=" * 70)
    print("AEGISRISK AI - DAY 7")
    print("AUDITABLE EXPLAINABILITY VERIFICATION REPORT")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Frozen model not found: {MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    print(f"\nLoading frozen model: {MODEL_PATH}")
    print(f"Loading dataset: {DATA_PATH}")

    pipeline = joblib.load(MODEL_PATH)
    dataset = pd.read_csv(DATA_PATH)

    print(f"\nDataset rows: {len(dataset):,}")
    print(f"Dataset columns: {len(dataset.columns)}")

    explainer = FraudExplainer(MODEL_PATH)

    input_feature_count = len(explainer.input_features)
    transformed_feature_count = len(
        explainer.transformed_feature_names
    )

    feature_order_verified = verify_feature_order(explainer)
    transformed_alignment_verified = verify_transformed_alignment(
        explainer
    )

    print("\n--- FROZEN MODEL ---")
    print(f"Input feature count: {input_feature_count}")
    print(
        f"Transformed feature count: "
        f"{transformed_feature_count}"
    )
    print(
        "Feature order verification: "
        f"{'PASSED' if feature_order_verified else 'FAILED'}"
    )
    print(
        "Transformed feature alignment: "
        f"{'PASSED' if transformed_alignment_verified else 'FAILED'}"
    )

    if not feature_order_verified:
        raise AssertionError(
            "Frozen model feature order verification failed."
        )

    if not transformed_alignment_verified:
        raise AssertionError(
            "Transformed feature alignment verification failed."
        )

    # SHAP availability/version is verified from the actual environment.
    shap_available = True
    shap_version = shap.__version__

    print("\n--- SHAP ---")
    print(f"SHAP available: {shap_available}")
    print(f"SHAP version: {shap_version}")

    # Deterministic real transaction from the repository dataset.
    transaction = dataset.head(1).copy()

    if "transaction_id" not in transaction.columns:
        raise ValueError(
            "Dataset does not contain transaction_id."
        )

    transaction_id = transaction.iloc[0]["transaction_id"]

    print("\n--- SAMPLE TRANSACTION ---")
    print(f"Transaction ID: {transaction_id}")

    # Generate all 29 contributions first.
    #
    # This is important for auditability:
    # the report derives TOP_K from the complete explanation,
    # rather than asking the explainer for only five values.
    explanations = explainer.explain_transaction(
        transaction,
        top_k=FULL_FEATURE_COUNT,
    )

    if len(explanations) != 1:
        raise AssertionError(
            "Expected exactly one explanation."
        )

    explanation = explanations[0]

    all_contributors = explanation["top_contributors"]

    if explanation["feature_count_explained"] != FULL_FEATURE_COUNT:
        raise AssertionError(
            "Explanation does not cover all transformed features."
        )

    if len(all_contributors) != FULL_FEATURE_COUNT:
        raise AssertionError(
            "Expected one contribution for each transformed feature."
        )

    local_explanation_generated = True

    contribution_ranking_verified = verify_ranking(
        all_contributors
    )

    if not contribution_ranking_verified:
        raise AssertionError(
            "Contribution ranking verification failed."
        )

    non_zero_contributions = [
        item
        for item in all_contributors
        if float(item["magnitude"]) > 0.0
    ]

    zero_contributions = [
        item
        for item in all_contributors
        if float(item["magnitude"]) == 0.0
    ]

    # The explainer already sorts by absolute magnitude.
    # We take the first TOP_K actual contributors without
    # modifying, filtering, or fabricating any values.
    top_contributors = all_contributors[:TOP_K]

    print("\n--- LOCAL EXPLANATION ---")
    print(
        "Local explanation generated: "
        f"{'PASSED' if local_explanation_generated else 'FAILED'}"
    )
    print(
        "Contribution ranking verification: "
        f"{'PASSED' if contribution_ranking_verified else 'FAILED'}"
    )

    print("\n--- CONTRIBUTION SUMMARY ---")
    print(
        f"Total transformed features explained: "
        f"{len(all_contributors)}"
    )
    print(
        f"Non-zero contributions: "
        f"{len(non_zero_contributions)}"
    )
    print(
        f"Zero contributions: "
        f"{len(zero_contributions)}"
    )

    print("\n--- TOP CONTRIBUTORS ---")

    for index, item in enumerate(
        top_contributors,
        start=1,
    ):
        value = make_json_safe(item["feature_value"])

        print(
            f"{index:02d}. "
            f"{item['internal_feature']} | "
            f"contribution={item['contribution']} | "
            f"magnitude={item['magnitude']} | "
            f"direction={item['direction']} | "
            f"value={value}"
        )

    report = {
        "stage": "day7_explainability_verification",
        "report_type": (
            "Explainability verification report, "
            "not a model performance report."
        ),
        "model": {
            "path": str(
                MODEL_PATH.relative_to(ROOT)
            ),
            "model_type": type(
                pipeline.named_steps["model"]
            ).__name__,
            "pipeline_type": type(pipeline).__name__,
            "input_feature_count": input_feature_count,
            "input_features": explainer.input_features,
            "transformed_feature_count": transformed_feature_count,
            "transformed_feature_names": (
                explainer.transformed_feature_names
            ),
        },
        "data": {
            "path": str(
                DATA_PATH.relative_to(ROOT)
            ),
            "rows": int(len(dataset)),
            "columns": int(len(dataset.columns)),
            "synthetic": True,
        },
        "shap": {
            "available": shap_available,
            "version": shap_version,
            "explainer_type": "LinearExplainer",
        },
        "verification": {
            "feature_order_verified": feature_order_verified,
            "transformed_feature_alignment_verified": (
                transformed_alignment_verified
            ),
            "local_explanation_generated": (
                local_explanation_generated
            ),
            "contribution_ranking_verified": (
                contribution_ranking_verified
            ),
            "all_transformed_features_explained": (
                len(all_contributors) == FULL_FEATURE_COUNT
            ),
            "total_features_explained": len(all_contributors),
            "non_zero_contribution_count": len(
                non_zero_contributions
            ),
            "zero_contribution_count": len(
                zero_contributions
            ),
        },
        "sample_explanation": {
            "transaction_id": str(transaction_id),
            "top_k": TOP_K,
            "feature_count_explained": int(
                explanation["feature_count_explained"]
            ),
            "top_contributors": [
                {
                    "internal_feature": item[
                        "internal_feature"
                    ],
                    "display_name": item[
                        "display_name"
                    ],
                    "contribution": float(
                        item["contribution"]
                    ),
                    "magnitude": float(
                        item["magnitude"]
                    ),
                    "direction": item["direction"],
                    "feature_value": make_json_safe(
                        item["feature_value"]
                    ),
                }
                for item in top_contributors
            ],
        },
        "limitations": [
            "Dataset is synthetic.",
            (
                "Explanations describe the frozen "
                "Logistic Regression model."
            ),
            (
                "SHAP explanations do not establish "
                "production fraud-detection performance."
            ),
            (
                "Results do not represent Razorpay "
                "production performance."
            ),
            (
                "No production fraud or merchant data "
                "was used."
            ),
        ],
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n--- REPORT ---")
    print(f"Report written: {REPORT_PATH}")

    print("\n" + "=" * 70)
    print("DAY 7 EXPLAINABILITY VERIFICATION: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()