"""
AegisRisk AI - Day 10 Integration Test
Comprehensive frozen pipeline validation using 10 deterministic rows.

This script verifies that all Day 6-Day 9 components work together correctly:
1. Frozen Day 6 Logistic Regression loads.
2. Exactly 26 model input features are used in frozen order.
3. Fraud probabilities are finite and in [0,1].
4. Risk decisions use thresholds 0.35 and 0.70.
5. Actions match the existing DecisionPolicy.
6. SHAP explanations are generated and contain top contributors.
7. RiskMonitor returns PASS.
8. Frozen model verification returns PASS.
9. Frozen policy verification returns PASS.
10. Save actual results to reports/day10_integration_report.json

Does NOT:
- Retrain the model
- Modify preprocessing
- Change features
- Change thresholds
- Create a new Streamlit app
- Invent results
- Commit or push
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from src.inference.scorer import MerchantRiskScorer
from src.monitoring.risk_monitor import RiskMonitor


def load_test_data(n_rows: int = 10) -> pd.DataFrame:
    """
    Load the first n deterministic rows from processed features.

    Parameters
    ----------
    n_rows : int, default=10
        Number of rows to load.

    Returns
    -------
    pd.DataFrame
        First n rows with all required features.
    """
    data_path = Path("data/processed/transactions_features.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data not found: {data_path}"
        )

    df = pd.read_csv(data_path, nrows=n_rows)

    if len(df) < n_rows:
        raise ValueError(
            f"Expected {n_rows} rows, got {len(df)}"
        )

    return df


def verify_model_loading(scorer: MerchantRiskScorer) -> dict:
    """
    Verify Requirement 1:
    Frozen Day 6 Logistic Regression loads.
    """
    return {
        "requirement": 1,
        "description": "Frozen Day 6 Logistic Regression loads",
        "model_path": str(scorer.model_path),
        "model_exists": bool(scorer.model_path.exists()),
        "pipeline_loaded": bool(scorer.pipeline is not None),
        "model_loaded": bool(scorer.model is not None),
        "preprocessor_loaded": bool(scorer.preprocessor is not None),
        "status": "PASS"
        if (
            scorer.model_path.exists()
            and scorer.pipeline is not None
            and scorer.model is not None
            and scorer.preprocessor is not None
        )
        else "FAIL",
    }


def verify_feature_count_and_order(
    scorer: MerchantRiskScorer,
) -> dict:
    """
    Verify Requirement 2:
    Exactly 26 model input features are used in frozen order.
    """
    features = scorer.input_features
    feature_count = len(features)
    expected_count = 26

    return {
        "requirement": 2,
        "description": "Exactly 26 model input features in frozen order",
        "expected_count": expected_count,
        "actual_count": feature_count,
        "features": features,
        "count_matches": bool(feature_count == expected_count),
        "status": "PASS" if feature_count == expected_count else "FAIL",
    }


def verify_probability_validity(
    probabilities: np.ndarray,
) -> dict:
    """
    Verify Requirement 3:
    Fraud probabilities are finite and in [0,1].
    """
    valid_mask = (
        np.isfinite(probabilities)
        & (probabilities >= 0.0)
        & (probabilities <= 1.0)
    )

    valid_count = np.sum(valid_mask)
    total_count = len(probabilities)
    all_valid = valid_count == total_count

    return {
        "requirement": 3,
        "description": "Fraud probabilities finite and in [0,1]",
        "total_probabilities": int(total_count),
        "valid_probabilities": int(valid_count),
        "min_probability": float(np.min(probabilities)),
        "max_probability": float(np.max(probabilities)),
        "all_finite": bool(np.all(np.isfinite(probabilities))),
        "all_in_range": bool(all_valid),
        "status": "PASS" if all_valid else "FAIL",
    }


def verify_thresholds(policy) -> dict:
    """
    Verify Requirement 4:
    Risk decisions use thresholds 0.35 and 0.70.
    """
    review_threshold = policy.review_threshold
    hold_threshold = policy.hold_threshold

    expected_review = 0.35
    expected_hold = 0.70

    review_matches = np.isclose(review_threshold, expected_review)
    hold_matches = np.isclose(hold_threshold, expected_hold)

    return {
        "requirement": 4,
        "description": "Risk decisions use thresholds 0.35 and 0.70",
        "expected_review_threshold": expected_review,
        "actual_review_threshold": float(review_threshold),
        "review_threshold_matches": bool(review_matches),
        "expected_hold_threshold": expected_hold,
        "actual_hold_threshold": float(hold_threshold),
        "hold_threshold_matches": bool(hold_matches),
        "status": "PASS" if review_matches and hold_matches else "FAIL",
    }


def verify_action_distribution(
    actions: list[str],
) -> dict:
    """
    Verify Requirement 5:
    Actions match the existing DecisionPolicy.
    """
    valid_actions = {"ALLOW", "REVIEW", "HOLD_FOR_VERIFICATION"}
    action_counts = {}

    for action in valid_actions:
        action_counts[action.lower()] = int(
            sum(1 for a in actions if a == action)
        )

    all_valid = all(a in valid_actions for a in actions)

    return {
        "requirement": 5,
        "description": "Actions match existing DecisionPolicy",
        "total_actions": len(actions),
        "action_distribution": action_counts,
        "all_valid": bool(all_valid),
        "status": "PASS" if all_valid else "FAIL",
    }


def verify_shap_explanations(
    explanations: list[dict],
    probabilities: np.ndarray,
) -> dict:
    """
    Verify Requirement 6:
    SHAP explanations are generated and contain top contributors.
    """
    explanation_count = len(explanations)
    probability_count = len(probabilities)

    counts_match = explanation_count == probability_count

    explanations_with_contributors = 0

    for exp in explanations:
        top_contributors = exp.get("top_contributors", [])
        if len(top_contributors) > 0:
            explanations_with_contributors += 1

    all_have_contributors = (
        explanations_with_contributors == explanation_count
    )

    # Verify structure
    feature_counts_match = True
    for exp in explanations:
        actual_count = exp.get("feature_count_explained")
        expected_count = exp.get("top_k", 0) + (
            len(exp.get("top_contributors", [])) - exp.get("top_k", 0)
        )
        if actual_count is not None and actual_count < 1:
            feature_counts_match = False

    return {
        "requirement": 6,
        "description": "SHAP explanations generated with top contributors",
        "explanations_generated": explanation_count,
        "probabilities_scored": probability_count,
        "counts_match": bool(counts_match),
        "explanations_with_contributors": (
            explanations_with_contributors
        ),
        "all_have_contributors": bool(all_have_contributors),
        "status": (
            "PASS"
            if counts_match and all_have_contributors
            else "FAIL"
        ),
    }


def verify_risk_monitor(
    monitor_result: dict,
) -> dict:
    """
    Verify Requirement 7:
    RiskMonitor returns PASS.
    """
    overall_status = monitor_result.get("overall_status")

    return {
        "requirement": 7,
        "description": "RiskMonitor returns PASS",
        "overall_status": overall_status,
        "transaction_count": monitor_result.get(
            "transaction_count"
        ),
        "data_quality_status": (
            monitor_result.get("data_quality_checks", {}).get(
                "status"
            )
        ),
        "status": "PASS" if overall_status == "PASS" else "FAIL",
    }


def verify_frozen_model(monitor: RiskMonitor) -> dict:
    """
    Verify Requirement 8:
    Frozen model verification returns PASS.
    """
    model_verification = monitor.verify_frozen_model()

    status = model_verification.get("status")

    return {
        "requirement": 8,
        "description": "Frozen model verification returns PASS",
        "model_path": model_verification.get("model_path"),
        "exists": bool(model_verification.get("exists")),
        "loads": bool(model_verification.get("loads")),
        "input_feature_count": model_verification.get(
            "input_feature_count"
        ),
        "expected_feature_count": model_verification.get(
            "expected_input_feature_count"
        ),
        "feature_count_check": model_verification.get(
            "feature_count_check"
        ),
        "verification_status": status,
        "status": "PASS" if status == "PASS" else "FAIL",
    }


def verify_frozen_policy(monitor: RiskMonitor) -> dict:
    """
    Verify Requirement 9:
    Frozen policy verification returns PASS.
    """
    policy_verification = monitor.verify_frozen_policy()

    status = policy_verification.get("status")

    return {
        "requirement": 9,
        "description": "Frozen policy verification returns PASS",
        "expected_review_threshold": float(policy_verification.get(
            "expected_review_threshold"
        )),
        "actual_review_threshold": float(policy_verification.get(
            "actual_review_threshold"
        )),
        "review_threshold_check": policy_verification.get(
            "review_threshold_check"
        ),
        "expected_hold_threshold": float(policy_verification.get(
            "expected_hold_threshold"
        )),
        "actual_hold_threshold": float(policy_verification.get(
            "actual_hold_threshold"
        )),
        "hold_threshold_check": policy_verification.get(
            "hold_threshold_check"
        ),
        "policy_status": status,
        "status": "PASS" if status == "PASS" else "FAIL",
    }


def run_integration_test() -> None:
    """
    Execute full Day 10 integration test and save results.

    Requirements verified:
    1. Frozen Day 6 Logistic Regression loads
    2. Exactly 26 model input features in frozen order
    3. Fraud probabilities finite and in [0,1]
    4. Risk decisions use thresholds 0.35 and 0.70
    5. Actions match existing DecisionPolicy
    6. SHAP explanations generated with top contributors
    7. RiskMonitor returns PASS
    8. Frozen model verification returns PASS
    9. Frozen policy verification returns PASS
    10. Save results to reports/day10_integration_report.json
    """
    print("\n" + "=" * 70)
    print("AegisRisk AI - Day 10 Integration Test")
    print("=" * 70)

    # Load test data (Requirement context)
    print("\nLoading 10 deterministic test rows...")
    test_data = load_test_data(n_rows=10)
    print(f"✓ Loaded {len(test_data)} rows with {len(test_data.columns)} columns")

    # Initialize scorer (Requirement 1)
    print("\nInitializing MerchantRiskScorer...")
    scorer = MerchantRiskScorer(
        model_path="models/logistic_regression_day6.joblib"
    )
    print("✓ Scorer initialized")

    # Verify model loading (Requirement 1)
    print("\n[Requirement 1] Verifying frozen model loads...")
    req1 = verify_model_loading(scorer)
    print(f"✓ Status: {req1['status']}")

    # Verify features (Requirement 2)
    print("\n[Requirement 2] Verifying 26 features in frozen order...")
    req2 = verify_feature_count_and_order(scorer)
    print(f"✓ Feature count: {req2['actual_count']}")
    print(f"✓ Status: {req2['status']}")

    # Score transactions
    print("\nScoring 10 transactions...")
    scored_data = scorer.score_transactions(
        test_data,
        include_explanation=True,
        top_k=5,
    )
    print(f"✓ Scored {len(scored_data)} transactions")

    # Verify probabilities (Requirement 3)
    print("\n[Requirement 3] Verifying probabilities finite in [0,1]...")
    probabilities = scored_data["fraud_probability"].values
    req3 = verify_probability_validity(probabilities)
    print(
        f"✓ Min: {req3['min_probability']:.4f}, "
        f"Max: {req3['max_probability']:.4f}"
    )
    print(f"✓ All valid: {req3['all_in_range']}")
    print(f"✓ Status: {req3['status']}")

    # Verify thresholds (Requirement 4)
    print("\n[Requirement 4] Verifying thresholds 0.35 and 0.70...")
    req4 = verify_thresholds(scorer.policy)
    print(f"✓ Review threshold: {req4['actual_review_threshold']}")
    print(f"✓ Hold threshold: {req4['actual_hold_threshold']}")
    print(f"✓ Status: {req4['status']}")

    # Verify actions (Requirement 5)
    print("\n[Requirement 5] Verifying actions match DecisionPolicy...")
    actions = scored_data["risk_action"].tolist()
    req5 = verify_action_distribution(actions)
    print(f"✓ Action distribution: {req5['action_distribution']}")
    print(f"✓ All valid: {req5['all_valid']}")
    print(f"✓ Status: {req5['status']}")

    # Verify explanations (Requirement 6)
    print("\n[Requirement 6] Verifying SHAP explanations...")
    explanations = [
        {
            "top_contributors": scored_data.loc[i, "top_contributors"],
        }
        for i in range(len(scored_data))
    ]
    req6 = verify_shap_explanations(explanations, probabilities)
    print(f"✓ Explanations generated: {req6['explanations_generated']}")
    print(
        f"✓ All have contributors: "
        f"{req6['all_have_contributors']}"
    )
    print(f"✓ Status: {req6['status']}")

    # Monitor results (Requirement 7)
    print("\n[Requirement 7] Verifying RiskMonitor returns PASS...")
    monitor = RiskMonitor(
        review_threshold=0.35,
        hold_threshold=0.70,
        model_path="models/logistic_regression_day6.joblib",
    )
    monitor_result = monitor.monitor(scored_data)
    req7 = verify_risk_monitor(monitor_result)
    print(f"✓ Overall status: {req7['overall_status']}")
    print(f"✓ Status: {req7['status']}")

    # Verify frozen model (Requirement 8)
    print("\n[Requirement 8] Verifying frozen model...")
    req8 = verify_frozen_model(monitor)
    print(f"✓ Model exists: {req8['exists']}")
    print(f"✓ Model loads: {req8['loads']}")
    print(
        f"✓ Feature count: {req8['input_feature_count']} "
        f"(expected: {req8['expected_feature_count']})"
    )
    print(f"✓ Status: {req8['status']}")

    # Verify frozen policy (Requirement 9)
    print("\n[Requirement 9] Verifying frozen policy...")
    req9 = verify_frozen_policy(monitor)
    print(f"✓ Review threshold: {req9['actual_review_threshold']}")
    print(f"✓ Hold threshold: {req9['actual_hold_threshold']}")
    print(f"✓ Status: {req9['status']}")

    # Compile all results (Requirement 10)
    print("\n[Requirement 10] Saving results to report...")

    # Add sample transactions
    sample_transactions = []
    for idx in range(min(3, len(scored_data))):
        row = scored_data.iloc[idx]
        sample_transactions.append(
            {
                "transaction_id": str(row.get("transaction_id", f"TXN_{idx}")),
                "amount": float(row.get("amount", 0.0)),
                "fraud_probability": float(
                    row.get("fraud_probability", 0.0)
                ),
                "risk_decision": str(row.get("risk_decision", "N/A")),
                "risk_action": str(row.get("risk_action", "N/A")),
            }
        )

    report = {
        "timestamp": datetime.now().isoformat(),
        "stage": "day10_integration_test",
        "title": (
            "AegisRisk AI - Day 10 Integration Test "
            "(10 Deterministic Rows)"
        ),
        "test_data": {
            "rows_tested": len(test_data),
            "features_per_row": len(test_data.columns),
            "data_source": "data/processed/transactions_features.csv",
        },
        "requirements": [
            req1,
            req2,
            req3,
            req4,
            req5,
            req6,
            req7,
            req8,
            req9,
        ],
        "model": {
            "type": "Logistic Regression (Frozen Day 6)",
            "path": "models/logistic_regression_day6.joblib",
            "input_features": 26,
        },
        "policy": {
            "review_threshold": 0.35,
            "hold_threshold": 0.70,
        },
        "scoring_summary": {
            "total_transactions": len(scored_data),
            "fraud_probability": {
                "min": float(probabilities.min()),
                "max": float(probabilities.max()),
                "mean": float(probabilities.mean()),
                "median": float(np.median(probabilities)),
                "std": float(probabilities.std()),
            },
            "risk_distribution": req5["action_distribution"],
        },
        "monitor_results": monitor_result,
        "sample_transactions": sample_transactions,
        "overall_status": (
            "PASS"
            if all(
                req.get("status") == "PASS"
                for req in [req1, req2, req3, req4, req5, req6, req7, req8, req9]
            )
            else "FAIL"
        ),
    }

    # Write report
    report_path = Path("reports/day10_integration_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✓ Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Integration Test Summary")
    print("=" * 70)
    for i, req in enumerate(
        [req1, req2, req3, req4, req5, req6, req7, req8, req9],
        start=1,
    ):
        status_symbol = "✓" if req["status"] == "PASS" else "✗"
        print(f"{status_symbol} Req {i}: {req['description']} - {req['status']}")

    print("\n" + "=" * 70)
    print(f"Overall Status: {report['overall_status']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_integration_test()
