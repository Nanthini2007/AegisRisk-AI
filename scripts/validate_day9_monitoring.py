"""
AegisRisk AI - Day 9
Fraud-Risk Monitoring Validation

Loads the existing processed transaction dataset, scores a real subset
using the frozen Day 6 / Day 8 inference pipeline, and runs the reusable
Day 9 RiskMonitor.

This script does not retrain the model, modify preprocessing, change
features, or tune thresholds.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ------------------------------------------------------------------
# Project-root import setup
# ------------------------------------------------------------------
# This allows:
#     python scripts\validate_day9_monitoring.py
# to resolve imports such as:
#     from src.inference.scorer import MerchantRiskScorer
#
# It does NOT modify the model, preprocessing, features, or policy.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd

from src.inference.scorer import MerchantRiskScorer
from src.monitoring.risk_monitor import RiskMonitor


# ------------------------------------------------------------------
# Repository paths
# ------------------------------------------------------------------

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "logistic_regression_day6.joblib"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day9_monitoring_report.json"
)


# ------------------------------------------------------------------
# Validation configuration
# ------------------------------------------------------------------
# This is only the number of existing transactions used for monitoring.
# It is NOT a training parameter and is NOT used for threshold tuning.
VALIDATION_SAMPLE_SIZE = 2_000


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------


def make_json_safe(value: Any) -> Any:
    """
    Convert common NumPy/Pandas values into JSON-safe Python values.
    """

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            make_json_safe(item)
            for item in value
        ]

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value


def print_section(title: str) -> None:
    """Print a consistent terminal section header."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------------
# Main validation workflow
# ------------------------------------------------------------------


def main() -> int:

    print("=" * 70)
    print("AEGISRISK AI - DAY 9")
    print("FRAUD-RISK MONITORING & MODEL HEALTH")
    print("=" * 70)

    # ==============================================================
    # 1. Verify source data
    # ==============================================================

    print_section("SOURCE DATA")

    if not DATA_PATH.exists():
        print(
            f"ERROR: Processed dataset not found:\n"
            f"{DATA_PATH}"
        )
        return 1

    try:
        source_data = pd.read_csv(DATA_PATH)
    except Exception as exc:
        print(
            f"ERROR: Failed to load processed dataset:\n"
            f"{exc}"
        )
        return 1

    print(f"Source dataset: {DATA_PATH}")
    print(f"Source rows: {len(source_data):,}")
    print(f"Source columns: {len(source_data.columns)}")

    if source_data.empty:
        print("ERROR: Source dataset is empty.")
        return 1

    # ==============================================================
    # 2. Initialize frozen scorer
    # ==============================================================

    print_section("FROZEN SCORER")

    if not MODEL_PATH.exists():
        print(
            f"ERROR: Frozen model not found:\n"
            f"{MODEL_PATH}"
        )
        return 1

    try:
        scorer = MerchantRiskScorer(
            model_path=MODEL_PATH,
        )
    except Exception as exc:
        print(
            f"ERROR: Frozen scorer initialization failed:\n"
            f"{exc}"
        )
        return 1

    scorer_metrics = scorer.get_metrics()

    print(f"Frozen model: {MODEL_PATH}")
    print(f"Model type: {scorer_metrics['model_type']}")
    print(
        "Input feature count: "
        f"{len(scorer.input_features)}"
    )
    print(
        "Review threshold: "
        f"{scorer.policy.review_threshold}"
    )
    print(
        "Hold threshold: "
        f"{scorer.policy.hold_threshold}"
    )

    # ==============================================================
    # 3. Initialize Day 9 monitor using existing frozen policy
    # ==============================================================

    try:
        monitor = RiskMonitor(
            review_threshold=scorer.policy.review_threshold,
            hold_threshold=scorer.policy.hold_threshold,
            model_path=MODEL_PATH,
        )
    except Exception as exc:
        print(
            f"ERROR: RiskMonitor initialization failed:\n"
            f"{exc}"
        )
        return 1

    # ==============================================================
    # 4. Frozen model verification
    # ==============================================================

    print_section("FROZEN MODEL VERIFICATION")

    try:
        frozen_model_check = (
            monitor.verify_frozen_model()
        )
    except Exception as exc:
        print(
            f"ERROR: Frozen model verification failed:\n"
            f"{exc}"
        )
        return 1

    print(
        "Frozen model exists: "
        f"{frozen_model_check['exists']}"
    )

    print(
        "Frozen model loads: "
        f"{frozen_model_check['loads']}"
    )

    print(
        "Frozen input feature count: "
        f"{frozen_model_check['input_feature_count']}"
    )

    print(
        "Feature-count check: "
        f"{frozen_model_check['feature_count_check']}"
    )

    # ==============================================================
    # 5. Frozen policy verification
    # ==============================================================

    print_section("FROZEN POLICY VERIFICATION")

    try:
        frozen_policy_check = (
            monitor.verify_frozen_policy()
        )
    except Exception as exc:
        print(
            f"ERROR: Frozen policy verification failed:\n"
            f"{exc}"
        )
        return 1

    print(
        "Expected review threshold: "
        f"{frozen_policy_check['expected_review_threshold']}"
    )

    print(
        "Actual review threshold: "
        f"{frozen_policy_check['actual_review_threshold']}"
    )

    print(
        "Review threshold check: "
        f"{frozen_policy_check['review_threshold_check']}"
    )

    print(
        "Expected hold threshold: "
        f"{frozen_policy_check['expected_hold_threshold']}"
    )

    print(
        "Actual hold threshold: "
        f"{frozen_policy_check['actual_hold_threshold']}"
    )

    print(
        "Hold threshold check: "
        f"{frozen_policy_check['hold_threshold_check']}"
    )

    # ==============================================================
    # 6. Select real transaction subset
    # ==============================================================

    print_section("REAL VALIDATION SUBSET")

    sample_size = min(
        VALIDATION_SAMPLE_SIZE,
        len(source_data),
    )

    validation_data = (
        source_data
        .head(sample_size)
        .copy()
    )

    print(
        "Transactions selected for monitoring: "
        f"{len(validation_data):,}"
    )

    print(
        "Selection: first rows of the existing "
        "processed dataset"
    )

    print(
        "Synthetic transactions created: False"
    )

    # ==============================================================
    # 7. Score using existing frozen inference pipeline
    # ==============================================================

    print_section("FROZEN SCORING")

    try:
        scored_data = scorer.score_transactions(
            validation_data,
            include_explanation=False,
        )
    except Exception as exc:
        print(
            f"ERROR: Frozen scoring failed:\n"
            f"{exc}"
        )
        return 1

    print(
        "Transactions successfully scored: "
        f"{len(scored_data):,}"
    )

    # ==============================================================
    # 8. Run Day 9 monitoring
    # ==============================================================

    print_section("RISK MONITORING")

    try:
        monitoring_result = monitor.monitor(
            scored_data
        )
    except Exception as exc:
        print(
            f"ERROR: Day 9 monitoring failed:\n"
            f"{exc}"
        )
        return 1

    # ==============================================================
    # 9. Probability statistics
    # ==============================================================

    print_section("PROBABILITY STATISTICS")

    probability_statistics = (
        monitoring_result[
            "probability_statistics"
        ]
    )

    for key, value in probability_statistics.items():

        if key == "percentiles":
            continue

        print(
            f"{key}: {value}"
        )

    if "percentiles" in probability_statistics:

        print("percentiles:")

        for key, value in (
            probability_statistics[
                "percentiles"
            ].items()
        ):
            print(
                f"  {key}: {value}"
            )

    # ==============================================================
    # 10. Risk distribution
    # ==============================================================

    print_section("RISK DISTRIBUTION")

    risk_distribution = (
        monitoring_result[
            "risk_distribution"
        ]
    )

    for risk_level, values in (
        risk_distribution.items()
    ):

        print(
            f"{risk_level}: "
            f"{values['count']:,} "
            f"({values['percentage']:.2f}%)"
        )

    # ==============================================================
    # 11. Action distribution
    # ==============================================================

    print_section("ACTION DISTRIBUTION")

    action_distribution = (
        monitoring_result[
            "action_distribution"
        ]
    )

    for action, values in (
        action_distribution.items()
    ):

        print(
            f"{action}: "
            f"{values['count']:,} "
            f"({values['percentage']:.2f}%)"
        )

    # ==============================================================
    # 12. Policy consistency
    # ==============================================================

    print_section("POLICY CONSISTENCY")

    policy_verification = (
        monitoring_result[
            "policy_verification"
        ]
    )

    policy_consistency = (
        policy_verification[
            "policy_consistency"
        ]
    )

    print(
        "Policy consistency: "
        f"{policy_consistency['status']}"
    )

    print(
        "Transactions checked: "
        f"{policy_consistency.get('checked_count')}"
    )

    print(
        "Inconsistent transactions: "
        f"{policy_consistency.get('inconsistent_count')}"
    )

    # ==============================================================
    # 13. Data-quality checks
    # ==============================================================

    print_section("DATA-QUALITY CHECKS")

    data_quality_checks = (
        monitoring_result[
            "data_quality_checks"
        ]
    )

    for check_name, check_result in (
        data_quality_checks.items()
    ):

        print(
            f"{check_name}: "
            f"{check_result.get('status')}"
        )

    # ==============================================================
    # 14. Overall monitoring status
    # ==============================================================

    overall_status = (
        monitoring_result[
            "overall_status"
        ]
    )

    print_section("OVERALL MONITORING STATUS")

    print(
        f"STATUS: {overall_status}"
    )

    # ==============================================================
    # 15. Warnings and alerts
    # ==============================================================

    warnings = monitoring_result.get(
        "warnings",
        [],
    )

    alerts = monitoring_result.get(
        "alerts",
        [],
    )

    print(
        f"Warnings: {len(warnings)}"
    )

    print(
        f"Alerts: {len(alerts)}"
    )

    if warnings:
        print("Warning details:")

        for warning in warnings:
            print(
                f"  - {warning}"
            )

    if alerts:
        print("Alert details:")

        for alert in alerts:
            print(
                f"  - {alert}"
            )

    # ==============================================================
    # 16. Build report
    # ==============================================================

    print_section("GENERATING REPORT")

    report = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "stage": (
            "Day 9 - Fraud-Risk Monitoring "
            "& Model Health"
        ),

        "monitoring_methodology": (
            "Reusable integrity and descriptive "
            "monitoring of transactions scored by "
            "the existing frozen AegisRisk inference "
            "pipeline. No retraining, feature changes, "
            "threshold tuning, or held-out test-set "
            "tuning is performed."
        ),

        "source_data": {
            "path": str(
                DATA_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "source_row_count": int(
                len(source_data)
            ),
            "source_column_count": int(
                len(source_data.columns)
            ),
            "validation_selection": (
                "First rows of the existing "
                "processed dataset"
            ),
            "synthetic_transactions_created": False,
        },

        "number_of_transactions_monitored": int(
            len(scored_data)
        ),

        "probability_statistics": (
            probability_statistics
        ),

        "risk_distribution": (
            risk_distribution
        ),

        "action_distribution": (
            action_distribution
        ),

        "policy_verification": (
            policy_verification
        ),

        "data_quality_checks": (
            data_quality_checks
        ),

        "frozen_component_checks": {
            "model": frozen_model_check,
            "policy": frozen_policy_check,
        },

        "warnings": warnings,

        "alerts": alerts,

        "overall_status": (
            overall_status
        ),

        "limitations": [
            (
                "Monitoring was performed on a "
                "validation subset of the existing "
                "processed dataset."
            ),
            (
                "Observed distributions are "
                "descriptive and are not treated as "
                "scientifically validated production "
                "limits."
            ),
            (
                "No production Razorpay transaction "
                "data was used."
            ),
            (
                "No production-readiness claim is made."
            ),
            (
                "This monitoring run does not establish "
                "model drift or future production "
                "performance."
            ),
        ],
    }

    report = make_json_safe(report)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with REPORT_PATH.open(
            "w",
            encoding="utf-8",
        ) as report_file:

            json.dump(
                report,
                report_file,
                indent=2,
            )

    except Exception as exc:

        print(
            f"ERROR: Failed to write report:\n"
            f"{exc}"
        )

        return 1

    print(
        "Report generated:"
    )

    print(
        REPORT_PATH
    )

    # ==============================================================
    # 17. Required validation checks
    # ==============================================================

    required_checks = [

        frozen_model_check["exists"],

        frozen_model_check["loads"],

        frozen_model_check[
            "feature_count_check"
        ] == "PASS",

        frozen_policy_check[
            "review_threshold_check"
        ] == "PASS",

        frozen_policy_check[
            "hold_threshold_check"
        ] == "PASS",

        frozen_policy_check[
            "status"
        ] == "PASS",

        overall_status == "PASS",
    ]

    print_section("VALIDATION RESULT")

    if not all(required_checks):

        print(
            "VALIDATION RESULT: FAILED"
        )

        print(
            "One or more required checks "
            "did not pass."
        )

        return 1

    print(
        "VALIDATION RESULT: PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )