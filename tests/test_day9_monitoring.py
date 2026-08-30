"""
AegisRisk AI - Day 9
Tests for Fraud-Risk Monitoring & Model Health.
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.risk_monitor import RiskMonitor


@pytest.fixture
def monitor():
    """Return the default frozen-policy monitor."""
    return RiskMonitor()


@pytest.fixture
def valid_scored_data():
    """Small deterministic scored dataset for unit tests."""
    return pd.DataFrame(
        {
            "fraud_probability": [
                0.10,
                0.20,
                0.349999,
                0.35,
                0.50,
                0.699999,
                0.70,
                0.90,
                1.00,
            ],
            "risk_decision": [
                "LOW_RISK",
                "LOW_RISK",
                "LOW_RISK",
                "MEDIUM_RISK",
                "MEDIUM_RISK",
                "MEDIUM_RISK",
                "HIGH_RISK",
                "HIGH_RISK",
                "HIGH_RISK",
            ],
            "risk_action": [
                "ALLOW",
                "ALLOW",
                "ALLOW",
                "REVIEW",
                "REVIEW",
                "REVIEW",
                "HOLD_FOR_VERIFICATION",
                "HOLD_FOR_VERIFICATION",
                "HOLD_FOR_VERIFICATION",
            ],
        }
    )


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------


def test_successful_initialization():
    monitor = RiskMonitor()

    assert monitor.review_threshold == pytest.approx(0.35)
    assert monitor.hold_threshold == pytest.approx(0.70)


def test_initialization_rejects_invalid_threshold_order():
    with pytest.raises(ValueError):
        RiskMonitor(
            review_threshold=0.70,
            hold_threshold=0.35,
        )


# ------------------------------------------------------------------
# Valid scored data
# ------------------------------------------------------------------


def test_valid_scored_data_returns_pass(
    monitor,
    valid_scored_data,
):
    result = monitor.monitor(valid_scored_data)

    assert result["transaction_count"] == 9
    assert result["overall_status"] == "PASS"


def test_empty_data_is_handled(monitor):
    data = pd.DataFrame(
        columns=[
            "fraud_probability",
            "risk_decision",
            "risk_action",
        ]
    )

    result = monitor.monitor(data)

    assert result["transaction_count"] == 0
    assert result["probability_statistics"]["status"] == "UNAVAILABLE"
    assert result["overall_status"] == "PASS"


# ------------------------------------------------------------------
# Required columns
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_column",
    [
        "fraud_probability",
        "risk_decision",
        "risk_action",
    ],
)
def test_missing_required_columns(
    monitor,
    valid_scored_data,
    missing_column,
):
    data = valid_scored_data.drop(columns=[missing_column])

    result = monitor.monitor(data)

    assert result["overall_status"] == "ALERT"
    assert (
        result["data_quality_checks"]["required_columns"]["status"]
        == "ALERT"
    )


# ------------------------------------------------------------------
# Probability validation
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "probability",
    [
        -0.01,
        -1.0,
        1.01,
        2.0,
    ],
)
def test_probability_outside_range_is_alert(
    monitor,
    valid_scored_data,
    probability,
):
    data = valid_scored_data.copy()
    data.loc[0, "fraud_probability"] = probability

    result = monitor.monitor(data)

    assert result["overall_status"] == "ALERT"
    assert (
        result["data_quality_checks"]["probabilities"]["status"]
        == "ALERT"
    )


@pytest.mark.parametrize(
    "probability",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_probability_value_is_alert(
    monitor,
    valid_scored_data,
    probability,
):
    data = valid_scored_data.copy()
    data.loc[0, "fraud_probability"] = probability

    result = monitor.monitor(data)

    assert result["overall_status"] == "ALERT"


@pytest.mark.parametrize(
    "probability",
    [
        0.0,
        0.35,
        0.70,
        1.0,
    ],
)
def test_probability_boundaries_are_valid(
    monitor,
    valid_scored_data,
    probability,
):
    data = pd.DataFrame(
        {
            "fraud_probability": [probability],
            "risk_decision": [
                (
                    "LOW_RISK"
                    if probability < 0.35
                    else (
                        "MEDIUM_RISK"
                        if probability < 0.70
                        else "HIGH_RISK"
                    )
                )
            ],
            "risk_action": [
                (
                    "ALLOW"
                    if probability < 0.35
                    else (
                        "REVIEW"
                        if probability < 0.70
                        else "HOLD_FOR_VERIFICATION"
                    )
                )
            ],
        }
    )

    result = monitor.monitor(data)

    assert result["overall_status"] == "PASS"


# ------------------------------------------------------------------
# Risk-level validation
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "risk_level",
    [
        "INVALID",
        "LOW",
        "MEDIUM",
        "HIGH",
        "",
        None,
    ],
)
def test_invalid_risk_level_is_alert(
    monitor,
    valid_scored_data,
    risk_level,
):
    data = valid_scored_data.copy()
    data.loc[0, "risk_decision"] = risk_level

    result = monitor.monitor(data)

    assert result["overall_status"] == "ALERT"
    assert (
        result["data_quality_checks"]["risk_levels"]["status"]
        == "ALERT"
    )


@pytest.mark.parametrize(
    "risk_level",
    [
        "LOW_RISK",
        "MEDIUM_RISK",
        "HIGH_RISK",
    ],
)
def test_valid_risk_level_is_accepted(
    monitor,
    risk_level,
):
    probability_map = {
        "LOW_RISK": 0.10,
        "MEDIUM_RISK": 0.50,
        "HIGH_RISK": 0.90,
    }

    action_map = {
        "LOW_RISK": "ALLOW",
        "MEDIUM_RISK": "REVIEW",
        "HIGH_RISK": "HOLD_FOR_VERIFICATION",
    }

    data = pd.DataFrame(
        {
            "fraud_probability": [
                probability_map[risk_level]
            ],
            "risk_decision": [risk_level],
            "risk_action": [action_map[risk_level]],
        }
    )

    result = monitor.monitor(data)

    assert (
        result["data_quality_checks"]["risk_levels"]["status"]
        == "PASS"
    )


# ------------------------------------------------------------------
# Action validation
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        "INVALID",
        "APPROVE",
        "REJECT",
        "",
        None,
    ],
)
def test_invalid_action_is_alert(
    monitor,
    valid_scored_data,
    action,
):
    data = valid_scored_data.copy()
    data.loc[0, "risk_action"] = action

    result = monitor.monitor(data)

    assert result["overall_status"] == "ALERT"
    assert (
        result["data_quality_checks"]["actions"]["status"]
        == "ALERT"
    )


@pytest.mark.parametrize(
    "action, probability",
    [
        ("ALLOW", 0.10),
        ("REVIEW", 0.50),
        ("HOLD_FOR_VERIFICATION", 0.90),
    ],
)
def test_valid_action_is_accepted(
    monitor,
    action,
    probability,
):
    risk_map = {
        "ALLOW": "LOW_RISK",
        "REVIEW": "MEDIUM_RISK",
        "HOLD_FOR_VERIFICATION": "HIGH_RISK",
    }

    data = pd.DataFrame(
        {
            "fraud_probability": [probability],
            "risk_decision": [risk_map[action]],
            "risk_action": [action],
        }
    )

    result = monitor.monitor(data)

    assert (
        result["data_quality_checks"]["actions"]["status"]
        == "PASS"
    )


# ------------------------------------------------------------------
# Policy consistency
# ------------------------------------------------------------------


def test_probability_risk_action_consistency_passes(
    monitor,
    valid_scored_data,
):
    result = monitor.monitor(valid_scored_data)

    consistency = result["policy_verification"][
        "policy_consistency"
    ]

    assert consistency["status"] == "PASS"
    assert consistency["inconsistent_count"] == 0


def test_inconsistent_risk_level_is_detected(
    monitor,
    valid_scored_data,
):
    data = valid_scored_data.copy()

    # 0.10 must be LOW_RISK.
    data.loc[0, "risk_decision"] = "HIGH_RISK"

    result = monitor.monitor(data)

    consistency = result["policy_verification"][
        "policy_consistency"
    ]

    assert result["overall_status"] == "ALERT"
    assert consistency["status"] == "ALERT"
    assert consistency["inconsistent_count"] == 1


def test_inconsistent_action_is_detected(
    monitor,
    valid_scored_data,
):
    data = valid_scored_data.copy()

    # 0.10 must map to ALLOW.
    data.loc[0, "risk_action"] = "HOLD_FOR_VERIFICATION"

    result = monitor.monitor(data)

    consistency = result["policy_verification"][
        "policy_consistency"
    ]

    assert result["overall_status"] == "ALERT"
    assert consistency["status"] == "ALERT"
    assert consistency["inconsistent_count"] == 1


# ------------------------------------------------------------------
# Explicit boundary behavior
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "probability, expected_risk, expected_action",
    [
        (0.349999, "LOW_RISK", "ALLOW"),
        (0.35, "MEDIUM_RISK", "REVIEW"),
        (0.350001, "MEDIUM_RISK", "REVIEW"),
        (0.699999, "MEDIUM_RISK", "REVIEW"),
        (0.70, "HIGH_RISK", "HOLD_FOR_VERIFICATION"),
        (0.700001, "HIGH_RISK", "HOLD_FOR_VERIFICATION"),
    ],
)
def test_policy_boundary_behavior(
    monitor,
    probability,
    expected_risk,
    expected_action,
):
    data = pd.DataFrame(
        {
            "fraud_probability": [probability],
            "risk_decision": [expected_risk],
            "risk_action": [expected_action],
        }
    )

    result = monitor.monitor(data)

    assert result["overall_status"] == "PASS"
    assert (
        result["policy_verification"][
            "policy_consistency"
        ]["inconsistent_count"]
        == 0
    )


# ------------------------------------------------------------------
# Probability statistics
# ------------------------------------------------------------------


def test_probability_statistics(
    monitor,
):
    data = pd.DataFrame(
        {
            "fraud_probability": [
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
            ],
            "risk_decision": [
                "LOW_RISK",
                "LOW_RISK",
                "LOW_RISK",
                "MEDIUM_RISK",
                "MEDIUM_RISK",
            ],
            "risk_action": [
                "ALLOW",
                "ALLOW",
                "ALLOW",
                "REVIEW",
                "REVIEW",
            ],
        }
    )

    result = monitor.monitor(data)
    stats = result["probability_statistics"]

    assert stats["count"] == 5
    assert stats["minimum"] == pytest.approx(0.10)
    assert stats["maximum"] == pytest.approx(0.50)
    assert stats["mean"] == pytest.approx(0.30)
    assert stats["median"] == pytest.approx(0.30)
    assert stats["percentiles"]["p50"] == pytest.approx(0.30)


# ------------------------------------------------------------------
# Distribution calculations
# ------------------------------------------------------------------


def test_risk_distribution_calculation(
    monitor,
):
    data = pd.DataFrame(
        {
            "fraud_probability": [
                0.10,
                0.20,
                0.50,
                0.60,
                0.90,
            ],
            "risk_decision": [
                "LOW_RISK",
                "LOW_RISK",
                "MEDIUM_RISK",
                "MEDIUM_RISK",
                "HIGH_RISK",
            ],
            "risk_action": [
                "ALLOW",
                "ALLOW",
                "REVIEW",
                "REVIEW",
                "HOLD_FOR_VERIFICATION",
            ],
        }
    )

    result = monitor.monitor(data)
    distribution = result["risk_distribution"]

    assert distribution["LOW_RISK"]["count"] == 2
    assert distribution["MEDIUM_RISK"]["count"] == 2
    assert distribution["HIGH_RISK"]["count"] == 1

    assert distribution["LOW_RISK"]["percentage"] == pytest.approx(
        40.0
    )
    assert distribution["MEDIUM_RISK"]["percentage"] == pytest.approx(
        40.0
    )
    assert distribution["HIGH_RISK"]["percentage"] == pytest.approx(
        20.0
    )


def test_action_distribution_calculation(
    monitor,
):
    data = pd.DataFrame(
        {
            "fraud_probability": [
                0.10,
                0.20,
                0.50,
                0.60,
                0.90,
            ],
            "risk_decision": [
                "LOW_RISK",
                "LOW_RISK",
                "MEDIUM_RISK",
                "MEDIUM_RISK",
                "HIGH_RISK",
            ],
            "risk_action": [
                "ALLOW",
                "ALLOW",
                "REVIEW",
                "REVIEW",
                "HOLD_FOR_VERIFICATION",
            ],
        }
    )

    result = monitor.monitor(data)
    distribution = result["action_distribution"]

    assert distribution["ALLOW"]["count"] == 2
    assert distribution["REVIEW"]["count"] == 2
    assert distribution[
        "HOLD_FOR_VERIFICATION"
    ]["count"] == 1

    assert distribution["ALLOW"]["percentage"] == pytest.approx(
        40.0
    )
    assert distribution["REVIEW"]["percentage"] == pytest.approx(
        40.0
    )
    assert distribution[
        "HOLD_FOR_VERIFICATION"
    ]["percentage"] == pytest.approx(20.0)


# ------------------------------------------------------------------
# Frozen component verification
# ------------------------------------------------------------------


def test_frozen_policy_verification(
    monitor,
):
    result = monitor.verify_frozen_policy()

    assert result["review_threshold_check"] == "PASS"
    assert result["hold_threshold_check"] == "PASS"
    assert result["status"] == "PASS"


def test_frozen_model_verification(
    monitor,
):
    result = monitor.verify_frozen_model()

    assert result["exists"] is True
    assert result["loads"] is True
    assert result["input_feature_count"] == 26
    assert result["feature_count_check"] == "PASS"
    assert result["status"] == "PASS"