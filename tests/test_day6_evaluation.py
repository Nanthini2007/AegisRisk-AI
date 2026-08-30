import pytest

from src.evaluation.cost_model import (
    CostModel,
    calculate_costs,
)
from src.evaluation.decision_policy import (
    ALLOW,
    HOLD_FOR_VERIFICATION,
    REVIEW,
    DecisionPolicy,
    decide,
)
from src.evaluation.threshold_selection import (
    evaluate_threshold,
    select_lowest_cost_threshold,
)


# ============================================================
# COST MODEL TESTS
# ============================================================

def test_calculate_costs_correctly():
    model = CostModel(
        fraud_miss_cost=1000.0,
        false_positive_cost=50.0,
    )

    result = calculate_costs(
        false_negatives=2,
        false_positives=3,
        cost_model=model,
    )

    assert result["fraud_miss_cost"] == 2000.0
    assert result["false_positive_cost"] == 150.0
    assert result["total_experimental_cost"] == 2150.0


def test_negative_cost_rejected():
    with pytest.raises(ValueError):
        CostModel(
            fraud_miss_cost=-1.0,
            false_positive_cost=50.0,
        )


def test_negative_confusion_count_rejected():
    model = CostModel(
        fraud_miss_cost=1000.0,
        false_positive_cost=50.0,
    )

    with pytest.raises(ValueError):
        calculate_costs(
            false_negatives=-1,
            false_positives=0,
            cost_model=model,
        )


# ============================================================
# THRESHOLD TESTS
# ============================================================

def test_threshold_boundaries():
    model = CostModel(
        fraud_miss_cost=1000.0,
        false_positive_cost=50.0,
    )

    result = evaluate_threshold(
        y_true=[0, 1],
        probabilities=[0.70, 0.70],
        threshold=0.70,
        cost_model=model,
    )

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 0


def test_invalid_threshold_rejected():
    model = CostModel(
        fraud_miss_cost=1000.0,
        false_positive_cost=50.0,
    )

    with pytest.raises(ValueError):
        evaluate_threshold(
            y_true=[0, 1],
            probabilities=[0.2, 0.8],
            threshold=1.1,
            cost_model=model,
        )


def test_lowest_cost_threshold_selected():
    results = [
        {
            "threshold": 0.30,
            "total_experimental_cost": 100.0,
            "recall": 1.0,
            "precision": 0.5,
        },
        {
            "threshold": 0.50,
            "total_experimental_cost": 50.0,
            "recall": 0.5,
            "precision": 0.8,
        },
    ]

    selected = select_lowest_cost_threshold(results)

    assert selected["threshold"] == 0.50


# ============================================================
# DECISION POLICY TESTS
# ============================================================

def test_decision_policy_boundaries():
    policy = DecisionPolicy(
        review_threshold=0.35,
        hold_threshold=0.70,
    )

    assert decide(0.34, policy) == ALLOW
    assert decide(0.35, policy) == REVIEW
    assert decide(0.69, policy) == REVIEW
    assert decide(0.70, policy) == HOLD_FOR_VERIFICATION


def test_invalid_decision_policy_rejected():
    with pytest.raises(ValueError):
        DecisionPolicy(
            review_threshold=0.70,
            hold_threshold=0.35,
        )


def test_invalid_probability_rejected():
    policy = DecisionPolicy(
        review_threshold=0.35,
        hold_threshold=0.70,
    )

    with pytest.raises(ValueError):
        decide(1.1, policy)