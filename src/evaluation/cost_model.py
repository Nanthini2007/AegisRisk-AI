"""
Experimental cost model for AegisRisk AI Day 6.

All monetary values are configurable experimental assumptions for
synthetic-data evaluation. They do not represent Razorpay production
economics or actual merchant financial losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class CostModel:
    """
    Configurable experimental cost assumptions.

    fraud_miss_cost:
        Estimated experimental cost assigned to one false negative
        (fraud predicted as legitimate).

    false_positive_cost:
        Estimated experimental intervention/review cost assigned to one
        false positive (legitimate transaction flagged as fraud).

    All values must be non-negative.
    """

    fraud_miss_cost: float
    false_positive_cost: float

    def __post_init__(self) -> None:
        if self.fraud_miss_cost < 0:
            raise ValueError("fraud_miss_cost must be non-negative")

        if self.false_positive_cost < 0:
            raise ValueError("false_positive_cost must be non-negative")


def calculate_costs(
    false_negatives: int,
    false_positives: int,
    cost_model: CostModel,
) -> dict[str, float]:
    """
    Calculate experimental fraud and false-positive costs.

    Total experimental cost =
        (false_negatives × fraud_miss_cost)
        +
        (false_positives × false_positive_cost)
    """

    if false_negatives < 0:
        raise ValueError("false_negatives must be non-negative")

    if false_positives < 0:
        raise ValueError("false_positives must be non-negative")

    fraud_miss_cost = false_negatives * cost_model.fraud_miss_cost
    false_positive_cost = (
        false_positives * cost_model.false_positive_cost
    )
    total_experimental_cost = fraud_miss_cost + false_positive_cost

    return {
        "fraud_miss_cost": float(fraud_miss_cost),
        "false_positive_cost": float(false_positive_cost),
        "total_experimental_cost": float(total_experimental_cost),
    }


def cost_model_to_dict(cost_model: CostModel) -> Mapping[str, float]:
    """Convert cost assumptions into a report-friendly dictionary."""

    return {
        "fraud_miss_cost_per_false_negative": cost_model.fraud_miss_cost,
        "false_positive_cost_per_false_positive": (
            cost_model.false_positive_cost
        ),
    }