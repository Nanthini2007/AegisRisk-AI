"""
Three-level merchant decision policy for AegisRisk AI.

Policy:
    LOW RISK    -> ALLOW
    MEDIUM RISK -> REVIEW
    HIGH RISK   -> HOLD FOR VERIFICATION

This module converts model probabilities into operational actions.
The boundaries are configurable and must be documented separately from
model performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


ALLOW = "ALLOW"
REVIEW = "REVIEW"
HOLD_FOR_VERIFICATION = "HOLD_FOR_VERIFICATION"


@dataclass(frozen=True)
class DecisionPolicy:
    """
    Configurable three-level fraud-risk policy.

    review_threshold:
        Probability at which a transaction moves from ALLOW to REVIEW.

    hold_threshold:
        Probability at which a transaction moves from REVIEW to
        HOLD FOR VERIFICATION.

    Required:
        0 <= review_threshold < hold_threshold <= 1
    """

    review_threshold: float = 0.35
    hold_threshold: float = 0.70

    def __post_init__(self) -> None:
        if not 0.0 <= self.review_threshold <= 1.0:
            raise ValueError(
                "review_threshold must be between 0 and 1"
            )

        if not 0.0 <= self.hold_threshold <= 1.0:
            raise ValueError(
                "hold_threshold must be between 0 and 1"
            )

        if self.review_threshold >= self.hold_threshold:
            raise ValueError(
                "review_threshold must be lower than hold_threshold"
            )


def decide(
    probability: float,
    policy: DecisionPolicy,
) -> str:
    """Return the operational action for one fraud probability."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "probability must be between 0 and 1"
        )

    if probability >= policy.hold_threshold:
        return HOLD_FOR_VERIFICATION

    if probability >= policy.review_threshold:
        return REVIEW

    return ALLOW


def decide_many(
    probabilities: Iterable[float],
    policy: DecisionPolicy,
) -> list[str]:
    """Return operational actions for multiple probabilities."""

    probability_array = np.asarray(
        list(probabilities),
        dtype=float,
    )

    return [
        decide(
            float(probability),
            policy,
        )
        for probability in probability_array
    ]


def policy_to_dict(
    policy: DecisionPolicy,
) -> dict[str, object]:
    """Convert policy configuration into a report-friendly dictionary."""

    return {
        "review_threshold": policy.review_threshold,
        "hold_threshold": policy.hold_threshold,
        "low_risk": {
            "range": (
                f"[0.00, "
                f"{policy.review_threshold:.2f})"
            ),
            "action": ALLOW,
        },
        "medium_risk": {
            "range": (
                f"[{policy.review_threshold:.2f}, "
                f"{policy.hold_threshold:.2f})"
            ),
            "action": REVIEW,
        },
        "high_risk": {
            "range": (
                f"[{policy.hold_threshold:.2f}, 1.00]"
            ),
            "action": HOLD_FOR_VERIFICATION,
        },
    }