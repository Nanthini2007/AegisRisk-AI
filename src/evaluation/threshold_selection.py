"""
Validation-only threshold evaluation utilities for AegisRisk AI Day 6.

The held-out test set must never be used for threshold selection.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.evaluation.cost_model import CostModel, calculate_costs


def evaluate_threshold(
    y_true,
    probabilities,
    threshold: float,
    cost_model: CostModel,
) -> dict:
    """Evaluate one probability threshold."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    costs = calculate_costs(
        false_negatives=int(fn),
        false_positives=int(fp),
        cost_model=cost_model,
    )

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        **costs,
    }


def evaluate_thresholds(
    y_true,
    probabilities,
    thresholds: Iterable[float],
    cost_model: CostModel,
) -> list[dict]:
    """Evaluate multiple thresholds on the same dataset."""

    threshold_list = [float(threshold) for threshold in thresholds]

    if not threshold_list:
        raise ValueError("At least one threshold must be provided")

    return [
        evaluate_threshold(
            y_true=y_true,
            probabilities=probabilities,
            threshold=threshold,
            cost_model=cost_model,
        )
        for threshold in threshold_list
    ]


def select_lowest_cost_threshold(
    results: list[dict],
) -> dict:
    """
    Select the threshold with the lowest experimental total cost.

    Tie-breaking:
    1. Higher recall
    2. Higher precision
    3. Higher threshold
    """

    if not results:
        raise ValueError("results must not be empty")

    return min(
        results,
        key=lambda result: (
            result["total_experimental_cost"],
            -result["recall"],
            -result["precision"],
            -result["threshold"],
        ),
    )