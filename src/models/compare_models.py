"""
AegisRisk AI
Day 5 - Model Comparison

Purpose:
    Compare Logistic Regression, Random Forest,
    and XGBoost using the validation results.

Primary ranking metric:
    PR-AUC

Secondary metrics:
    Precision
    Recall
    F1

Important:
    This module does NOT retrain models.

    It only reads the evaluation report generated
    by evaluate_models.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

REPORT_PATH = Path(
    "reports/day5_evaluation_report.json"
)


# ============================================================
# LOAD REPORT
# ============================================================

def load_report() -> dict:
    """
    Load the Day 5 evaluation report.
    """

    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation report not found: "
            f"{REPORT_PATH}"
        )

    with REPORT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

def build_comparison_table(
    report: dict,
) -> pd.DataFrame:
    """
    Convert evaluation metrics into a
    comparison DataFrame.
    """

    metrics = report[
        "evaluation"
    ][
        "metrics"
    ]

    rows = []

    for model_name, values in (
        metrics.items()
    ):

        rows.append(
            {
                "Model": model_name,

                "Precision": values[
                    "precision"
                ],

                "Recall": values[
                    "recall"
                ],

                "F1": values[
                    "f1"
                ],

                "PR-AUC": values[
                    "pr_auc"
                ],
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    return comparison


# ============================================================
# RANK MODELS
# ============================================================

def rank_models(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank models primarily by PR-AUC.

    Higher PR-AUC is better.
    """

    ranked = (
        comparison
        .sort_values(
            "PR-AUC",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    ranked.insert(
        0,
        "Rank",
        range(
            1,
            len(ranked) + 1,
        ),
    )

    return ranked


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_comparison(
    ranked: pd.DataFrame,
):
    """
    Print a readable model comparison table.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "AegisRisk AI — Day 5 Model Comparison"
    )

    print(
        "=" * 80
    )

    print()

    print(
        ranked.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


# ============================================================
# ANALYZE TRADE-OFF
# ============================================================

def analyze_tradeoff(
    ranked: pd.DataFrame,
):
    """
    Print a simple interpretation of the
    precision-recall trade-off.
    """

    best = ranked.iloc[0]

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MODEL SELECTION ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nCandidate by PR-AUC: "
        f"{best['Model']}"
    )

    print(
        f"PR-AUC: "
        f"{best['PR-AUC']:.4f}"
    )

    print(
        f"Precision: "
        f"{best['Precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best['Recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best['F1']:.4f}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "The model is selected primarily using "
        "PR-AUC, but precision and recall must "
        "also be considered before deploying a "
        "risk policy."
    )

    print(
        "\nImportant:"
    )

    print(
        "The selected model is only a Day 5 "
        "candidate. It is NOT automatically a "
        "transaction blocking decision."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading Day 5 evaluation report..."
    )

    report = load_report()

    comparison = build_comparison_table(
        report
    )

    if comparison.empty:
        raise ValueError(
            "No model evaluation results found."
        )

    required_columns = {
        "Model",
        "Precision",
        "Recall",
        "F1",
        "PR-AUC",
    }

    missing = (
        required_columns
        - set(comparison.columns)
    )

    if missing:
        raise ValueError(
            "Comparison table is missing "
            f"columns: {sorted(missing)}"
        )

    ranked = rank_models(
        comparison
    )

    print_comparison(
        ranked
    )

    analyze_tradeoff(
        ranked
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DAY 5 — STEP 9 COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()