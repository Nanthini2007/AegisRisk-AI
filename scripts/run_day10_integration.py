#!/usr/bin/env python3
"""
Day 10 deterministic integration script.

Saves results to reports/day10_integration_report.json.

This script intentionally:
 - does not retrain or modify the frozen model
 - preserves the frozen feature order
 - does not change thresholds or policy
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference.scorer import MerchantRiskScorer
from src.monitoring.risk_monitor import RiskMonitor
from src.evaluation.decision_policy import policy_to_dict

REPORT_PATH = Path("reports/day10_integration_report.json")
# Updated to use the actual processed dataset tracked in the repository
DEFAULT_PROCESSED_PATH = Path("data/processed/transactions_features.csv")
SAMPLE_SIZE = 10


def load_processed(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {path}")
    df = pd.read_csv(path)
    return df


def sample_deterministic(df: pd.DataFrame, n: int) -> pd.DataFrame:
    # Deterministic selection: first n rows
    return df.iloc[:n].reset_index(drop=True)


def make_probability_statistics(probs: np.ndarray) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=float)
    return {
        "count": int(len(probs)),
        "minimum": float(np.min(probs)) if len(probs) > 0 else None,
        "maximum": float(np.max(probs)) if len(probs) > 0 else None,
        "mean": float(np.mean(probs)) if len(probs) > 0 else None,
        "median": float(np.median(probs)) if len(probs) > 0 else None,
        "std": float(np.std(probs, ddof=0)) if len(probs) > 0 else None,
    }


def run_integration(
    processed_path: Path = DEFAULT_PROCESSED_PATH,
    sample_size: int = SAMPLE_SIZE,
) -> dict[str, Any]:
    # Load processed dataset
    df = load_processed(processed_path)

    # Deterministic sample
    sample = sample_deterministic(df, sample_size)
    actual_sample_size = int(len(sample))

    # Initialize scorer (uses frozen model)
    scorer = MerchantRiskScorer()

    # Preserve exact frozen feature order is handled inside scorer.score_transactions
    scored = scorer.score_transactions(sample, include_explanation=True, top_k=5)

    # Validate probabilities
    probs = pd.to_numeric(scored["fraud_probability"], errors="coerce")
    invalid_mask = (
        probs.isna()
        | ~np.isfinite(probs)
        | (probs < 0.0)
        | (probs > 1.0)
    )
    probability_validation = {
        "invalid_count": int(invalid_mask.sum()),
        "status": "PASS" if int(invalid_mask.sum()) == 0 else "ALERT",
    }

    # Verify decisions and actions counts
    risk_counts = scored["risk_decision"].value_counts(dropna=False).to_dict()
    action_counts = scored["risk_action"].value_counts(dropna=False).to_dict()

    # Policy consistency via RiskMonitor
    monitor = RiskMonitor()
    monitor_result = monitor.monitor(scored)

    # SHAP explanation verification: exist and top_contributors non-empty
    shap_checks = []
    if "top_contributors" in scored.columns:
        for idx, entry in enumerate(scored["top_contributors"].tolist()):
            exists = isinstance(entry, list) and len(entry) > 0
            shap_checks.append({"index": idx, "has_top_contributors": exists})
    shap_verification = {
        "checked_count": len(shap_checks),
        "all_have_top_contributors": all(c["has_top_contributors"] for c in shap_checks)
        if shap_checks
        else False,
    }

    # Frozen model verification using RiskMonitor helper
    frozen_model_verification = monitor.verify_frozen_model()
    frozen_policy_verification = monitor.verify_frozen_policy()

    # Build the report
    report = {
        "stage": "DAY_10_INTEGRATION",
        "status": monitor_result.get("overall_status"),
        "sample_size": actual_sample_size,
        "source_dataset_path": str(processed_path),
        "model_path": scorer.get_metrics().get("model_path"),
        "model_type": scorer.get_metrics().get("model_type"),
        "feature_count": scorer.get_metrics().get("input_features"),
        "probability_validation": probability_validation,
        "probability_statistics": make_probability_statistics(probs.to_numpy()),
        "risk_decision_counts": {k: int(v) for k, v in (risk_counts.items())},
        "action_counts": {k: int(v) for k, v in (action_counts.items())},
        "policy_consistency_result": monitor_result.get("policy_verification"),
        "shap_explanation_verification": shap_verification,
        "risk_monitor_result": monitor_result,
        "frozen_model_verification": frozen_model_verification,
        "frozen_policy_verification": frozen_policy_verification,
        "warnings": [],
        "limitations": [
            "This integration assumes the processed dataset is the canonical input matching the frozen feature schema.",
            "No models or policies are modified by this script.",
        ],
    }

    # Persist report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf8") as fh:
        json.dump(report, fh, indent=2)

    return report


if __name__ == "__main__":
    try:
        result = run_integration()
        print("Day 10 integration completed. Report written to:", REPORT_PATH)
    except Exception as exc:  # pragma: no cover - top-level runner
        print("Day 10 integration failed:", str(exc))
        raise
