import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference.scorer import MerchantRiskScorer
from src.monitoring.risk_monitor import RiskMonitor
from src.evaluation.decision_policy import DecisionPolicy, decide

PROCESSED_PATH = Path("data/processed/transactions_features.csv")
REPORT_PATH = Path("reports/day10_integration_report.json")
SAMPLE_SIZE = 10


def load_processed_or_skip():
    if not PROCESSED_PATH.exists():
        pytest.skip(f"Processed dataset missing at {PROCESSED_PATH}")
    return pd.read_csv(PROCESSED_PATH)


def test_deterministic_sample_selection():
    df = load_processed_or_skip()
    sample = df.iloc[:SAMPLE_SIZE].reset_index(drop=True)
    assert len(sample) <= SAMPLE_SIZE
    if len(sample) > 0:
        assert sample.iloc[0].to_dict() == df.iloc[0].to_dict()


def test_scoring_and_probabilities_valid():
    df = load_processed_or_skip()
    sample = df.iloc[:SAMPLE_SIZE].reset_index(drop=True)
    scorer = MerchantRiskScorer()
    scored = scorer.score_transactions(sample, include_explanation=True, top_k=3)

    probs = pd.to_numeric(scored["fraud_probability"], errors="coerce")
    assert not probs.isna().any()
    assert np.all(np.isfinite(probs))
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert scorer.get_metrics()["input_features"] == 26


def test_boundary_policy_behavior_and_consistency():
    df = load_processed_or_skip()
    sample = df.iloc[:SAMPLE_SIZE].reset_index(drop=True)
    scorer = MerchantRiskScorer()
    scored = scorer.score_transactions(sample, include_explanation=False)

    policy = DecisionPolicy(review_threshold=0.35, hold_threshold=0.70)
    for prob, action in zip(scored["fraud_probability"], scored["risk_action"]):
        expected_action = decide(float(prob), policy)
        assert expected_action == action

    monitor = RiskMonitor()
    monitor_result = monitor.monitor(scored)
    assert "overall_status" in monitor_result


def test_shap_explanations_present():
    df = load_processed_or_skip()
    sample = df.iloc[:SAMPLE_SIZE].reset_index(drop=True)
    scorer = MerchantRiskScorer()
    scored = scorer.score_transactions(sample, include_explanation=True, top_k=5)

    assert "top_contributors" in scored.columns
    for entry in scored["top_contributors"].tolist():
        assert isinstance(entry, list)


def test_risk_monitor_and_frozen_checks_produced_report(tmp_path):
    df = load_processed_or_skip()
    sample = df.iloc[:SAMPLE_SIZE].reset_index(drop=True)
    scorer = MerchantRiskScorer()
    scored = scorer.score_transactions(sample, include_explanation=False)

    monitor = RiskMonitor()
    monitor_result = monitor.monitor(scored)
    frozen_model_check = monitor.verify_frozen_model()
    frozen_policy_check = monitor.verify_frozen_policy()

    assert "loads" in frozen_model_check
    assert frozen_policy_check["expected_review_threshold"] == pytest.approx(0.35)
    assert frozen_policy_check["expected_hold_threshold"] == pytest.approx(0.70)
