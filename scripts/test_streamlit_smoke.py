"""
AegisRisk AI - Day 8
Streamlit Application Smoke Test

Checks:
1. Streamlit app syntax
2. Project imports
3. MerchantRiskScorer functionality

This script is designed to run directly with:

    python scripts\test_streamlit_smoke.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path


# ============================================================
# PROJECT ROOT / IMPORT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# TEST HELPERS
# ============================================================

def test_streamlit_syntax() -> bool:
    """Check Streamlit application syntax."""
    app_path = ROOT_DIR / "app" / "streamlit_app.py"

    print(f"Checking Streamlit app syntax: {app_path.relative_to(ROOT_DIR)}")

    if not app_path.exists():
        print(f"  ✗ App file not found: {app_path}")
        return False

    try:
        source = app_path.read_text(encoding="utf-8")
        compile(source, str(app_path), "exec")
        print("✓ Syntax check passed")
        return True

    except SyntaxError as exc:
        print(f"  ✗ Syntax error: {exc}")
        return False

    except Exception as exc:
        print(f"  ✗ Syntax check failed: {exc}")
        return False


def test_imports() -> bool:
    """Check that required project modules can be imported."""
    print("\nTesting imports...")

    try:
        from src.inference.scorer import MerchantRiskScorer
        from src.evaluation.decision_policy import DecisionPolicy
        from src.risk.explainability import FraudExplainer

        # Prevent unused-import warnings in static analysis.
        _ = MerchantRiskScorer
        _ = DecisionPolicy
        _ = FraudExplainer

        print("  ✓ Project imports passed")
        return True

    except Exception as exc:
        print(f"  ✗ Import failed: {exc}")
        traceback.print_exc()
        return False


def test_scorer_functionality() -> bool:
    """Check that the frozen merchant risk scorer initializes."""
    print("\nTesting scorer functionality...")

    try:
        from src.inference.scorer import MerchantRiskScorer

        model_path = ROOT_DIR / "models" / "logistic_regression_day6.joblib"

        if not model_path.exists():
            print(f"  ✗ Frozen model not found: {model_path}")
            return False

        scorer = MerchantRiskScorer(model_path=model_path)

        metrics = scorer.get_metrics()

        assert metrics["model_type"] == (
            "Logistic Regression (Frozen Day 6)"
        )

        assert metrics["input_features"] == 26

        assert metrics["policy_review_threshold"] == 0.35
        assert metrics["policy_hold_threshold"] == 0.70

        assert metrics["last_scored_count"] == 0

        print("  ✓ Scorer initialization passed")
        print("  ✓ Frozen model verified")
        print("  ✓ 26 input features verified")
        print("  ✓ Decision thresholds verified")

        return True

    except Exception as exc:
        print(f"  ✗ Scorer test failed: {exc}")
        traceback.print_exc()
        return False


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    """Run all Day 8 smoke tests."""

    print("=" * 70)
    print("AEGISRISK AI - DAY 8")
    print("STREAMLIT APPLICATION SMOKE TEST")
    print("=" * 70)

    syntax_ok = test_streamlit_syntax()
    imports_ok = test_imports()
    scorer_ok = test_scorer_functionality()

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"{'✓ PASS' if syntax_ok else '✗ FAIL'}: "
        "Streamlit Syntax Check"
    )

    print(
        f"{'✓ PASS' if imports_ok else '✗ FAIL'}: "
        "Module Imports"
    )

    print(
        f"{'✓ PASS' if scorer_ok else '✗ FAIL'}: "
        "Scorer Functionality"
    )

    print("=" * 70)

    if syntax_ok and imports_ok and scorer_ok:
        print("\n✓ DAY 8 SMOKE TEST PASSED")
        return 0

    print("\n✗ SMOKE TEST FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())