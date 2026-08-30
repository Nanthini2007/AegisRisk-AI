"""
AegisRisk AI - Day 8
Streamlit Application Smoke Test

Validates that the Streamlit app can be imported and initialized
without errors.
"""

import sys
from pathlib import Path
import subprocess

def run_streamlit_syntax_check():
    """Check Streamlit app syntax by running python -m py_compile."""
    app_path = Path("app/streamlit_app.py")
    
    print(f"Checking Streamlit app syntax: {app_path}")
    
    if not app_path.exists():
        print(f"✗ App not found: {app_path}")
        return False
    
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(app_path)],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"✗ Syntax error:")
        print(result.stderr)
        return False
    
    print(f"✓ Syntax check passed")
    return True


def test_imports():
    """Test that app imports work."""
    print("\nTesting imports...")
    
    try:
        from src.inference.scorer import MerchantRiskScorer
        print("  ✓ MerchantRiskScorer imported")
        
        from src.risk.explainability import FraudExplainer
        print("  ✓ FraudExplainer imported")
        
        from src.evaluation.decision_policy import DecisionPolicy, decide_many
        print("  ✓ Decision policy imported")
        
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_scorer_functionality():
    """Test basic scorer functionality."""
    print("\nTesting scorer functionality...")
    
    try:
        import pandas as pd
        from src.inference.scorer import MerchantRiskScorer
        
        # Load minimal data
        data = pd.read_csv("data/processed/transactions_features.csv", nrows=5)
        print(f"  ✓ Loaded {len(data)} test transactions")
        
        # Initialize scorer
        scorer = MerchantRiskScorer()
        print(f"  ✓ Scorer initialized with {len(scorer.input_features)} features")
        
        # Score a transaction
        scored = scorer.score_transactions(data.head(1), include_explanation=False)
        print(f"  ✓ Transaction scored successfully")
        
        # Check output
        required_cols = ["fraud_probability", "risk_decision", "risk_action"]
        missing = [c for c in required_cols if c not in scored.columns]
        
        if missing:
            print(f"  ✗ Missing columns: {missing}")
            return False
        
        print(f"  ✓ All required output columns present")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Scorer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("AEGISRISK AI - DAY 8")
    print("STREAMLIT APPLICATION SMOKE TEST")
    print("=" * 70 + "\n")
    
    results = []
    
    # Test 1: Syntax check
    results.append(("Streamlit Syntax Check", run_streamlit_syntax_check()))
    
    # Test 2: Imports
    results.append(("Module Imports", test_imports()))
    
    # Test 3: Scorer Functionality
    results.append(("Scorer Functionality", test_scorer_functionality()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n✓ SMOKE TEST PASSED - App is ready to run")
        print("\nTo start the app, run:")
        print("  streamlit run app/streamlit_app.py")
        return 0
    else:
        print("\n✗ SMOKE TEST FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
