"""
Debug script to identify all_actions_valid failure
"""
import pandas as pd
from src.inference.scorer import MerchantRiskScorer

data = pd.read_csv('data/processed/transactions_features.csv', nrows=2000)
scorer = MerchantRiskScorer()
scored = scorer.score_transactions(data, include_explanation=True, top_k=5)

print('Probability stats:')
print(f'  Min: {scored["fraud_probability"].min():.4f}')
print(f'  Max: {scored["fraud_probability"].max():.4f}')
print(f'  Mean: {scored["fraud_probability"].mean():.4f}')
print(f'  All in [0,1]: {((scored["fraud_probability"] >= 0) & (scored["fraud_probability"] <= 1)).all()}')

print('\nAction distribution:')
print(scored['risk_action'].value_counts())

invalid_actions = scored[~scored['risk_action'].isin(['ALLOW', 'REVIEW', 'HOLD_FOR_VERIFICATION'])]
if not invalid_actions.empty:
    print(f'\nInvalid actions found ({len(invalid_actions)}):')
    print(invalid_actions[['transaction_id', 'fraud_probability', 'risk_action']].head(10))
else:
    print('\nNo invalid actions found')

print('\nDecision distribution:')
print(scored['risk_decision'].value_counts())
