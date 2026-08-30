import json
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/logistic_regression_day6.joblib")
DATA_PATH = Path("data/processed/transactions_features.csv")

FEATURE_COLUMNS = [
    "amount",
    "payment_method",
    "account_age_days",
    "failed_attempts",
    "amount_log",
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "customer_txn_count_before",
    "customer_amount_mean_before",
    "customer_amount_std_before",
    "amount_vs_customer_mean",
    "customer_amount_zscore",
    "customer_seconds_since_prev",
    "customer_has_history",
    "customer_txn_count_10m",
    "customer_txn_count_1h",
    "customer_txn_count_24h",
    "merchant_txn_count_before",
    "merchant_amount_mean_before",
    "merchant_seconds_since_prev",
    "merchant_txn_count_1h",
    "customer_device_seen_before",
    "device_seen_before",
]

TRANSACTION_ID = "TXN_00107369"


def main():
    print("=" * 70)
    print("AEGISRISK AI - DAY 7")
    print("EXPLANATION CONTRIBUTION INSPECTION")
    print("=" * 70)

    print(f"\nLoading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    row = df.loc[df["transaction_id"] == TRANSACTION_ID]

    if row.empty:
        raise ValueError(
            f"Transaction {TRANSACTION_ID} was not found in the dataset."
        )

    X = row[FEATURE_COLUMNS]

    print(f"\nTransaction ID: {TRANSACTION_ID}")
    print(f"Model type: {type(model)}")
    print(f"Input feature count: {len(FEATURE_COLUMNS)}")

    # Verify the exact feature order stored by the frozen pipeline.
    stored_features = list(model.feature_names_in_)

    print("\n--- FEATURE ORDER CHECK ---")
    print(f"Expected count: {len(FEATURE_COLUMNS)}")
    print(f"Model count:    {len(stored_features)}")

    if stored_features != FEATURE_COLUMNS:
        print("FEATURE ORDER: FAILED")
        print("\nExpected:")
        for i, feature in enumerate(FEATURE_COLUMNS, 1):
            print(f"  {i:02d}. {feature}")

        print("\nModel:")
        for i, feature in enumerate(stored_features, 1):
            print(f"  {i:02d}. {feature}")

        raise ValueError("Frozen model feature order does not match.")

    print("FEATURE ORDER: PASSED")

    # Transform using the actual frozen preprocessing pipeline.
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    X_transformed = preprocessor.transform(X)

    transformed_names = list(
        preprocessor.get_feature_names_out()
    )

    print("\n--- TRANSFORMED REPRESENTATION ---")
    print(f"Transformed feature count: {X_transformed.shape[1]}")
    print(f"Named transformed features: {len(transformed_names)}")

    # Logistic regression decision contributions.
    coefficients = classifier.coef_[0]

    if len(coefficients) != X_transformed.shape[1]:
        raise ValueError(
            "Coefficient count does not match transformed feature count."
        )

    transformed_values = X_transformed.toarray()[0] \
        if hasattr(X_transformed, "toarray") \
        else X_transformed[0]

    contributions = coefficients * transformed_values

    print("\n--- NON-ZERO CONTRIBUTIONS ---")

    records = []

    for name, value, coefficient, contribution in zip(
        transformed_names,
        transformed_values,
        coefficients,
        contributions,
    ):
        records.append(
            {
                "feature": name,
                "transformed_value": float(value),
                "coefficient": float(coefficient),
                "contribution": float(contribution),
                "magnitude": abs(float(contribution)),
            }
        )

    records.sort(
        key=lambda item: item["magnitude"],
        reverse=True,
    )

    non_zero = [
        record
        for record in records
        if record["magnitude"] > 1e-12
    ]

    if not non_zero:
        print("WARNING: No non-zero contributions found.")
    else:
        for index, record in enumerate(non_zero[:10], 1):
            print(
                f"{index:02d}. "
                f"{record['feature']:<50} "
                f"value={record['transformed_value']:.6f} "
                f"coef={record['coefficient']:.6f} "
                f"contribution={record['contribution']:.6f}"
            )

    print("\n--- TOP 10 BY MAGNITUDE ---")

    for index, record in enumerate(records[:10], 1):
        print(
            f"{index:02d}. "
            f"{record['feature']:<50} "
            f"contribution={record['contribution']:.10f} "
            f"magnitude={record['magnitude']:.10f}"
        )

    # Compare model probability with the linear decision function.
    probability = float(model.predict_proba(X)[0, 1])
    decision_score = float(model.decision_function(X)[0])

    print("\n--- MODEL OUTPUT ---")
    print(f"Fraud probability: {probability:.12f}")
    print(f"Decision function: {decision_score:.12f}")
    print(
        f"Sum of local linear contributions: "
        f"{sum(contributions):.12f}"
    )

    print("\n--- INTERPRETATION ---")
    print(
        "For Logistic Regression, local feature contribution is "
        "coefficient × transformed feature value."
    )
    print(
        "A contribution near zero can be genuine, but the "
        "full ranking must be based on the actual values above."
    )

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()