from pathlib import Path

import joblib
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_day6.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "transactions_features.csv"


def main():
    print("=" * 70)
    print("AEGISRISK AI - DAY 7")
    print("SHAP COMPATIBILITY TEST")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Load frozen Day 6 model pipeline
    # ---------------------------------------------------------------
    pipeline = joblib.load(MODEL_PATH)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    print(f"\nModel path: {MODEL_PATH}")
    print(f"Model type: {type(model)}")

    # ---------------------------------------------------------------
    # 2. Load feature data
    # ---------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    input_features = list(pipeline.feature_names_in_)

    missing_features = [
        feature for feature in input_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Required frozen model features are missing: {missing_features}"
        )

    # CRITICAL: preserve the exact frozen feature order
    X = df.loc[:, input_features]

    print(f"\nInput feature count: {X.shape[1]}")
    print("Feature order verification:")

    for index, feature in enumerate(X.columns, start=1):
        print(f"  {index:02d}. {feature}")

    if list(X.columns) != input_features:
        raise ValueError("Input feature order does not match frozen model order.")

    print("\nExact feature order: PASSED")

    # ---------------------------------------------------------------
    # 3. Transform using the frozen Day 6 preprocessor
    # ---------------------------------------------------------------
    sample_size = min(100, len(X))
    X_sample = X.iloc[:sample_size].copy()

    X_transformed = preprocessor.transform(X_sample)
    transformed_feature_names = list(
        preprocessor.get_feature_names_out()
    )

    print(f"\nSample rows used: {sample_size}")
    print(f"Transformed feature count: {X_transformed.shape[1]}")
    print(
        f"Transformed feature-name count: "
        f"{len(transformed_feature_names)}"
    )

    if X_transformed.shape[1] != len(transformed_feature_names):
        raise ValueError(
            "Transformed feature matrix does not match feature names."
        )

    if X_transformed.shape[1] != model.n_features_in_:
        raise ValueError(
            "Transformed feature count does not match "
            "Logistic Regression input count."
        )

    print("Frozen preprocessing alignment: PASSED")

    # ---------------------------------------------------------------
    # 4. Build SHAP LinearExplainer on the actual trained model
    # ---------------------------------------------------------------
    print("\nCreating SHAP LinearExplainer...")

    explainer = shap.LinearExplainer(
        model,
        X_transformed
    )

    shap_values = explainer.shap_values(X_transformed[:5])

    print("SHAP explanation generation: PASSED")
    print(f"SHAP values type: {type(shap_values)}")
    print(f"SHAP values shape: {getattr(shap_values, 'shape', None)}")

    # ---------------------------------------------------------------
    # 5. Verify SHAP output alignment
    # ---------------------------------------------------------------
    if len(shap_values.shape) != 2:
        raise ValueError(
            "Expected 2-dimensional SHAP values for binary Logistic Regression."
        )

    if shap_values.shape[1] != len(transformed_feature_names):
        raise ValueError(
            "SHAP feature count does not match transformed feature names."
        )

    print("SHAP feature alignment: PASSED")

    # ---------------------------------------------------------------
    # 6. Show one sample's top contributions
    # ---------------------------------------------------------------
    sample_shap = shap_values[0]

    contributions = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "shap_value": sample_shap,
            "absolute_magnitude": abs(sample_shap),
        }
    ).sort_values(
        "absolute_magnitude",
        ascending=False,
    )

    print("\nTop 10 contributions for sample transaction:")
    print(
        contributions[
            ["feature", "shap_value", "absolute_magnitude"]
        ].head(10).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("DAY 7 SHAP COMPATIBILITY: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()