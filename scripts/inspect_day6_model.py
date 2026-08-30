from pathlib import Path
import joblib

MODEL_PATH = Path("models/logistic_regression_day6.joblib")

model = joblib.load(MODEL_PATH)

print("=" * 70)
print("AEGISRISK AI - DAY 7")
print("FROZEN DAY 6 MODEL INSPECTION")
print("=" * 70)

print(f"\nModel path: {MODEL_PATH}")
print(f"Model type: {type(model)}")

print("\n--- Model representation ---")
print(model)

print("\n--- Top-level attributes ---")
for attr in [
    "feature_names_in_",
    "n_features_in_",
    "named_steps",
    "coef_",
    "classes_",
]:
    if hasattr(model, attr):
        value = getattr(model, attr)

        if attr == "feature_names_in_":
            print(f"{attr}:")
            for index, feature in enumerate(value, start=1):
                print(f"  {index:02d}. {feature}")
        else:
            print(f"{attr}: {value}")

if hasattr(model, "named_steps"):
    print("\n--- Pipeline steps ---")

    for step_name, step in model.named_steps.items():
        print(f"\nStep: {step_name}")
        print(f"Type: {type(step)}")
        print(f"Object: {step}")

        if hasattr(step, "feature_names_in_"):
            print("Input features:")
            for index, feature in enumerate(
                step.feature_names_in_,
                start=1
            ):
                print(f"  {index:02d}. {feature}")

        if hasattr(step, "get_feature_names_out"):
            try:
                output_features = step.get_feature_names_out()

                print(
                    f"Output feature count: "
                    f"{len(output_features)}"
                )

                print("Output features:")
                for index, feature in enumerate(
                    output_features,
                    start=1
                ):
                    print(f"  {index:02d}. {feature}")

            except Exception as error:
                print(
                    "Could not retrieve output feature names: "
                    f"{error}"
                )

print("\n--- Inspection complete ---")