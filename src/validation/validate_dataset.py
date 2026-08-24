"""
AegisRisk AI
Day 3 - Dataset Validation and Exploratory Analysis

Purpose:
Validate the synthetic temporal transaction dataset before
feature engineering and machine learning.

The generated data is synthetic and must not be represented
as real Razorpay production data.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("data/raw/transactions.csv")
REPORT_PATH = Path("reports/day3_validation_report.json")
FIGURE_DIR = Path("figures")


REQUIRED_COLUMNS = {
    "transaction_id",
    "timestamp",
    "customer_id",
    "merchant_id",
    "device_id",
    "amount",
    "payment_method",
    "account_age_days",
    "failed_attempts",
    "is_new_device",
    "scenario_id",
    "fraud_label",
}


def load_data(path: Path) -> pd.DataFrame:
    """Load the synthetic transaction dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

    return df


def schema_checks(df: pd.DataFrame) -> dict:
    """Check required columns, unexpected columns and IDs."""

    actual_columns = set(df.columns)

    missing_columns = sorted(
        REQUIRED_COLUMNS - actual_columns
    )

    unexpected_columns = sorted(
        actual_columns - REQUIRED_COLUMNS
    )

    duplicate_ids = 0

    if "transaction_id" in df.columns:
        duplicate_ids = int(
            df["transaction_id"].duplicated().sum()
        )

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_required_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "duplicate_transaction_ids": duplicate_ids,
        "passed": (
            len(missing_columns) == 0
            and duplicate_ids == 0
        ),
    }


def data_quality_checks(df: pd.DataFrame) -> dict:
    """Check missing values and invalid values."""

    missing_values = {
        str(column): int(count)
        for column, count in df.isna().sum().items()
        if count > 0
    }

    duplicate_rows = int(
        df.duplicated().sum()
    )

    invalid_amounts = 0
    invalid_labels = 0
    invalid_account_age = 0
    invalid_failed_attempts = 0
    invalid_new_device = 0
    invalid_timestamps = 0

    if "amount" in df.columns:
        invalid_amounts = int(
            (df["amount"] <= 0).sum()
        )

    if "fraud_label" in df.columns:
        invalid_labels = int(
            (~df["fraud_label"].isin([0, 1])).sum()
        )

    if "account_age_days" in df.columns:
        invalid_account_age = int(
            (df["account_age_days"] < 0).sum()
        )

    if "failed_attempts" in df.columns:
        invalid_failed_attempts = int(
            (df["failed_attempts"] < 0).sum()
        )

    if "is_new_device" in df.columns:
        invalid_new_device = int(
            (~df["is_new_device"].isin([0, 1, True, False])).sum()
        )

    if "timestamp" in df.columns:
        invalid_timestamps = int(
            df["timestamp"].isna().sum()
        )

    passed = (
        len(missing_values) == 0
        and duplicate_rows == 0
        and invalid_amounts == 0
        and invalid_labels == 0
        and invalid_account_age == 0
        and invalid_failed_attempts == 0
        and invalid_new_device == 0
        and invalid_timestamps == 0
    )

    return {
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "invalid_amounts": invalid_amounts,
        "invalid_fraud_labels": invalid_labels,
        "invalid_account_age": invalid_account_age,
        "invalid_failed_attempts": invalid_failed_attempts,
        "invalid_is_new_device": invalid_new_device,
        "invalid_timestamps": invalid_timestamps,
        "passed": passed,
    }


def distribution_analysis(df: pd.DataFrame) -> dict:
    """Analyze transaction distributions."""

    amount_percentiles = df["amount"].quantile(
        [0.25, 0.50, 0.75, 0.95, 0.99]
    )

    payment_methods = (
        df["payment_method"]
        .value_counts()
        .to_dict()
    )

    payment_method_percentages = (
        df["payment_method"]
        .value_counts(normalize=True)
        .mul(100)
        .to_dict()
    )

    transactions_per_device = (
        df.groupby("device_id")
        .size()
    )

    failed_attempt_percentiles = (
        df["failed_attempts"]
        .quantile([0.50, 0.95, 0.99])
    )

    account_age_percentiles = (
        df["account_age_days"]
        .quantile([0.25, 0.50, 0.75, 0.95])
    )

    return {
        "amount": {
            "mean": float(df["amount"].mean()),
            "median": float(df["amount"].median()),
            "p25": float(amount_percentiles.loc[0.25]),
            "p50": float(amount_percentiles.loc[0.50]),
            "p75": float(amount_percentiles.loc[0.75]),
            "p95": float(amount_percentiles.loc[0.95]),
            "p99": float(amount_percentiles.loc[0.99]),
            "minimum": float(df["amount"].min()),
            "maximum": float(df["amount"].max()),
        },
        "payment_methods": {
            "counts": {
                str(key): int(value)
                for key, value in payment_methods.items()
            },
            "percentages": {
                str(key): float(value)
                for key, value in payment_method_percentages.items()
            },
        },
        "devices": {
            "unique_devices": int(df["device_id"].nunique()),
            "transactions_per_device_median": float(
                transactions_per_device.median()
            ),
            "transactions_per_device_p95": float(
                transactions_per_device.quantile(0.95)
            ),
            "transactions_per_device_max": int(
                transactions_per_device.max()
            ),
        },
        "failed_attempts": {
            "median": float(
                failed_attempt_percentiles.loc[0.50]
            ),
            "p95": float(
                failed_attempt_percentiles.loc[0.95]
            ),
            "p99": float(
                failed_attempt_percentiles.loc[0.99]
            ),
            "maximum": int(
                df["failed_attempts"].max()
            ),
        },
        "account_age_days": {
            "median": float(
                account_age_percentiles.loc[0.50]
            ),
            "p25": float(
                account_age_percentiles.loc[0.25]
            ),
            "p75": float(
                account_age_percentiles.loc[0.75]
            ),
            "p95": float(
                account_age_percentiles.loc[0.95]
            ),
            "maximum": int(
                df["account_age_days"].max()
            ),
        },
    }


def fraud_analysis(df: pd.DataFrame) -> dict:
    """Analyze fraud class distribution."""

    fraud_count = int(
        (df["fraud_label"] == 1).sum()
    )

    legitimate_count = int(
        (df["fraud_label"] == 0).sum()
    )

    return {
        "legitimate_count": legitimate_count,
        "fraud_count": fraud_count,
        "fraud_rate": float(
            df["fraud_label"].mean()
        ),
        "accuracy_warning": (
            "Accuracy should not be the primary metric for "
            "imbalanced fraud detection because a model can "
            "achieve high accuracy by mostly predicting the "
            "majority legitimate class while missing fraud."
        ),
    }


def scenario_analysis(df: pd.DataFrame) -> dict:
    """Analyze simulator scenarios."""

    scenario_summary = {}

    total_rows = len(df)

    for scenario, group in df.groupby("scenario_id"):
        percentiles = group["amount"].quantile(
            [0.50, 0.95]
        )

        scenario_summary[str(scenario)] = {
            "count": int(len(group)),
            "percentage": float(
                len(group) / total_rows * 100
            ),
            "fraud_rate": float(
                group["fraud_label"].mean()
            ),
            "amount_median": float(
                percentiles.loc[0.50]
            ),
            "amount_p95": float(
                percentiles.loc[0.95]
            ),
        }

    return {
        "scenario_statistics": scenario_summary,
        "feature_policy": (
            "scenario_id is simulation metadata and must "
            "not be used as an ML prediction feature."
        ),
    }


def customer_analysis(df: pd.DataFrame) -> dict:
    """Analyze persistent customer behaviour."""

    transactions_per_customer = (
        df.groupby("customer_id")
        .size()
    )

    median_amount_per_customer = (
        df.groupby("customer_id")["amount"]
        .median()
    )

    customer_share = (
        transactions_per_customer / len(df)
    )

    return {
        "unique_customers": int(
            df["customer_id"].nunique()
        ),
        "transactions_per_customer": {
            "mean": float(
                transactions_per_customer.mean()
            ),
            "median": float(
                transactions_per_customer.median()
            ),
            "p95": float(
                transactions_per_customer.quantile(0.95)
            ),
            "maximum": int(
                transactions_per_customer.max()
            ),
        },
        "median_amount_per_customer": {
            "overall_median": float(
                median_amount_per_customer.median()
            ),
            "p95": float(
                median_amount_per_customer.quantile(0.95)
            ),
        },
        "activity_concentration": {
            "top_10_customers_transaction_share": float(
                customer_share.nlargest(10).sum()
            ),
            "most_active_customer_share": float(
                customer_share.max()
            ),
        },
    }


def merchant_analysis(df: pd.DataFrame) -> dict:
    """Analyze merchant transaction activity."""

    transactions_per_merchant = (
        df.groupby("merchant_id")
        .size()
    )

    return {
        "unique_merchants": int(
            df["merchant_id"].nunique()
        ),
        "transactions_per_merchant": {
            "mean": float(
                transactions_per_merchant.mean()
            ),
            "median": float(
                transactions_per_merchant.median()
            ),
            "p95": float(
                transactions_per_merchant.quantile(0.95)
            ),
            "maximum": int(
                transactions_per_merchant.max()
            ),
        },
    }


def temporal_analysis(df: pd.DataFrame) -> dict:
    """Analyze temporal coverage and variation."""

    invalid_timestamps = int(
        df["timestamp"].isna().sum()
    )

    valid_df = df.dropna(
        subset=["timestamp"]
    ).copy()

    sorted_correctly = bool(
        valid_df["timestamp"].is_monotonic_increasing
    )

    if valid_df.empty:
        return {
            "invalid_timestamps": invalid_timestamps,
            "passed": False,
            "error": "No valid timestamps available.",
        }

    valid_df["date"] = (
        valid_df["timestamp"].dt.date
    )

    valid_df["hour"] = (
        valid_df["timestamp"].dt.hour
    )

    daily = (
        valid_df.groupby("date")
        .size()
    )

    daily_fraud_count = (
        valid_df.groupby("date")["fraud_label"]
        .sum()
    )

    daily_fraud_rate = (
        valid_df.groupby("date")["fraud_label"]
        .mean()
    )

    hourly_volume = (
        valid_df.groupby("hour")
        .size()
    )

    hourly_fraud_rate = (
        valid_df.groupby("hour")["fraud_label"]
        .mean()
    )

    date_index = pd.date_range(
        valid_df["timestamp"].min().normalize(),
        valid_df["timestamp"].max().normalize(),
        freq="D",
    )

    observed_dates = pd.to_datetime(
        pd.Series(daily.index)
    )

    gaps = sorted(
        set(date_index.date)
        - set(observed_dates.dt.date)
    )

    return {
        "invalid_timestamps": invalid_timestamps,
        "chronologically_sorted": sorted_correctly,
        "date_range": {
            "start": str(
                valid_df["timestamp"].min()
            ),
            "end": str(
                valid_df["timestamp"].max()
            ),
        },
        "number_of_days": int(
            valid_df["date"].nunique()
        ),
        "temporal_gaps": [
            str(gap) for gap in gaps
        ],
        "daily_volume": {
            "minimum": int(daily.min()),
            "median": float(daily.median()),
            "maximum": int(daily.max()),
        },
        "daily_fraud_count": {
            "minimum": int(daily_fraud_count.min()),
            "maximum": int(daily_fraud_count.max()),
        },
        "daily_fraud_rate": {
            "minimum": float(
                daily_fraud_rate.min()
            ),
            "median": float(
                daily_fraud_rate.median()
            ),
            "maximum": float(
                daily_fraud_rate.max()
            ),
        },
        "hourly_volume": {
            str(hour): int(count)
            for hour, count in hourly_volume.items()
        },
        "hourly_fraud_rate": {
            str(hour): float(rate)
            for hour, rate in hourly_fraud_rate.items()
        },
        "passed": (
            invalid_timestamps == 0
            and sorted_correctly
        ),
    }


def leakage_audit(df: pd.DataFrame) -> dict:
    """Perform formal Day 3 leakage review."""

    excluded_columns = {
        "fraud_label": (
            "Ground truth target. Using it as a feature "
            "would directly leak the answer."
        ),
        "scenario_id": (
            "Simulation metadata describing how synthetic "
            "data was generated. It is not an observable "
            "transaction feature for prediction."
        ),
    }

    future_information_rule = (
        "For a transaction at timestamp t, every ML feature "
        "must be computed using only information available "
        "at or before timestamp t. Historical aggregates "
        "must never use transactions occurring after t."
    )

    identifier_warning = {
        "customer_id": (
            "Do not use as a raw predictive shortcut. It may "
            "be used to build historical features using only "
            "past or current information."
        ),
        "merchant_id": (
            "Do not use future merchant outcomes when "
            "constructing historical features."
        ),
        "device_id": (
            "Do not use future device activity when "
            "constructing historical features."
        ),
    }

    return {
        "ground_truth": ["fraud_label"],
        "simulation_metadata": ["scenario_id"],
        "excluded_from_ml_features": excluded_columns,
        "identifier_temporal_warnings": identifier_warning,
        "future_information_rule": future_information_rule,
        "passed": True,
    }


def simulator_review(df: pd.DataFrame) -> dict:
    """Classify dataset findings as PASS, WARNING or REVISE."""

    pass_findings = []
    warning_findings = []
    revise_findings = []

    if len(df) > 0:
        pass_findings.append(
            "Dataset contains transactions and is non-empty."
        )

    if df["amount"].nunique() > 10:
        pass_findings.append(
            "Transaction amounts show non-trivial variation."
        )
    else:
        revise_findings.append(
            "Transaction amount diversity is too low."
        )

    scenario_percentages = (
        df["scenario_id"]
        .value_counts(normalize=True)
        * 100
    )

    if scenario_percentages.max() > 80:
        warning_findings.append(
            "One simulation scenario represents more than "
            "80% of transactions, indicating concentration."
        )

    scenario_fraud_rates = (
        df.groupby("scenario_id")["fraud_label"]
        .mean()
    )

    if (
        len(scenario_fraud_rates) > 1
        and (
            scenario_fraud_rates.max()
            - scenario_fraud_rates.min()
        ) > 0.95
    ):
        revise_findings.append(
            "Fraud rate differs by more than 95 percentage "
            "points across scenarios. Review whether "
            "simulation metadata makes fraud overly "
            "deterministic."
        )

    customer_counts = (
        df.groupby("customer_id")
        .size()
    )

    if customer_counts.median() < 2:
        warning_findings.append(
            "Median transactions per customer is below 2. "
            "Historical customer behaviour may be limited."
        )
    else:
        pass_findings.append(
            "Customers have repeated transaction history."
        )

    if (
        df["timestamp"].nunique()
        <= 1
    ):
        revise_findings.append(
            "Dataset has insufficient timestamp variation."
        )

    if (
        df["fraud_label"].nunique()
        < 2
    ):
        revise_findings.append(
            "Only one fraud class is present."
        )

    return {
        "PASS": pass_findings,
        "WARNING": warning_findings,
        "REVISE": revise_findings,
    }


def final_recommendation(
    schema: dict,
    quality: dict,
    temporal: dict,
    findings: dict,
) -> str:
    """Determine final Day 3 recommendation."""

    if (
        not schema["passed"]
        or not quality["passed"]
        or not temporal["passed"]
    ):
        return "FAIL"

    if len(findings["REVISE"]) > 0:
        return "REVISE"

    return "PASS"


def create_figures(df: pd.DataFrame) -> None:
    """Create required Day 3 visualizations."""

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Transaction amount distribution
    plt.figure(figsize=(9, 5))
    plt.hist(df["amount"], bins=60)
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")
    plt.title("Transaction Amount Distribution")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "amount_distribution.png",
        dpi=150,
    )
    plt.close()

    # 2. Fraud class distribution
    fraud_counts = (
        df["fraud_label"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(7, 5))
    plt.bar(
        fraud_counts.index.astype(str),
        fraud_counts.values,
    )
    plt.xlabel("Fraud Label")
    plt.ylabel("Transaction Count")
    plt.title("Fraud Class Distribution")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "fraud_distribution.png",
        dpi=150,
    )
    plt.close()

    working_df = df.dropna(
        subset=["timestamp"]
    ).copy()

    working_df["date"] = (
        working_df["timestamp"].dt.date
    )

    # 3. Daily transaction volume
    daily_volume = (
        working_df.groupby("date")
        .size()
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        daily_volume.index.astype(str),
        daily_volume.values,
    )
    plt.xlabel("Date")
    plt.ylabel("Transactions")
    plt.title("Daily Transaction Volume")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "daily_transactions.png",
        dpi=150,
    )
    plt.close()

    # 4. Daily fraud rate
    daily_fraud_rate = (
        working_df.groupby("date")["fraud_label"]
        .mean()
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        daily_fraud_rate.index.astype(str),
        daily_fraud_rate.values,
    )
    plt.xlabel("Date")
    plt.ylabel("Fraud Rate")
    plt.title("Daily Fraud Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "daily_fraud_rate.png",
        dpi=150,
    )
    plt.close()

    # 5. Scenario distribution
    scenario_counts = (
        df["scenario_id"]
        .value_counts()
    )

    plt.figure(figsize=(9, 5))
    plt.bar(
        scenario_counts.index.astype(str),
        scenario_counts.values,
    )
    plt.xlabel("Scenario")
    plt.ylabel("Transaction Count")
    plt.title("Simulation Scenario Distribution")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "scenario_distribution.png",
        dpi=150,
    )
    plt.close()

    # 6. Fraud rate by scenario
    scenario_fraud_rate = (
        df.groupby("scenario_id")["fraud_label"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(9, 5))
    plt.bar(
        scenario_fraud_rate.index.astype(str),
        scenario_fraud_rate.values,
    )
    plt.xlabel("Scenario")
    plt.ylabel("Fraud Rate")
    plt.title("Fraud Rate by Simulation Scenario")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "scenario_fraud_rate.png",
        dpi=150,
    )
    plt.close()


def build_report(df: pd.DataFrame) -> dict:
    """Build the complete Day 3 validation report."""

    schema = schema_checks(df)
    quality = data_quality_checks(df)
    distribution = distribution_analysis(df)
    fraud = fraud_analysis(df)
    scenarios = scenario_analysis(df)
    customers = customer_analysis(df)
    merchants = merchant_analysis(df)
    temporal = temporal_analysis(df)
    leakage = leakage_audit(df)
    findings = simulator_review(df)

    recommendation = final_recommendation(
        schema,
        quality,
        temporal,
        findings,
    )

    return {
        "project": "AegisRisk AI",
        "stage": "Day 3 - Dataset Validation and EDA",
        "synthetic_data_notice": (
            "This dataset is synthetic and must not be "
            "represented as real Razorpay production data."
        ),
        "dataset_size": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },
        "schema_results": schema,
        "data_quality": quality,
        "amount_statistics": distribution["amount"],
        "distribution_statistics": distribution,
        "fraud_statistics": fraud,
        "scenario_statistics": scenarios,
        "customer_statistics": customers,
        "merchant_statistics": merchants,
        "temporal_statistics": temporal,
        "leakage_audit": leakage,
        "warnings": findings["WARNING"],
        "findings": findings,
        "final_recommendation": recommendation,
    }


def main() -> None:
    """Run Day 3 validation."""

    df = load_data(DATA_PATH)

    report = build_report(df)

    create_figures(df)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("=" * 65)
    print("AegisRisk AI - Day 3 Dataset Validation and EDA")
    print("=" * 65)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(
        f"Customers: "
        f"{report['customer_statistics']['unique_customers']:,}"
    )
    print(
        f"Merchants: "
        f"{report['merchant_statistics']['unique_merchants']:,}"
    )
    print(
        f"Fraud rate: "
        f"{report['fraud_statistics']['fraud_rate']:.2%}"
    )

    date_range = (
        report["temporal_statistics"]
        .get("date_range", {})
    )

    if date_range:
        print(
            f"Date range: "
            f"{date_range.get('start')} -> "
            f"{date_range.get('end')}"
        )

    print("\nValidation Results:")
    print(
        "Schema: "
        + (
            "PASS"
            if report["schema_results"]["passed"]
            else "FAIL"
        )
    )
    print(
        "Data quality: "
        + (
            "PASS"
            if report["data_quality"]["passed"]
            else "FAIL"
        )
    )
    print(
        "Temporal: "
        + (
            "PASS"
            if report["temporal_statistics"]["passed"]
            else "FAIL"
        )
    )

    print("\nFindings:")

    for status in ["PASS", "WARNING", "REVISE"]:
        findings = report["findings"][status]

        if findings:
            print(f"\n{status}:")
            for finding in findings:
                print(f"  - {finding}")

    print(
        "\nFinal recommendation: "
        f"{report['final_recommendation']}"
    )

    print(
        f"\nReport saved to: {REPORT_PATH}"
    )
    print(
        f"Figures saved to: {FIGURE_DIR}"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()