"""
AegisRisk AI
Day 2 - Reproducible Temporal Behavioural Simulator

Generates synthetic transaction events for fraud-risk experiments.

IMPORTANT:
The generated records are synthetic and must not be represented
as real Razorpay production data.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


CONFIG_PATH = Path("configs/simulation_config.yaml")


def load_config(path: Path) -> dict:
    """Load and validate the YAML simulation configuration."""

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Simulation configuration must contain a YAML mapping."
        )

    return config


def create_population(
    config: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Create synthetic customers with persistent behavioural baselines.

    These are latent behavioural properties, not final ML features.
    """

    population = config["population"]

    n_customers = int(
        population["customers"]["value"]
    )

    min_devices = int(
        population["devices_per_customer"]["min"]["value"]
    )

    max_devices = int(
        population["devices_per_customer"]["max"]["value"]
    )

    customer_ids = [
        f"CUST_{i:06d}"
        for i in range(1, n_customers + 1)
    ]

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,

            "account_age_days": rng.integers(
                low=1,
                high=731,
                size=n_customers,
            ),

            "baseline_amount": np.clip(
                rng.lognormal(
                    mean=np.log(1200),
                    sigma=0.65,
                    size=n_customers,
                ),
                100,
                20000,
            ),

            "daily_transaction_rate": rng.gamma(
                shape=2.0,
                scale=1.0,
                size=n_customers,
            ),

            "preferred_hour": rng.integers(
                low=8,
                high=23,
                size=n_customers,
            ),

            "device_count": rng.integers(
                low=min_devices,
                high=max_devices + 1,
                size=n_customers,
            ),
        }
    )

    return customers


def create_merchants(
    config: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create synthetic merchant entities."""

    n_merchants = int(
        config["population"]["merchants"]["value"]
    )

    merchant_ids = [
        f"MERCHANT_{i:04d}"
        for i in range(1, n_merchants + 1)
    ]

    merchant_segments = rng.choice(
        ["small", "medium", "large"],
        size=n_merchants,
        p=[0.60, 0.30, 0.10],
    )

    return pd.DataFrame(
        {
            "merchant_id": merchant_ids,
            "merchant_segment": merchant_segments,
        }
    )


def create_devices(
    customers: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Create persistent synthetic devices for each customer.

    Returns:
        Mapping from customer_id to known device IDs.
    """

    device_map: dict[str, list[str]] = {}

    for row in customers.itertuples(index=False):

        devices = [
            f"DEV_{row.customer_id}_{device_no}"
            for device_no in range(
                1,
                row.device_count + 1,
            )
        ]

        device_map[row.customer_id] = devices

    return device_map


def generate_transactions(
    config: dict,
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    devices: dict[str, list[str]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate timestamped synthetic transaction events.

    Only raw transaction information is generated here.

    Historical behavioural ML features such as velocity or
    amount-vs-baseline are intentionally NOT generated here.
    """

    temporal = config["temporal_period"]
    transaction_config = config["transaction_behaviour"]

    start_date = datetime.fromisoformat(
        temporal["start_date"]["value"]
    )

    duration_days = int(
        temporal["duration_days"]["value"]
    )

    transactions_per_day = int(
        transaction_config[
            "transaction_volume"
        ]["expected_transactions_per_day"]["value"]
    )

    total_transactions = (
        transactions_per_day * duration_days
    )

    customer_indices = rng.integers(
        0,
        len(customers),
        size=total_transactions,
    )

    merchant_indices = rng.integers(
        0,
        len(merchants),
        size=total_transactions,
    )

    customer_sample = (
        customers.iloc[customer_indices]
        .reset_index(drop=True)
    )

    merchant_sample = (
        merchants.iloc[merchant_indices]
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Generate timestamps using customer preferred activity time
    # ---------------------------------------------------------

    day_offsets = rng.integers(
        0,
        duration_days,
        size=total_transactions,
    )

    preferred_hours = (
        customer_sample["preferred_hour"]
        .to_numpy()
    )

    hour_noise = rng.integers(
        -2,
        3,
        size=total_transactions,
    )

    hours = np.clip(
        preferred_hours + hour_noise,
        0,
        23,
    )

    minutes = rng.integers(
        0,
        60,
        size=total_transactions,
    )

    seconds = rng.integers(
        0,
        60,
        size=total_transactions,
    )

    timestamps = [
        start_date
        + timedelta(
            days=int(day),
            hours=int(hour),
            minutes=int(minute),
            seconds=int(second),
        )
        for day, hour, minute, second in zip(
            day_offsets,
            hours,
            minutes,
            seconds,
        )
    ]

    # ---------------------------------------------------------
    # Transaction amounts
    # ---------------------------------------------------------

    baseline_amount = (
        customer_sample["baseline_amount"]
        .to_numpy()
    )

    min_amount = float(
        transaction_config[
            "transaction_amount"
        ]["min_amount"]["value"]
    )

    max_amount = float(
        transaction_config[
            "transaction_amount"
        ]["max_amount"]["value"]
    )

    amounts = np.clip(
        baseline_amount
        * rng.lognormal(
            mean=0.0,
            sigma=0.45,
            size=total_transactions,
        ),
        min_amount,
        max_amount,
    )

    # ---------------------------------------------------------
    # Payment methods
    # ---------------------------------------------------------

    payment_methods = rng.choice(
        [
            "card",
            "upi",
            "netbanking",
            "wallet",
        ],
        size=total_transactions,
        p=[
            0.35,
            0.45,
            0.10,
            0.10,
        ],
    )

    # ---------------------------------------------------------
    # Persistent customer devices
    # ---------------------------------------------------------

    customer_ids = (
        customer_sample["customer_id"]
        .to_numpy()
    )

    device_ids = [
        rng.choice(
            devices[customer_id]
        )
        for customer_id in customer_ids
    ]

    # ---------------------------------------------------------
    # Failed attempts
    # ---------------------------------------------------------

    failed_attempts = rng.poisson(
        lam=0.08,
        size=total_transactions,
    )

    transactions = pd.DataFrame(
        {
            "transaction_id": [
                f"TXN_{i:08d}"
                for i in range(
                    1,
                    total_transactions + 1,
                )
            ],

            "timestamp": pd.to_datetime(
                timestamps
            ),

            "customer_id": customer_ids,

            "merchant_id": (
                merchant_sample[
                    "merchant_id"
                ].to_numpy()
            ),

            "device_id": device_ids,

            "amount": np.round(
                amounts,
                2,
            ),

            "payment_method": payment_methods,

            "account_age_days": (
                customer_sample[
                    "account_age_days"
                ].to_numpy()
            ),

            "failed_attempts": failed_attempts,

            "is_new_device": False,

            "scenario_id": "NORMAL",

            "fraud_label": 0,
        }
    )

    transactions = (
        transactions
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return transactions


def inject_scenarios(
    transactions: pd.DataFrame,
    config: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Inject controlled synthetic fraud-risk scenarios.

    Scenario IDs are simulation metadata only.
    They must NOT be used as ML features later.

    Important simulation rule:
    Suspicious patterns increase fraud probability, but no scenario
    should automatically determine fraud with certainty.
    """

    fraud_config = config["fraud_behaviour"]

    base_probability = float(
        fraud_config[
            "base_probability"
        ]["value"]
    )

    # ---------------------------------------------------------
    # Base probabilistic fraud
    # ---------------------------------------------------------

    base_fraud = (
        rng.random(len(transactions))
        < base_probability
    )

    transactions.loc[
        base_fraud,
        "fraud_label",
    ] = 1

    # ---------------------------------------------------------
    # Scenario configuration
    # ---------------------------------------------------------

    scenarios = config[
        "scenario_configuration"
    ]["scenarios"]

    for scenario_name, settings in scenarios.items():

        if not settings.get("enabled", False):
            continue

        probability = float(
            settings["probability"]
        )

        mask = (
            rng.random(len(transactions))
            < probability
        )

        # -----------------------------------------------------
        # Velocity Burst
        # -----------------------------------------------------

        if scenario_name == "velocity_burst":

            transactions.loc[
                mask,
                "scenario_id",
            ] = "VELOCITY_BURST"

            transactions.loc[
                mask,
                "failed_attempts",
            ] += rng.poisson(
                lam=1.5,
                size=mask.sum(),
            )

            transactions.loc[
                mask,
                "fraud_label",
            ] = (
                rng.random(mask.sum()) < 0.80
            ).astype(int)

        # -----------------------------------------------------
        # Amount Anomaly
        # -----------------------------------------------------

        elif scenario_name == "amount_anomaly":

            transactions.loc[
                mask,
                "scenario_id",
            ] = "AMOUNT_ANOMALY"

            multiplier = rng.uniform(
                3.0,
                10.0,
                size=mask.sum(),
            )

            transactions.loc[
                mask,
                "amount",
            ] = np.round(
                transactions.loc[
                    mask,
                    "amount",
                ].to_numpy()
                * multiplier,
                2,
            )

            transactions.loc[
                mask,
                "fraud_label",
            ] = (
                rng.random(mask.sum()) < 0.70
            ).astype(int)

        # -----------------------------------------------------
        # New Device
        # -----------------------------------------------------

        elif scenario_name == "new_device":

            transactions.loc[
                mask,
                "scenario_id",
            ] = "NEW_DEVICE"

            transactions.loc[
                mask,
                "is_new_device",
            ] = True

            transactions.loc[
                mask,
                "fraud_label",
            ] = (
                rng.random(mask.sum()) < 0.60
            ).astype(int)

        # -----------------------------------------------------
        # Coordinated Attack
        # -----------------------------------------------------

        elif scenario_name == "coordinated_attack":

            transactions.loc[
                mask,
                "scenario_id",
            ] = "COORDINATED_ATTACK"

            transactions.loc[
                mask,
                "failed_attempts",
            ] += rng.poisson(
                lam=2.0,
                size=mask.sum(),
            )

            # High-risk scenario, but NOT deterministically fraud.
            # 0.85 is a synthetic simulator assumption, not a
            # real-world fraud statistic.
            transactions.loc[
                mask,
                "fraud_label",
            ] = (
                rng.random(mask.sum()) < 0.85
            ).astype(int)

        # -----------------------------------------------------
        # Behaviour Shift
        # -----------------------------------------------------

        elif scenario_name == "behaviour_shift":

            transactions.loc[
                mask,
                "scenario_id",
            ] = "BEHAVIOUR_SHIFT"

            multiplier = rng.uniform(
                2.0,
                6.0,
                size=mask.sum(),
            )

            transactions.loc[
                mask,
                "amount",
            ] = np.round(
                transactions.loc[
                    mask,
                    "amount",
                ].to_numpy()
                * multiplier,
                2,
            )

            transactions.loc[
                mask,
                "fraud_label",
            ] = (
                rng.random(mask.sum()) < 0.50
            ).astype(int)

    # ---------------------------------------------------------
    # Final amount rounding
    # ---------------------------------------------------------

    transactions["amount"] = (
        transactions["amount"]
        .round(2)
    )

    # ---------------------------------------------------------
    # Ensure fraud_label remains integer
    # ---------------------------------------------------------

    transactions["fraud_label"] = (
        transactions["fraud_label"]
        .astype(int)
    )

    return transactions


def validate_dataset(
    df: pd.DataFrame,
) -> None:
    """Run basic integrity checks on the generated dataset."""

    required_columns = {
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

    # ---------------------------------------------------------
    # Dataset must not be empty
    # ---------------------------------------------------------

    if df.empty:
        raise ValueError(
            "Generated dataset is empty."
        )

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    # ---------------------------------------------------------
    # Unique transaction IDs
    # ---------------------------------------------------------

    if df["transaction_id"].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs detected."
        )

    # ---------------------------------------------------------
    # Timestamp validation
    # ---------------------------------------------------------

    if df["timestamp"].isna().any():
        raise ValueError(
            "Missing timestamps detected."
        )

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Dataset timestamps are not chronological."
        )

    # ---------------------------------------------------------
    # Amount validation
    # ---------------------------------------------------------

    if (df["amount"] <= 0).any():
        raise ValueError(
            "Non-positive transaction amount detected."
        )

    # ---------------------------------------------------------
    # Fraud label validation
    # ---------------------------------------------------------

    if not df["fraud_label"].isin([0, 1]).all():
        raise ValueError(
            "fraud_label must contain only 0 or 1."
        )

    print("Validation: PASSED")


def main() -> None:
    """Generate, validate and save the synthetic dataset."""

    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------

    config = load_config(
        CONFIG_PATH
    )

    # ---------------------------------------------------------
    # 2. Reproducible random generator
    # ---------------------------------------------------------

    seed = int(
        config[
            "simulation"
        ]["seed"]["value"]
    )

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 3. Create synthetic entities
    # ---------------------------------------------------------

    customers = create_population(
        config,
        rng,
    )

    merchants = create_merchants(
        config,
        rng,
    )

    devices = create_devices(
        customers,
    )

    # ---------------------------------------------------------
    # 4. Generate raw transactions
    # ---------------------------------------------------------

    transactions = generate_transactions(
        config,
        customers,
        merchants,
        devices,
        rng,
    )

    # ---------------------------------------------------------
    # 5. Inject controlled scenarios
    # ---------------------------------------------------------

    transactions = inject_scenarios(
        transactions,
        config,
        rng,
    )

    # ---------------------------------------------------------
    # 6. Final chronological ordering
    # ---------------------------------------------------------

    transactions = (
        transactions
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 7. Validate
    # ---------------------------------------------------------

    validate_dataset(
        transactions
    )

    # ---------------------------------------------------------
    # 8. Output path
    # ---------------------------------------------------------

    output_path = Path(
        config[
            "output"
        ]["path"]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 9. Save CSV
    # ---------------------------------------------------------

    transactions.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 10. Final report
    # ---------------------------------------------------------

    print("=" * 60)
    print(
        "AegisRisk AI - Day 2 Simulator"
    )
    print("=" * 60)

    print(
        f"Seed: {seed}"
    )

    print(
        f"Customers: "
        f"{len(customers):,}"
    )

    print(
        f"Merchants: "
        f"{len(merchants):,}"
    )

    print(
        f"Transactions: "
        f"{len(transactions):,}"
    )

    print(
        f"Date range: "
        f"{transactions['timestamp'].min()} → "
        f"{transactions['timestamp'].max()}"
    )

    print(
        f"Fraud rate: "
        f"{transactions['fraud_label'].mean():.2%}"
    )

    print("\nScenario distribution:")

    print(
        transactions[
            "scenario_id"
        ]
        .value_counts()
        .to_string()
    )

    print(
        f"\nSaved to: "
        f"{output_path}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()