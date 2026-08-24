"""
Tests for the AegisRisk AI Day 2 simulator.
"""

from pathlib import Path

import numpy as np

from src.simulation.generate_dataset import (
    create_devices,
    create_merchants,
    create_population,
    generate_transactions,
    inject_scenarios,
    load_config,
    validate_dataset,
)


CONFIG_PATH = Path("configs/simulation_config.yaml")


def create_test_config() -> dict:
    """Return a small configuration for fast testing."""

    return {
        "simulation": {
            "seed": {
                "value": 42
            }
        },
        "population": {
            "customers": {
                "value": 20
            },
            "merchants": {
                "value": 5
            },
            "devices_per_customer": {
                "min": {
                    "value": 1
                },
                "max": {
                    "value": 3
                },
            },
        },
        "temporal_period": {
            "start_date": {
                "value": "2026-01-01"
            },
            "duration_days": {
                "value": 3
            },
        },
        "transaction_behaviour": {
            "transaction_volume": {
                "expected_transactions_per_day": {
                    "value": 50
                }
            },
            "transaction_amount": {
                "min_amount": {
                    "value": 10.0
                },
                "max_amount": {
                    "value": 100000.0
                },
            },
        },
        "fraud_behaviour": {
            "base_probability": {
                "value": 0.02
            }
        },
        "scenario_configuration": {
            "scenarios": {
                "velocity_burst": {
                    "enabled": True,
                    "probability": 0.003,
                },
                "amount_anomaly": {
                    "enabled": True,
                    "probability": 0.004,
                },
                "new_device": {
                    "enabled": True,
                    "probability": 0.005,
                },
                "coordinated_attack": {
                    "enabled": True,
                    "probability": 0.002,
                },
                "behaviour_shift": {
                    "enabled": True,
                    "probability": 0.003,
                },
            }
        },
    }


def generate_test_dataset(seed: int = 42):
    """Generate a small deterministic test dataset."""

    config = create_test_config()

    rng = np.random.default_rng(seed)

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

    transactions = generate_transactions(
        config,
        customers,
        merchants,
        devices,
        rng,
    )

    transactions = inject_scenarios(
        transactions,
        config,
        rng,
    )

    transactions = (
        transactions
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return transactions


def test_dataset_generation():
    """Dataset generation should succeed."""

    df = generate_test_dataset()

    assert not df.empty


def test_required_columns():
    """All required transaction columns must exist."""

    df = generate_test_dataset()

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

    assert required_columns.issubset(df.columns)


def test_transaction_ids_unique():
    """Transaction IDs must be unique."""

    df = generate_test_dataset()

    assert df["transaction_id"].is_unique


def test_amounts_positive():
    """All transaction amounts must be positive."""

    df = generate_test_dataset()

    assert (df["amount"] > 0).all()


def test_fraud_labels_valid():
    """Fraud labels must contain only 0 and 1."""

    df = generate_test_dataset()

    assert set(
        df["fraud_label"].unique()
    ).issubset({0, 1})


def test_timestamps_chronological():
    """Transactions must be ordered chronologically."""

    df = generate_test_dataset()

    assert df["timestamp"].is_monotonic_increasing


def test_reproducibility():
    """Same seed must generate identical datasets."""

    df_one = generate_test_dataset(
        seed=123
    )

    df_two = generate_test_dataset(
        seed=123
    )

    assert df_one.equals(df_two)