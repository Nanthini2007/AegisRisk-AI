"""
Chronological dataset splitting utilities for Day 6 evaluation.

The held-out test period must remain untouched during model and threshold
selection. Splits are deterministic and preserve temporal ordering.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def chronological_train_validation_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split an already time-sorted dataset into chronological
    train, validation, and held-out test periods.

    The function does not shuffle rows.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if df.empty:
        raise ValueError("df must not be empty")

    ratios = (train_ratio, validation_ratio, test_ratio)

    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("All split ratios must be greater than 0")

    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(
            "train_ratio + validation_ratio + test_ratio must equal 1.0"
        )

    total_rows = len(df)
    train_end = int(total_rows * train_ratio)
    validation_end = train_end + int(total_rows * validation_ratio)

    if train_end == 0 or validation_end == train_end or validation_end >= total_rows:
        raise ValueError(
            "Dataset is too small to create non-empty train, validation, and test splits"
        )

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    return train_df, validation_df, test_df


def assert_temporal_separation(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> None:
    """
    Assert strict chronological separation between all three periods.
    """

    for name, split_df in {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }.items():
        if split_df.empty:
            raise AssertionError(f"{name} split must not be empty")

        if timestamp_column not in split_df.columns:
            raise AssertionError(
                f"Missing timestamp column '{timestamp_column}' in {name} split"
            )

    train_end = train_df[timestamp_column].max()
    validation_start = validation_df[timestamp_column].min()
    validation_end = validation_df[timestamp_column].max()
    test_start = test_df[timestamp_column].min()

    if train_end >= validation_start:
        raise AssertionError(
            "Temporal overlap detected between train and validation splits"
        )

    if validation_end >= test_start:
        raise AssertionError(
            "Temporal overlap detected between validation and held-out test splits"
        )