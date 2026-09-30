"""
features.py — Step 5: Feature Engineering, documented with formula /
purpose / prediction-time availability / leakage risk for every feature.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def add_hour_of_day(df: pd.DataFrame, time_col: str | None = None) -> pd.DataFrame:
    time_col = time_col or config.FEATURE_CONFIG["time"]
    df = df.copy()
    if time_col not in df.columns:
        logger.warning(f"Column '{time_col}' not found — skipping hour_of_day.")
        return df
    if pd.api.types.is_numeric_dtype(df[time_col]):
        # PaySim's "step" = 1 simulated hour. Take mod 24 to get hour-of-day.
        df["hour_of_day"] = df[time_col] % 24
        return df
    parsed = pd.to_datetime(df[time_col], errors="coerce")
    if parsed.notna().any():
        df["hour_of_day"] = parsed.dt.hour
    else:
        logger.warning(f"Column '{time_col}' could not be parsed — hour_of_day skipped.")
    return df


def add_day_of_week(df: pd.DataFrame, time_col: str | None = None) -> pd.DataFrame:
    time_col = time_col or config.FEATURE_CONFIG["time"]
    df = df.copy()
    if time_col not in df.columns:
        logger.warning(f"Column '{time_col}' not found — skipping day_of_week.")
        return df
    if pd.api.types.is_numeric_dtype(df[time_col]):
        df["day_of_week"] = (df[time_col] // 24) % 7
        return df
    parsed = pd.to_datetime(df[time_col], errors="coerce")
    if parsed.notna().any():
        df["day_of_week"] = parsed.dt.dayofweek
    return df


def add_amount_log_transform(df: pd.DataFrame, amount_col: str | None = None) -> pd.DataFrame:
    import numpy as np

    amount_col = amount_col or config.FEATURE_CONFIG["amount"]
    df = df.copy()
    if amount_col not in df.columns:
        logger.warning(f"Column '{amount_col}' not found — skipping amount_log.")
        return df
    safe_amount = df[amount_col].clip(lower=0)
    df["amount_log"] = np.log1p(safe_amount)
    return df


def add_amount_deviation_from_account_history(
    df: pd.DataFrame,
    amount_col: str | None = None,
    history_col: str | None = None,
) -> pd.DataFrame:
    amount_col = amount_col or config.FEATURE_CONFIG["amount"]
    history_col = history_col or config.FEATURE_CONFIG["account_history"]
    df = df.copy()
    if amount_col not in df.columns or history_col not in df.columns:
        logger.warning(
            f"Columns '{amount_col}' and/or '{history_col}' not found — skipping "
            "amount_deviation_from_history."
        )
        return df
    df["amount_deviation_from_history"] = (
        df[amount_col] - df[history_col]
    ) / (df[history_col].abs() + 1e-6)
    return df


def add_frequency_bucket(df: pd.DataFrame, frequency_col: str | None = None) -> pd.DataFrame:
    frequency_col = frequency_col or config.FEATURE_CONFIG["frequency"]
    df = df.copy()
    if frequency_col not in df.columns:
        logger.warning(f"Column '{frequency_col}' not found — skipping frequency_bucket.")
        return df
    bins = [-float("inf"), 3, 10, float("inf")]
    labels = ["low", "medium", "high"]
    df["frequency_bucket"] = pd.cut(df[frequency_col], bins=bins, labels=labels)
    return df


def add_transaction_count_per_account(
    df: pd.DataFrame,
    account_col: str = "nameOrig",
    time_col: str | None = None,
) -> pd.DataFrame:
    """
    PaySim has no native "frequency" column — this reconstructs one: for
    each row, counts PRIOR transactions by the same account (nameOrig),
    strictly before the current one when sorted by time. Leakage-safe.
    """
    time_col = time_col or config.FEATURE_CONFIG["time"]
    df = df.copy()
    if account_col not in df.columns or time_col not in df.columns:
        logger.warning(
            f"Columns '{account_col}' and/or '{time_col}' not found — "
            "skipping transaction_count_orig."
        )
        return df
    df = df.sort_values(time_col)
    df["transaction_count_orig"] = df.groupby(account_col).cumcount()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_hour_of_day(df)
    df = add_day_of_week(df)
    df = add_amount_log_transform(df)
    df = add_amount_deviation_from_account_history(df)
    df = add_transaction_count_per_account(df)  # MUST run before add_frequency_bucket
    df = add_frequency_bucket(df)
    return df


ENGINEERED_NUMERICAL_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "amount_log",
    "amount_deviation_from_history",
]
ENGINEERED_CATEGORICAL_FEATURES = ["frequency_bucket"]