"""
data_loader.py
===============
Step 1 — Data Understanding.

Loads the raw dataset and provides reusable, honest inspection functions:
shape, dtypes, missing values, duplicates, unique-value counts, descriptive
statistics, target distribution/imbalance, and a first pass at flagging
columns that could leak target information.

Every function here returns real computed values from whatever dataset is
loaded — nothing is hard-coded or fabricated. If no dataset is present,
`load_raw_data()` raises `DatasetNotFoundError` with instructions rather than
inventing data.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.utils import DatasetNotFoundError, get_logger

logger = get_logger(__name__)


def load_raw_data(path=None) -> pd.DataFrame:
    """
    Load the raw transaction dataset from CSV.

    Parameters
    ----------
    path : Path, optional
        Overrides config.RAW_DATA_PATH if provided.

    Raises
    ------
    DatasetNotFoundError
        If the CSV file does not exist yet.
    """
    data_path = path or config.RAW_DATA_PATH
    if not data_path.exists():
        raise DatasetNotFoundError(data_path)

    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset from {data_path} — shape {df.shape}")
    return df


def basic_overview(df: pd.DataFrame) -> dict:
    """Shape, columns, dtypes — the first things to check about any dataset."""
    return {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column, sorted descending."""
    n_missing = df.isnull().sum()
    pct_missing = (n_missing / len(df) * 100).round(2)
    report = pd.DataFrame({"n_missing": n_missing, "pct_missing": pct_missing})
    return report.sort_values("n_missing", ascending=False)


def duplicate_report(df: pd.DataFrame) -> dict:
    """Count of fully duplicated rows."""
    n_duplicates = int(df.duplicated().sum())
    return {
        "n_duplicate_rows": n_duplicates,
        "pct_duplicate_rows": round(n_duplicates / len(df) * 100, 2) if len(df) else 0.0,
    }


def unique_value_counts(df: pd.DataFrame, max_columns: int | None = None) -> pd.Series:
    """Number of unique values per column — useful for spotting ID-like or
    constant columns."""
    counts = df.nunique().sort_values(ascending=False)
    return counts if max_columns is None else counts.head(max_columns)


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Standard pandas describe() across all columns (numeric + categorical)."""
    return df.describe(include="all").transpose()


def target_distribution(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    """
    Class counts and percentages for the target column.

    Uses config.FEATURE_CONFIG['target'] unless a column name is given
    explicitly.
    """
    target_col = target_col or config.FEATURE_CONFIG["target"]
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in dataset. "
            f"Update src/config.py FEATURE_CONFIG['target'] to match the "
            f"real column name. Available columns: {list(df.columns)}"
        )
    counts = df[target_col].value_counts()
    pct = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct})


def class_imbalance_ratio(df: pd.DataFrame, target_col: str | None = None) -> float:
    """
    Ratio of majority-class count to minority-class count.
    A ratio near 1.0 means the classes are balanced; a large ratio (e.g. >10)
    signals a class-imbalance problem that must be addressed explicitly
    (see src/train.py / STEP 8 in the project plan).
    """
    dist = target_distribution(df, target_col)["count"]
    if len(dist) < 2:
        raise ValueError(
            "Target column has fewer than 2 classes present — cannot compute "
            "an imbalance ratio. Check the data and FEATURE_CONFIG['target']."
        )
    return round(dist.max() / dist.min(), 2)


def numerical_feature_summary(df: pd.DataFrame, numerical_cols: list[str]) -> pd.DataFrame:
    """Descriptive stats restricted to the configured numerical features."""
    present = [c for c in numerical_cols if c in df.columns]
    missing = set(numerical_cols) - set(present)
    if missing:
        logger.warning(f"Configured numerical columns not found in data: {missing}")
    return df[present].describe().transpose()


def categorical_feature_summary(df: pd.DataFrame, categorical_cols: list[str]) -> dict:
    """Value counts for each configured categorical feature."""
    present = [c for c in categorical_cols if c in df.columns]
    missing = set(categorical_cols) - set(present)
    if missing:
        logger.warning(f"Configured categorical columns not found in data: {missing}")
    return {col: df[col].value_counts().to_dict() for col in present}


def outlier_report_iqr(df: pd.DataFrame, numerical_cols: list[str]) -> pd.DataFrame:
    """
    IQR-based outlier counts per numerical column.
    This is a DETECTION report only — see src/preprocessing.py for the
    justified handling decision (outliers are NOT blindly dropped).
    """
    rows = []
    for col in numerical_cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        rows.append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "n_outliers": n_outliers,
                "pct_outliers": round(n_outliers / len(df) * 100, 2) if len(df) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def flag_potential_leakage_columns(df: pd.DataFrame, target_col: str | None = None) -> list[str]:
    """
    Heuristic first-pass flag for columns that MIGHT leak target information
    (e.g. a column that is a near-perfect proxy for the label, or a
    post-transaction outcome field such as "chargeback_filed" or
    "investigation_result"). This is a heuristic starting point for manual
    review, NOT a guarantee — the student must confirm each flagged column
    against the actual data dictionary.
    """
    target_col = target_col or config.FEATURE_CONFIG["target"]
    suspicious_keywords = [
        "result", "outcome", "flagged", "investigation", "chargeback",
        "resolved", "reviewed", "label", "is_fraud", "fraud_flag",
    ]
    flagged = []
    for col in df.columns:
        if col == target_col:
            continue
        lower_col = col.lower()
        if any(keyword in lower_col for keyword in suspicious_keywords):
            flagged.append(col)
    return flagged


def full_data_understanding_report(df: pd.DataFrame) -> dict:
    """Run every Step-1 check and return a single consolidated dict, ready
    to be printed in the notebook or dumped as JSON to reports/metrics/."""
    target_col = config.FEATURE_CONFIG["target"]
    report = {
        "overview": basic_overview(df),
        "missing_values": missing_value_report(df).to_dict(orient="index"),
        "duplicates": duplicate_report(df),
        "unique_value_counts": unique_value_counts(df).to_dict(),
    }
    try:
        report["target_distribution"] = target_distribution(df, target_col).to_dict(orient="index")
        report["class_imbalance_ratio"] = class_imbalance_ratio(df, target_col)
    except (KeyError, ValueError) as e:
        report["target_distribution_error"] = str(e)
    report["potential_leakage_columns"] = flag_potential_leakage_columns(df, target_col)
    return report


if __name__ == "__main__":
    config.ensure_directories()
    try:
        raw_df = load_raw_data()
        print(full_data_understanding_report(raw_df))
    except DatasetNotFoundError as e:
        print(e)
