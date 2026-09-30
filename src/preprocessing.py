"""
preprocessing.py
=================
Steps 2 & 3 — Data Cleaning and Data-Leakage Prevention.

This module builds ONE sklearn Pipeline/ColumnTransformer that is:
    - fit ONLY on the training split (never on test data),
    - reused identically at inference time (saved via joblib),
so that training-time and Streamlit-time preprocessing can never drift apart.

Cleaning decisions implemented here, and why:

1. Missing numerical values -> median imputation
   - What was detected: possible missing values in numeric columns.
   - Why it's a problem: most scikit-learn estimators cannot handle NaN.
   - Method: median imputation (SimpleImputer, strategy="median").
   - Why median: transaction amounts/frequencies are typically right-skewed,
     so the median is more robust to outliers than the mean.

2. Missing categorical values -> most-frequent imputation
   - Why: preserves a valid, real category rather than inventing a new one
     that never occurred in the data; simple and explainable for a viva.

3. Numerical scaling -> StandardScaler
   - Why: Logistic Regression and KNN are distance/gradient based and are
     sensitive to feature scale; Decision Trees are scale-invariant but
     scaling them causes no harm, which keeps ONE shared pipeline valid for
     all three baseline models.

4. Categorical encoding -> OneHotEncoder(handle_unknown="ignore")
   - Why one-hot: merchant/category is nominal (no inherent order).
   - handle_unknown="ignore": prevents a crash in Streamlit if a user
     selects/enters a category unseen during training, instead of silently
     mis-encoding it.

5. Outliers are NOT blindly deleted.
   - Extreme transaction amounts may be exactly the anomalies we are trying
     to detect, so dropping them by an unconditional IQR rule would remove
     real signal (and could remove positive-class examples entirely).
   - Outlier counts are reported (see data_loader.outlier_report_iqr) for
     awareness, but handled implicitly through scaling + robust models
     rather than row deletion, unless the actual dataset's error-analysis
     stage shows a documented reason to treat a specific outlier group as a
     data-entry error.

Leakage prevention:
   - `build_preprocessing_pipeline()` returns an UNFIT sklearn transformer.
   - Callers must call `.fit(X_train)` — never `.fit(X)` on the full data —
     and reuse the fitted object (`.transform(X_test)`) for the test split
     and for any future prediction request.
   - `src/train.py` enforces this by only ever calling `.fit` inside the
     train/test-split-aware training function.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fully duplicated rows. Applied BEFORE the train/test split so
    the same physical transaction cannot appear in both splits."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    return df, removed


def coerce_dtypes(
    df: pd.DataFrame,
    numerical_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Force configured numerical columns to numeric and categorical columns to
    string dtype. Values that cannot be coerced become NaN (and are then
    handled by the imputers in the pipeline) rather than crashing the run.
    """
    numerical_cols = numerical_cols or config.NUMERICAL_FEATURES
    categorical_cols = categorical_cols or config.CATEGORICAL_FEATURES
    df = df.copy()
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def build_preprocessing_pipeline(
    numerical_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> ColumnTransformer:
    """
    Build (but do NOT fit) the shared preprocessing ColumnTransformer.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfit transformer. Fit only on X_train.
    """
    numerical_cols = numerical_cols or config.NUMERICAL_FEATURES
    categorical_cols = categorical_cols or config.CATEGORICAL_FEATURES

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Retrieve human-readable feature names AFTER fitting, e.g. for
    coefficient/importance interpretation in Step 10.
    Must be called only after preprocessor.fit(...) has run.
    """
    return list(preprocessor.get_feature_names_out())
