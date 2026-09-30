"""
test_preprocessing.py
======================
Tests for src/preprocessing.py. These use small, synthetic, clearly-labeled
TEST FIXTURES (not the real dataset, and never presented as real results) to
verify the preprocessing pipeline behaves correctly.
"""

import pandas as pd
import pytest

from src.preprocessing import (
    build_preprocessing_pipeline,
    coerce_dtypes,
    drop_duplicate_rows,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "amount": [10.0, 20.0, None, 40.0, 10.0],
            "time": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-01"],
            "frequency": [1, 2, 3, 4, 1],
            "account_history": [100.0, 200.0, 150.0, None, 100.0],
            "merchant_category": ["grocery", "electronics", None, "grocery", "grocery"],
        }
    )


def test_drop_duplicate_rows_removes_exact_duplicates(sample_df):
    deduped, n_removed = drop_duplicate_rows(sample_df)
    assert n_removed == 1
    assert len(deduped) == 4


def test_coerce_dtypes_forces_numeric_and_string(sample_df):
    df = coerce_dtypes(
        sample_df,
        numerical_cols=["amount", "frequency", "account_history"],
        categorical_cols=["merchant_category"],
    )
    assert pd.api.types.is_numeric_dtype(df["amount"])
    assert pd.api.types.is_numeric_dtype(df["frequency"])
    assert pd.api.types.is_string_dtype(df["merchant_category"])


def test_pipeline_fits_and_transforms_without_error(sample_df):
    df = coerce_dtypes(
        sample_df,
        numerical_cols=["amount", "frequency", "account_history"],
        categorical_cols=["merchant_category"],
    )
    pipeline = build_preprocessing_pipeline(
        numerical_cols=["amount", "frequency", "account_history"],
        categorical_cols=["merchant_category"],
    )
    transformed = pipeline.fit_transform(df)
    # 3 numeric + one-hot columns for however many categories are present
    assert transformed.shape[0] == len(df)


def test_pipeline_handles_missing_values(sample_df):
    df = coerce_dtypes(
        sample_df,
        numerical_cols=["amount", "frequency", "account_history"],
        categorical_cols=["merchant_category"],
    )
    pipeline = build_preprocessing_pipeline(
        numerical_cols=["amount", "frequency", "account_history"],
        categorical_cols=["merchant_category"],
    )
    # Should not raise despite NaNs present in the fixture.
    pipeline.fit_transform(df)


def test_pipeline_handles_unseen_category_at_inference():
    train_df = pd.DataFrame(
        {
            "amount": [10.0, 20.0],
            "frequency": [1, 2],
            "account_history": [100.0, 200.0],
            "merchant_category": ["grocery", "electronics"],
        }
    )
    test_df = pd.DataFrame(
        {
            "amount": [15.0],
            "frequency": [1],
            "account_history": [120.0],
            "merchant_category": ["unseen_category"],
        }
    )
    pipeline = build_preprocessing_pipeline(
        numerical_cols=["amount", "frequency", "account_history"],
        categorical_cols=["merchant_category"],
    )
    pipeline.fit(train_df)
    # handle_unknown="ignore" must prevent a crash on an unseen category.
    pipeline.transform(test_df)
