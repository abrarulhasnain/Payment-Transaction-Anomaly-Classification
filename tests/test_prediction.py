"""
test_prediction.py
====================
Tests for src/predict.py.

We train a tiny throwaway pipeline on synthetic fixture data (clearly not
the real project results) purely so predict_transaction() can be exercised
without requiring the real dataset to be present in CI.
"""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.predict import (
    InvalidTransactionInput,
    predict_transaction,
    validate_transaction_input,
)
from src.preprocessing import build_preprocessing_pipeline


@pytest.fixture
def toy_pipeline():
    train_df = pd.DataFrame(
        {
            "amount": [10.0, 20.0, 500.0, 15.0, 480.0, 12.0],
            "time": [
                "2026-01-01 10:00:00",
                "2026-01-01 11:00:00",
                "2026-01-01 03:00:00",
                "2026-01-02 10:00:00",
                "2026-01-02 02:30:00",
                "2026-01-02 09:00:00",
            ],
            "frequency": [2, 3, 1, 2, 1, 3],
            "account_history": [50.0, 60.0, 40.0, 55.0, 45.0, 58.0],
            "merchant_category": ["grocery", "grocery", "electronics", "grocery", "electronics", "grocery"],
        }
    )
    y_train = pd.Series([0, 0, 1, 0, 1, 0])

    preprocessor = build_preprocessing_pipeline(
        numerical_cols=["amount", "frequency", "account_history"],
        categorical_cols=["merchant_category"],
    )
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", LogisticRegression())]
    )
    pipeline.fit(train_df, y_train)
    return pipeline


def valid_transaction():
    return {
        config.FEATURE_CONFIG["amount"]: 25.0,
        config.FEATURE_CONFIG["time"]: "2026-01-03 10:00:00",
        config.FEATURE_CONFIG["frequency"]: 2,
        config.FEATURE_CONFIG["merchant_category"]: "grocery",
        config.FEATURE_CONFIG["account_history"]: 52.0,
    }


def test_validate_transaction_input_accepts_valid_input():
    validate_transaction_input(valid_transaction())  # should not raise


def test_validate_transaction_input_rejects_missing_field():
    data = valid_transaction()
    del data[config.FEATURE_CONFIG["amount"]]
    with pytest.raises(InvalidTransactionInput):
        validate_transaction_input(data)


def test_validate_transaction_input_rejects_negative_amount():
    data = valid_transaction()
    data[config.FEATURE_CONFIG["amount"]] = -5.0
    with pytest.raises(InvalidTransactionInput):
        validate_transaction_input(data)


def test_validate_transaction_input_rejects_negative_frequency():
    data = valid_transaction()
    data[config.FEATURE_CONFIG["frequency"]] = -1
    with pytest.raises(InvalidTransactionInput):
        validate_transaction_input(data)


def test_predict_transaction_returns_expected_shape(toy_pipeline):
    result = predict_transaction(valid_transaction(), pipeline=toy_pipeline)
    assert set(result.keys()) == {"prediction", "probability"}
    assert result["prediction"] in config.CLASS_NAMES.values()
    assert result["probability"] is None or 0.0 <= result["probability"] <= 1.0


def test_predict_transaction_raises_on_invalid_input(toy_pipeline):
    data = valid_transaction()
    data[config.FEATURE_CONFIG["amount"]] = "not_a_number"
    with pytest.raises(InvalidTransactionInput):
        predict_transaction(data, pipeline=toy_pipeline)
