"""
test_app_inputs.py
====================
Additional edge-case tests focused on the kinds of malformed input the
Streamlit form could realistically produce (empty strings, wrong types),
distinct from the core prediction-logic tests in test_prediction.py.
"""

import pytest

from src import config
from src.predict import InvalidTransactionInput, validate_transaction_input


def base_transaction():
    return {
        config.FEATURE_CONFIG["amount"]: 10.0,
        config.FEATURE_CONFIG["time"]: "2026-01-01 10:00:00",
        config.FEATURE_CONFIG["frequency"]: 1,
        config.FEATURE_CONFIG["merchant_category"]: "grocery",
        config.FEATURE_CONFIG["account_history"]: 50.0,
    }


def test_empty_dict_raises():
    with pytest.raises(InvalidTransactionInput):
        validate_transaction_input({})


def test_all_required_fields_present_passes():
    validate_transaction_input(base_transaction())  # should not raise


def test_string_amount_raises():
    data = base_transaction()
    data[config.FEATURE_CONFIG["amount"]] = "twenty"
    with pytest.raises(InvalidTransactionInput):
        validate_transaction_input(data)


def test_zero_amount_and_frequency_are_valid_edge_case():
    data = base_transaction()
    data[config.FEATURE_CONFIG["amount"]] = 0.0
    data[config.FEATURE_CONFIG["frequency"]] = 0
    validate_transaction_input(data)  # zero is a valid, non-negative value


def test_missing_merchant_category_raises():
    data = base_transaction()
    del data[config.FEATURE_CONFIG["merchant_category"]]
    with pytest.raises(InvalidTransactionInput):
        validate_transaction_input(data)
