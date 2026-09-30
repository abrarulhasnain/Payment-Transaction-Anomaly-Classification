"""
predict.py
==========
Step 14 — Prediction Function.

Provides `predict_transaction()`, the single clean API used by BOTH the
Streamlit app (app/app.py) and the test suite (tests/test_prediction.py),
so there is exactly one code path from "raw transaction input" to
"prediction output" — matching the pipeline diagram in the project plan:

    Raw Input -> Validation -> Preprocessing -> Feature Engineering
              -> Trained Model -> Prediction -> Probability/Risk Indicator

Nothing in this file hard-codes an example prediction; `_DEMO_RESULT` below
exists only to document the expected return shape and is never returned by
`predict_transaction()` itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src import config
from src.features import engineer_features
from src.utils import get_logger

logger = get_logger(__name__)

REQUIRED_FIELDS = list(config.FEATURE_CONFIG.keys())
REQUIRED_FIELDS.remove("target")  # target is never part of a prediction request

# Documents the expected output shape only — NOT a real prediction.
_DEMO_RESULT_SHAPE = {"prediction": "<Routine|Anomalous>", "probability": "<float 0-1>"}


class InvalidTransactionInput(ValueError):
    """Raised when a transaction dict fails validation."""


def validate_transaction_input(transaction_data: dict) -> None:
    """
    Validate a raw transaction dict before it reaches preprocessing.

    Checks:
        - all required fields (per config.FEATURE_CONFIG) are present
        - amount is numeric and non-negative
        - frequency is numeric and non-negative
    Raises InvalidTransactionInput with a clear message on failure.
    """
    missing = [
        field for field in REQUIRED_FIELDS
        if config.FEATURE_CONFIG[field] not in transaction_data
    ]
    if missing:
        raise InvalidTransactionInput(
            f"Missing required field(s): {missing}. "
            f"Expected keys (from config.FEATURE_CONFIG): "
            f"{[config.FEATURE_CONFIG[f] for f in REQUIRED_FIELDS]}"
        )

    amount_key = config.FEATURE_CONFIG["amount"]
    amount_val = transaction_data[amount_key]
    if not isinstance(amount_val, (int, float)) or amount_val < 0:
        raise InvalidTransactionInput(
            f"'{amount_key}' must be a non-negative number, got: {amount_val!r}"
        )

    frequency_key = config.FEATURE_CONFIG["frequency"]
    frequency_val = transaction_data[frequency_key]
    if not isinstance(frequency_val, (int, float)) or frequency_val < 0:
        raise InvalidTransactionInput(
            f"'{frequency_key}' must be a non-negative number, got: {frequency_val!r}"
        )


def load_trained_pipeline(path: Path | None = None):
    """
    Load the saved end-to-end sklearn Pipeline (preprocessing + model).

    Raises FileNotFoundError with a clear message if training has not been
    run yet — never falls back to a fabricated/untrained model.
    """
    model_path = path or config.FINAL_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `python -m "
            "src.train` (after placing the dataset in data/raw/) before "
            "calling predict_transaction()."
        )
    return joblib.load(model_path)


def predict_transaction(transaction_data: dict, pipeline=None) -> dict[str, Any]:
    """
    Predict whether a single transaction is Routine or Anomalous.

    Parameters
    ----------
    transaction_data : dict
        Keys must match the raw column names configured in
        config.FEATURE_CONFIG (e.g. {"amount": 120.0, "time": "...", ...}).
    pipeline : sklearn Pipeline, optional
        Pass a pre-loaded pipeline to avoid reloading from disk on every
        call (used by the Streamlit app, which loads it once per session).

    Returns
    -------
    dict with keys "prediction" (str) and "probability" (float, probability
    of the ANOMALOUS class) when the model supports predict_proba, else
    "probability": None.
    """
    validate_transaction_input(transaction_data)

    if pipeline is None:
        pipeline = load_trained_pipeline()

    row = {config.FEATURE_CONFIG[field]: transaction_data[config.FEATURE_CONFIG[field]] for field in REQUIRED_FIELDS}
    X = pd.DataFrame([row])
    X = engineer_features(X)

    pred = pipeline.predict(X)[0]
    proba = None
    if hasattr(pipeline, "predict_proba"):
        proba = float(pipeline.predict_proba(X)[0][1])

    label = config.CLASS_NAMES.get(int(pred), str(pred))

    return {"prediction": label, "probability": proba}
