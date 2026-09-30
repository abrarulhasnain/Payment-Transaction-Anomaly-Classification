"""
app.py
======
Step 15/16 — Streamlit application.

Run with:
    streamlit run app/app.py

This app loads the ACTUAL trained pipeline saved by src/train.py
(models/final_model.joblib) and calls the SAME predict_transaction()
function used by the test suite — there is no separate/duplicated
prediction logic here, so the app can never drift from what was trained.

If no trained model exists yet, the app tells the user exactly what to run
instead of faking a result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow running via `streamlit run app/app.py` from the project root.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.predict import (  # noqa: E402
    InvalidTransactionInput,
    load_trained_pipeline,
    predict_transaction,
)

st.set_page_config(page_title="Payment Transaction Anomaly Classification", page_icon="🏦", layout="centered")

st.title("🏦 Payment Transaction Anomaly Detection")

st.markdown(
    """
### About the Project
This is an educational capstone prototype for **binary classification of
payment transactions** into **Routine** or **Anomalous**, built with
Logistic Regression / KNN / Decision Tree baselines on a free/open-source
stack (scikit-learn + Streamlit).

**This is NOT a production fraud-detection system.** It is a university
capstone demo intended to show an end-to-end ML workflow, and its
predictions should not be used for real financial decisions.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Load the trained pipeline once per session
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = {
    "logistic_regression": "Logistic Regression",
    "knn": "K-Nearest Neighbors",
    "decision_tree": "Decision Tree",
}

def _saved_models():
    return {
        key: config.MODELS_DIR / f"{key}.joblib"
        for key in AVAILABLE_MODELS
        if (config.MODELS_DIR / f"{key}.joblib").exists()
    }

@st.cache_resource
def _get_pipeline(model_key: str):
    import joblib
    return joblib.load(config.MODELS_DIR / f"{model_key}.joblib")

saved_models = _saved_models()
model_available = len(saved_models) > 0

selected_model_key = None
if model_available:
    selected_model_key = st.selectbox(
        "Choose which trained model to use for prediction",
        options=list(saved_models.keys()),
        format_func=lambda k: AVAILABLE_MODELS[k],
    )

if not model_available:
    st.warning(
        "No trained model was found yet.\n\n"
        "To generate one:\n"
        "1. Place the dataset CSV in `data/raw/` (see `data/README.md`).\n"
        "2. Update `src/config.py` -> `FEATURE_CONFIG` with the real column names.\n"
        "3. Run `python -m src.train` from the project root.\n"
        "4. Restart this app.\n\n"
        "The form below is shown for demonstration of the input layout, but "
        "the **Predict** button will not produce a result until a model has "
        "been trained — this app does not fabricate predictions."
    )

st.header("Transaction Details")

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Transaction amount", min_value=0.0, value=0.0, step=1.0)
    time_value = st.number_input(
        "Transaction step (hour number in simulation, 1–744)",
        min_value=1,
        max_value=744,
        value=1,
        step=1,
        help="PaySim dataset's 'step' = simulated hour. 1 step = 1 hour, 744 steps = 30 days.",
        )
with col2:
    frequency = st.number_input(
        "Transaction frequency (recent transaction count for this account)",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )
    merchant_category = st.text_input("Merchant / category", value="")

st.header("Account / Historical Information")
account_history = st.number_input(
    "Account history indicator (e.g. average historical transaction amount)",
    min_value=0.0,
    value=0.0,
    step=1.0,
    help="Placeholder input — adapt this control once the real dataset's "
    "account-history feature definition is known.",
)

st.divider()
st.header("Prediction")

predict_clicked = st.button("Predict", type="primary", disabled=not model_available)

if predict_clicked:
    transaction_data = {
        config.FEATURE_CONFIG["amount"]: amount,
        config.FEATURE_CONFIG["time"]: time_value,
        config.FEATURE_CONFIG["frequency"]: frequency,
        config.FEATURE_CONFIG["merchant_category"]: merchant_category,
        config.FEATURE_CONFIG["account_history"]: account_history,
    }
    try:
        pipeline =  _get_pipeline(selected_model_key)
        result = predict_transaction(transaction_data, pipeline=pipeline)
        label = result["prediction"]
        proba = result["probability"]

        if label == "Anomalous":
            st.error(f"Prediction: **ANOMALOUS**")
        else:
            st.success(f"Prediction: **ROUTINE**")

        if proba is not None:
            st.metric("Anomaly probability", f"{proba * 100:.1f}%")
            st.progress(min(max(proba, 0.0), 1.0))
        else:
            st.caption("The selected model does not output a probability, only a class label.")

        with st.expander("Prediction explanation"):
            st.write(
                "This prediction was produced by the trained pipeline saved at "
                f"`{config.FINAL_MODEL_PATH.relative_to(config.PROJECT_ROOT)}`, "
                "which applies the exact same preprocessing and feature "
                "engineering used during training."
            )
    except InvalidTransactionInput as e:
        st.error(f"Invalid input: {e}")

st.divider()
st.header("Model Information")
if model_available and config.MODEL_CARD_PATH.exists():
    import json

    with open(config.MODEL_CARD_PATH) as f:
        model_card = json.load(f)

    st.write(f"**Selected model:** `{model_card['final_model']}`")
    st.caption(model_card["selection_reasoning"])

    m = model_card["metrics"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
    col2.metric("Precision", f"{m['precision']*100:.2f}%")
    col3.metric("Recall", f"{m['recall']*100:.1f}%")
    col4.metric("F1-score", f"{m['f1']:.4f}")
    if m.get("roc_auc") is not None:
        st.metric("ROC-AUC", f"{m['roc_auc']:.4f}")

    with st.expander("Best hyperparameters found"):
        st.write(model_card["best_params"])

    with st.expander("Features used"):
        st.write("**Numerical:**", ", ".join(model_card["numerical_features_used"]))
        st.write("**Categorical:**", ", ".join(model_card["categorical_features_used"]))
else:
    st.caption("Model card will appear here after training (`models/model_card.json`).")
st.divider()
st.info(
    "**Disclaimer:** This application is an educational prototype built for "
    "a BS Artificial Intelligence capstone project. It is not audited, not "
    "production-grade, and must not be used for real transaction monitoring "
    "or financial decision-making."
)
