"""
evaluate.py
============
Steps 7, 9, 10 — Model evaluation, error analysis (esp. false positives),
and interpretability.

Every function computes real metrics from the model/predictions passed in.
Nothing here contains a pre-filled number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config
from src.preprocessing import get_output_feature_names
from src.utils import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Accuracy, precision, recall, F1, and ROC-AUC (if probabilities are
    available) for the positive (anomalous) class."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError as e:
            logger.warning(f"ROC-AUC could not be computed: {e}")
            metrics["roc_auc"] = None
    return metrics


def evaluate_model(pipeline, X_test, y_test, model_name: str) -> dict:
    """Run a fitted pipeline on the test split and package predictions,
    metrics, and the confusion matrix for downstream reporting."""
    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"[{model_name}] metrics: {metrics}")

    return {
        "model_name": model_name,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_array": cm,  # kept for in-process plotting; stripped before JSON save
    }


def error_breakdown(y_true, y_pred) -> dict:
    """
    Split predictions into TP / TN / FP / FN counts and rates.

    FP = routine transaction incorrectly flagged as anomalous
         (a "false alarm" — costly in a real monitoring system because it
         wastes investigator time / annoys legitimate customers).
    FN = anomalous transaction incorrectly classified as routine
         (a missed anomaly — the more dangerous error in a fraud context).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    n_actual_negative = tn + fp
    n_actual_positive = tp + fn

    return {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": round(fp / n_actual_negative, 4) if n_actual_negative else None,
        "false_negative_rate": round(fn / n_actual_positive, 4) if n_actual_positive else None,
    }


def error_case_dataframe(X_test: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
    """
    Return the original (pre-preprocessing) test rows annotated with the
    true label, predicted label, and an 'error_type' column
    (TP/TN/FP/FN) — the raw material for investigating behavioral patterns
    among errors (Step 9).
    """
    df = X_test.copy().reset_index(drop=True)
    df["y_true"] = np.asarray(y_true)
    df["y_pred"] = np.asarray(y_pred)

    def classify(row):
        if row["y_true"] == 1 and row["y_pred"] == 1:
            return "TP"
        if row["y_true"] == 0 and row["y_pred"] == 0:
            return "TN"
        if row["y_true"] == 0 and row["y_pred"] == 1:
            return "FP"
        return "FN"

    df["error_type"] = df.apply(classify, axis=1)
    return df


def false_positive_pattern_summary(error_df: pd.DataFrame, group_cols: list[str]) -> dict:
    """
    For each candidate grouping column (e.g. merchant_category,
    frequency_bucket, hour_of_day), compare the distribution of that column
    among false positives vs. true negatives, to look for patterns in what
    kinds of routine transactions get incorrectly flagged.

    Returns real, computed value_counts — no assumptions about which
    patterns will appear.
    """
    summary = {}
    fp_df = error_df[error_df["error_type"] == "FP"]
    tn_df = error_df[error_df["error_type"] == "TN"]
    for col in group_cols:
        if col not in error_df.columns:
            continue
        summary[col] = {
            "false_positive_distribution": fp_df[col].value_counts(normalize=True).round(3).to_dict(),
            "true_negative_distribution": tn_df[col].value_counts(normalize=True).round(3).to_dict(),
        }
    return summary


def logistic_regression_coefficients(pipeline) -> pd.DataFrame | None:
    """
    Step 10 — Interpretability for Logistic Regression.
    Returns a dataframe of (feature, coefficient) sorted by absolute value,
    or None if the pipeline's model is not a LogisticRegression.
    """
    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")
    if model is None or not hasattr(model, "coef_"):
        return None
    feature_names = get_output_feature_names(preprocessor)
    coefs = model.coef_[0]
    df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    df["abs_coefficient"] = df["coefficient"].abs()
    return df.sort_values("abs_coefficient", ascending=False).drop(columns="abs_coefficient")


def tree_feature_importances(pipeline) -> pd.DataFrame | None:
    """
    Step 10 — Interpretability for Decision Tree / Random Forest.
    Returns a dataframe of (feature, importance) sorted descending, or None
    if the pipeline's model has no feature_importances_.
    """
    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")
    if model is None or not hasattr(model, "feature_importances_"):
        return None
    feature_names = get_output_feature_names(preprocessor)
    df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    )
    return df.sort_values("importance", ascending=False)


def interpretation_note() -> str:
    """A reusable disclaimer to attach next to any importance/coefficient
    table in the notebook, paper, or app."""
    return (
        "These values describe association/importance WITHIN this model, "
        "not a causal claim about what makes a transaction anomalous. "
        "Correlated or confounded features can show non-obvious importance; "
        "conclusions should be validated against domain knowledge before "
        "being treated as causal."
    )
