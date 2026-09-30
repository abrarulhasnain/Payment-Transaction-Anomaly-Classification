"""
train.py
========
Steps 6, 7, 8, 11, 12, 13 — Baseline model, model comparison, class-imbalance
handling, hyperparameter tuning, final model selection, and pipeline saving.

Running this script end-to-end:
    1. Loads the raw dataset (fails loudly via DatasetNotFoundError if it's
       not present — see src/utils.py).
    2. Cleans it (drop duplicates, coerce dtypes).
    3. Splits into train/test BEFORE any fitting happens (leakage prevention).
    4. Engineers features (leakage-documented, see src/features.py).
    5. Fits the shared preprocessing pipeline on the TRAIN split only.
    6. Trains three foundational models (Logistic Regression, KNN,
       Decision Tree) inside one shared sklearn Pipeline object each, so
       preprocessing + model are always saved/loaded together.
    7. Runs lightweight hyperparameter tuning with cross-validation
       (GridSearchCV, scoring = F1 on the anomalous class).
    8. Evaluates all three on the held-out test split (via evaluate.py).
    9. Selects a final model using documented, non-accuracy-only reasoning
       (see select_final_model()) and saves it + the preprocessing pipeline
       to models/.

No metric value in this file is hard-coded — every number is computed from
whatever dataset is actually present when this script runs.
"""

from __future__ import annotations

import json
import time

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src import config
from src.data_loader import load_raw_data, class_imbalance_ratio
from src.evaluate import evaluate_model
from src.features import (
    engineer_features,
    ENGINEERED_NUMERICAL_FEATURES,
    ENGINEERED_CATEGORICAL_FEATURES,
)
from src.preprocessing import (
    build_preprocessing_pipeline,
    coerce_dtypes,
    drop_duplicate_rows,
)
from src.utils import get_logger, save_json

logger = get_logger(__name__)

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
    "knn": KNeighborsClassifier(),
    "decision_tree": DecisionTreeClassifier(random_state=config.RANDOM_STATE),
}


def prepare_data():
    """Load, clean, split, and feature-engineer the dataset. Returns
    X_train, X_test, y_train, y_test, plus the full numerical/categorical
    column lists actually usable for modeling."""
    df = load_raw_data()
    df, n_dropped = drop_duplicate_rows(df)
    logger.info(f"Dropped {n_dropped} duplicate rows.")

    df = coerce_dtypes(df)

    imbalance_ratio = class_imbalance_ratio(df)
    logger.info(f"Class imbalance ratio (majority:minority) = {imbalance_ratio}:1")

    target_col = config.FEATURE_CONFIG["target"]
    X_raw = df.drop(columns=[target_col])
    y = df[target_col]

    # Split BEFORE feature engineering that could depend on distributional
    # stats, and BEFORE fitting any preprocessing — this is the core
    # leakage-prevention control for this project (Step 3).
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    X_train = engineer_features(X_train_raw)
    X_test = engineer_features(X_test_raw)

    numerical_cols = [
        c for c in config.NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES
        if c in X_train.columns
    ]
    categorical_cols = [
        c for c in config.CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES
        if c in X_train.columns
    ]

    return X_train, X_test, y_train, y_test, numerical_cols, categorical_cols


def build_model_pipeline(model_name: str, numerical_cols, categorical_cols) -> Pipeline:
    """Combine the shared preprocessor with a given estimator into one
    end-to-end sklearn Pipeline (so preprocessing is always saved together
    with the model that was trained on its output)."""
    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)
    model = MODEL_REGISTRY[model_name]
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def tune_model(pipeline: Pipeline, model_name: str, X_train, y_train) -> GridSearchCV:
    """Run GridSearchCV with cross-validation for a given model pipeline."""
    param_grid = config.PARAM_GRIDS.get(model_name, {})
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=config.TUNING_SCORING,
        cv=config.CV_FOLDS,
        n_jobs=-1,
        refit=True,
    )
    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info(
        f"[{model_name}] best_params={search.best_params_} "
        f"best_cv_{config.TUNING_SCORING}={search.best_score_:.4f} "
        f"(tuned in {elapsed:.1f}s)"
    )
    return search, elapsed


def select_final_model(results: dict) -> str:
    """
    Choose the final model using documented, multi-criteria reasoning —
    NOT simply "highest accuracy".

    Selection criteria (in priority order, matching the project's fraud/
    anomaly-monitoring context where missed anomalies and false alarms both
    carry real cost):
        1. Recall on the anomalous class (missing a real anomaly is the
           most costly error type in this domain).
        2. F1-score on the anomalous class (balances recall against
           precision so the model is not flagging everything as anomalous).
        3. Interpretability (Logistic Regression / Decision Tree are easier
           to explain in a viva than KNN, which has no coefficients or
           native feature importances).
        4. Training/inference complexity appropriate for an educational
           prototype.

    This function only ranks whatever metrics were actually computed by
    evaluate.py for the models that were actually trained in this run — it
    does not assume which model will win ahead of time.
    """
    ranked = sorted(
        results.items(),
        key=lambda item: (
            item[1]["metrics"]["recall"],
            item[1]["metrics"]["f1"],
        ),
        reverse=True,
    )
    best_name = ranked[0][0]
    logger.info(
        "Model ranking (by recall, then F1, on the anomalous class): "
        + ", ".join(f"{name}={m['metrics']['recall']:.3f}/{m['metrics']['f1']:.3f}" for name, m in ranked)
    )
    return best_name


def run_training_pipeline():
    """Full end-to-end training + comparison + selection + saving."""
    config.ensure_directories()

    X_train, X_test, y_train, y_test, numerical_cols, categorical_cols = prepare_data()

    results = {}
    fitted_pipelines = {}

    for model_name in MODEL_REGISTRY:
        logger.info(f"--- Training {model_name} ---")
        pipeline = build_model_pipeline(model_name, numerical_cols, categorical_cols)
        search, elapsed = tune_model(pipeline, model_name, X_train, y_train)
        best_pipeline = search.best_estimator_
        fitted_pipelines[model_name] = best_pipeline

        eval_result = evaluate_model(best_pipeline, X_test, y_test, model_name)
        eval_result["best_params"] = search.best_params_
        eval_result["cv_best_score"] = search.best_score_
        eval_result["training_time_seconds"] = round(elapsed, 2)
        results[model_name] = eval_result

        # Save EVERY model, not just the final one, so the app can offer a
        # dropdown to switch between them.
        joblib.dump(best_pipeline, config.MODELS_DIR / f"{model_name}.joblib")

    save_json(
        {name: {k: v for k, v in r.items() if k != "confusion_matrix_array"} for name, r in results.items()},
        config.METRICS_DIR / "model_comparison.json",
    )

    final_model_name = select_final_model(results)
    final_pipeline = fitted_pipelines[final_model_name]

    joblib.dump(final_pipeline, config.FINAL_MODEL_PATH)
    joblib.dump(final_pipeline.named_steps["preprocessor"], config.PREPROCESSING_PIPELINE_PATH)

    model_card = {
        "final_model": final_model_name,
        "selection_reasoning": (
            "Selected using recall then F1 on the anomalous class, not "
            "accuracy alone. See train.select_final_model() docstring for "
            "full criteria."
        ),
        "metrics": results[final_model_name]["metrics"],
        "best_params": results[final_model_name]["best_params"],
        "numerical_features_used": numerical_cols,
        "categorical_features_used": categorical_cols,
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
    }
    save_json(model_card, config.MODEL_CARD_PATH)

    logger.info(f"Final model saved: {final_model_name} -> {config.FINAL_MODEL_PATH}")
    logger.info(f"Preprocessing pipeline saved -> {config.PREPROCESSING_PIPELINE_PATH}")
    logger.info(f"Model comparison metrics saved -> {config.METRICS_DIR / 'model_comparison.json'}")

    return results, final_model_name


if __name__ == "__main__":
    run_training_pipeline()
