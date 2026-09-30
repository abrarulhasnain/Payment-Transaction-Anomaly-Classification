"""
config.py — Centralized configuration for the Payment Transaction
Anomaly Classification capstone project.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILENAME = "transactions.csv"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME

MODELS_DIR = PROJECT_ROOT / "models"
PREPROCESSING_PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
ERROR_ANALYSIS_DIR = REPORTS_DIR / "error_analysis"

# ---------------------------------------------------------------------------
# DATASET: PaySim — "Synthetic Financial Datasets For Fraud Detection"
# Source: https://www.kaggle.com/datasets/ealaxi/paysim1 (CC BY-SA 4.0)
# Raw columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
#              nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
# ---------------------------------------------------------------------------
FEATURE_CONFIG = {
    "amount": "amount",
    "time": "step",
    "frequency": "transaction_count_orig",   # ENGINEERED — see features.py
    "merchant_category": "type",
    "account_history": "oldbalanceOrg",
    "target": "isFraud",
}

TARGET_POSITIVE_LABEL = 1
CLASS_NAMES = {0: "Routine", 1: "Anomalous"}

# Derived from FEATURE_CONFIG's VALUES (not its keys) so these never drift
# out of sync with FEATURE_CONFIG again.
NUMERICAL_FEATURES = [
    FEATURE_CONFIG["amount"],
    FEATURE_CONFIG["time"],
    FEATURE_CONFIG["frequency"],
    FEATURE_CONFIG["account_history"],
]
CATEGORICAL_FEATURES = [FEATURE_CONFIG["merchant_category"]]

# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# MODEL HYPERPARAMETER SEARCH SPACES
# ---------------------------------------------------------------------------
PARAM_GRIDS = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },
    "knn": {
        "model__n_neighbors": [3, 5, 7, 9, 15],
        "model__weights": ["uniform", "distance"],
    },
    "decision_tree": {
        "model__max_depth": [3, 5, 8, 12, None],
        "model__min_samples_leaf": [1, 5, 10, 20],
        "model__class_weight": [None, "balanced"],
    },
}

TUNING_SCORING = "f1"


def ensure_directories() -> None:
    """Create all project directories if they do not already exist."""
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        ERROR_ANALYSIS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print(f"Project root: {PROJECT_ROOT}")
    print("All project directories verified/created.")