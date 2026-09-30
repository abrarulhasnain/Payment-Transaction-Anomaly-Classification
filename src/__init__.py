"""
Payment Transaction Anomaly Classification — source package.

Modules:
    config        - paths, feature mapping, reproducibility settings
    utils         - shared helpers (logging, IO)
    data_loader   - dataset loading + data-understanding functions
    preprocessing - cleaning + sklearn preprocessing pipeline
    features      - feature engineering
    train         - model training, tuning, comparison
    evaluate      - metrics, confusion matrices, error analysis
    predict       - single-transaction prediction API used by the Streamlit app
"""
