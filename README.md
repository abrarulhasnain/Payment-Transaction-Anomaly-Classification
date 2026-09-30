# Payment Transaction Anomaly Classification

Artificial Intelligence capstone project — 10-day scope — Problem 26
(FinTech, Binary Classification).

> **Status: framework complete, awaiting final dataset.** This repository
> is fully built and reproducible end-to-end, but every metric/figure
> placeholder marked `[GENERATED AFTER TRAINING]` will only contain real
> numbers once an approved dataset is placed in `data/raw/` and
> `python -m src.train` is run. See [`data/README.md`](data/README.md).

---

## 1. Project Overview

This project classifies payment transaction records into **Routine** or
**Anomalous** using transaction characteristics (amount, time, frequency,
merchant/category, account history). It was built as an educational
prototype for a  AI capstone — **not** a production financial-security
system.

## 2. Problem Statement

> Classify transaction records into routine and anomalous categories using
> transaction characteristics.

## 3. Objective

Build an end-to-end, zero-cost ML solution: data investigation → EDA →
preprocessing → feature engineering → model training/comparison →
evaluation → error analysis → final model selection → Streamlit app →
full documentation (paper, presentation, viva prep).

## 4. Research Direction

The project specifically investigates **behavioral patterns in
classification errors**, with emphasis on **false positives** (routine
transactions incorrectly flagged as anomalous) — see
[`src/evaluate.py`](src/evaluate.py) and Step 9 in the notebook.

## 5. Dataset

Not yet finalized. A recommended free/open candidate (PaySim, on Kaggle)
is documented with full source, license-verification instructions, feature
list, and known limitations in [`data/README.md`](data/README.md).

## 6. Features

Configured via a single mapping in [`src/config.py`](src/config.py)
(`FEATURE_CONFIG`) so the codebase is not hard-coded to any one dataset's
column names:

- Transaction amount
- Transaction time
- Transaction frequency
- Merchant/category indicator
- Account-history feature

Engineered features (with leakage-risk documentation) are in
[`src/features.py`](src/features.py): `hour_of_day`, `day_of_week`,
`amount_log`, `amount_deviation_from_history`, `frequency_bucket`.

## 7. Methodology

```text
Raw Data → Cleaning → Train/Test Split → Feature Engineering
        → Preprocessing (fit on train only) → Model Training (3 baselines)
        → Hyperparameter Tuning (CV) → Evaluation → Error Analysis
        → Final Model Selection → Save Pipeline → Streamlit Inference
```

See [`paper/technical_paper.md`](paper/technical_paper.md) for full
methodology write-up.

## 8. EDA

Target distribution, amount distribution, amount-by-class, correlation
heatmap, and more — implemented in the notebook
([`notebooks/01_payment_transaction_anomaly_analysis.ipynb`](notebooks/01_payment_transaction_anomaly_analysis.ipynb))
using reusable functions from `src/data_loader.py`.

## 9. Models

Three foundational baselines, compared on equal footing (same preprocessing
pipeline, same CV scheme):

| Model | Why included |
|---|---|
| Logistic Regression | Simple, interpretable baseline (coefficients) |
| K-Nearest Neighbors | Non-parametric, distance-based comparison |
| Decision Tree | Non-linear, interpretable (feature importances) |

## 10. Evaluation

Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix for
every model — computed in [`src/evaluate.py`](src/evaluate.py). The final
model is **not** selected by accuracy alone; see
`src/train.py -> select_final_model()` for the documented, multi-criteria
selection logic (recall → F1 → interpretability → complexity).

## 11. Error Analysis

TP/TN/FP/FN breakdown, false-positive/negative rates, and behavioral
pattern analysis of errors across merchant category, time, and frequency —
see `src/evaluate.py -> error_breakdown / error_case_dataframe /
false_positive_pattern_summary`.

## 12. Final Model

`[GENERATED AFTER TRAINING]` — will be recorded in
`models/model_card.json` and summarized here once training has run on the
real dataset.

## 13. Project Structure

```text
payment-transaction-anomaly-classification/
│
├── data/
│   ├── raw/                  # place the approved dataset CSV here
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   └── 01_payment_transaction_anomaly_analysis.ipynb
│
├── src/
│   ├── config.py              # paths + FEATURE_CONFIG mapping (edit this first)
│   ├── data_loader.py         # Step 1 — data understanding
│   ├── preprocessing.py       # Steps 2–3 — cleaning + leakage-safe pipeline
│   ├── features.py            # Step 5 — feature engineering
│   ├── train.py                # Steps 6–8, 11–13 — training, tuning, selection
│   ├── evaluate.py             # Steps 7, 9, 10 — metrics, error analysis, interpretability
│   ├── predict.py              # Step 14 — prediction API used by the app
│   └── utils.py
│
├── models/
│   ├── preprocessing_pipeline.joblib   # created by training
│   └── final_model.joblib              # created by training
│
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── error_analysis/
│
├── app/
│   └── app.py                 # Streamlit application
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│   └── test_app_inputs.py
│
├── paper/technical_paper.md
├── presentation/presentation.md
├── viva/
│   ├── viva_questions.md
│   ├── viva_answers.md
│   └── demo_script.md
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## 14. Installation

```bash
git clone <repository-url>
cd payment-transaction-anomaly-classification
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Then:

```bash
pip install -r requirements.txt
```

## 15. Dataset Placement

See [`data/README.md`](data/README.md). In short:

1. Obtain the approved dataset CSV.
2. Place it at `data/raw/transactions.csv` (or update
   `src/config.py -> RAW_DATA_FILENAME`).
3. Update `src/config.py -> FEATURE_CONFIG` with the dataset's real column
   names and `TARGET_POSITIVE_LABEL`.

## 16. Running the Notebook

```bash
jupyter notebook notebooks/01_payment_transaction_anomaly_analysis.ipynb
```

Run all cells top-to-bottom. Cells that require the dataset will raise a
clear error (not fabricate output) if it isn't present yet.

## 17. Training the Model

```bash
python -m src.train
```

This performs cleaning, splitting, feature engineering, preprocessing,
training + tuning all three baselines, evaluation, error analysis inputs,
final-model selection, and saves:

- `models/preprocessing_pipeline.joblib`
- `models/final_model.joblib`
- `models/model_card.json`
- `reports/metrics/model_comparison.json`

## 18. Running Streamlit

```bash
streamlit run app/app.py
```

If no trained model exists yet, the app explains exactly what to run
instead of showing a fake prediction.

## 19. Example Usage

```python
from src.predict import predict_transaction, load_trained_pipeline

pipeline = load_trained_pipeline()
result = predict_transaction(
    {
        "amount": 120.0,
        "time": "2026-01-15 14:32:00",
        "frequency": 2,
        "merchant_category": "electronics",
        "account_history": 95.0,
    },
    pipeline=pipeline,
)
print(result)  # {"prediction": "Routine" | "Anomalous", "probability": <float or None>}
```

## 20. Results

`[GENERATED AFTER TRAINING]` — accuracy, precision, recall, F1, ROC-AUC,
and confusion matrices per model will be written to
`reports/metrics/model_comparison.json` and summarized here once training
has run on the approved dataset. **No results are fabricated in this
repository.**

## 21. Limitations

- Educational prototype only — not audited or production-grade.
- Dataset not finalized at the time of writing this README.
- If a synthetic dataset (e.g. PaySim) is used, results describe simulated,
  not real, transaction behavior.
- Engineered features involving `account_history` carry a documented
  leakage risk that must be verified against the real dataset's data
  dictionary before trusting the model's performance (see
  `src/features.py`).

## 22. Future Work

- Extend to additional foundational models (e.g. Random Forest) if
  justified by error analysis.
- Explore cost-sensitive thresholding once real-world false-positive/
  false-negative costs are known.
- Add SHAP-based local explanations if permitted by the capstone scope
  (currently out of scope — see paper §15/§16).

## 23. Disclaimer

This is a university capstone educational prototype. It must not be used
for real transaction monitoring or financial decision-making.

## 24. References

- E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, "PaySim: A financial
  mobile money simulator for fraud detection," in *Proc. 28th European
  Modeling and Simulation Symposium (EMSS)*, Larnaca, Cyprus, 2016.
- Pedregosa, F. et al., "Scikit-learn: Machine Learning in Python,"
  *Journal of Machine Learning Research*, 12, 2011.
- Streamlit documentation: https://docs.streamlit.io
- Additional references to be added as they are actually used — see
  `paper/technical_paper.md` §18.

## 25. Author

Name: Muhammad (Abrar-ul-hasnain)