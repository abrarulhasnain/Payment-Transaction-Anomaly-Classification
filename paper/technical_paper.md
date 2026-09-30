# Payment Transaction Anomaly Classification: A Capstone Study in Binary Classification for FinTech

**Author:** Muhammad (Abrar-ul-hasnain) — BS Artificial Intelligence, Air University, Islamabad
**Project duration:** 10 days
**Repository:** payment-transaction-anomaly-classification

---

# Abstract

This paper presents an end-to-end machine learning pipeline for classifying
payment transactions as routine or anomalous, developed as a 10-day BS
Artificial Intelligence capstone project. The system uses a
dataset-agnostic configuration layer, a leakage-aware preprocessing and
feature-engineering pipeline, and three foundational classification
models — Logistic Regression, K-Nearest Neighbors, and Decision Tree —
compared under identical preprocessing and cross-validation conditions.
Model selection prioritizes recall and F1-score on the anomalous class
over raw accuracy, reflecting the class-imbalanced nature of transaction
anomaly detection. `[RESULT TO BE GENERATED FROM EXPERIMENT]` — final
performance figures, the selected model, and error-analysis findings will
be inserted here once the approved dataset has been finalized and the
training pipeline (`src/train.py`) has been executed. No results in this
paper are fabricated in advance of that experiment.

# 1. Introduction

Payment transaction anomaly detection is a foundational problem in
financial technology, sitting between fraud detection and general
behavioral-pattern monitoring. This project treats the task as a
supervised binary classification problem — distinguishing "routine" from
"anomalous" transactions using features available at the time a
transaction is submitted. The project is explicitly scoped as an
**educational prototype**: it is designed to demonstrate a complete,
defensible ML workflow rather than to serve as a production
fraud-detection system.

# 2. Problem Definition

Given a transaction record described by attributes such as amount, time,
frequency, merchant/category, and account-history indicators, the task is
to predict a binary label: **Routine** or **Anomalous**. This is framed as
supervised binary classification, trained on historical labeled
transactions and evaluated on a held-out test split.

# 3. Research Direction

Beyond raw predictive performance, this project specifically investigates:

1. **Behavioral patterns** associated with anomalous transactions (e.g.
   whether anomalies cluster by merchant category, time of day, or
   deviation from an account's historical spending).
2. **Classification errors, especially false positives** — routine
   transactions incorrectly flagged as anomalous — since these directly
   affect the practical usability of any transaction-monitoring system.

# 4. Related Work

Transaction anomaly and fraud detection is a widely studied FinTech
problem. Foundational classifiers such as Logistic Regression and Decision
Trees remain common baselines in the literature due to their
interpretability, while imbalance-aware evaluation (precision, recall, F1,
ROC-AUC) is standard practice given that anomalous transactions are
typically a small minority of all transactions. This project follows that
established baseline-first, imbalance-aware methodology rather than
introducing novel modeling techniques, consistent with the capstone's
10-day, foundational-models scope. `[Additional related work to be added
if/when specific papers are reviewed and cited — no references are
included here that have not actually been consulted.]`

# 5. Dataset

`[GENERATED AFTER DATASET IS FINALIZED]` — This section will describe: the
exact dataset used, its source and license, number of samples, feature
definitions, target definition, and known limitations. A candidate dataset
(PaySim, a synthetic mobile-money transaction simulator) is documented in
`data/README.md` pending confirmation. See that file for full source,
citation, and limitation details already gathered.

# 6. Data Preprocessing

Implemented in `src/preprocessing.py`. Key decisions:

- **Duplicate rows** are dropped before the train/test split (so the same
  physical transaction cannot appear in both splits).
- **Missing numerical values** are imputed with the median (robust to the
  right-skewed distributions typical of transaction amounts).
- **Missing categorical values** are imputed with the most frequent
  category, preserving a real observed value rather than inventing one.
- **Numerical features** are standardized (`StandardScaler`) since
  Logistic Regression and KNN are scale-sensitive.
- **Categorical features** are one-hot encoded with `handle_unknown="ignore"`
  so that a category unseen during training does not crash inference.
- **Outliers are not blindly removed** — extreme amounts may be exactly
  the anomalies the model is trying to detect; outlier counts are reported
  for awareness (`src/data_loader.py -> outlier_report_iqr`) rather than
  used to silently delete rows.

# 7. Exploratory Data Analysis

`[GENERATED AFTER TRAINING]` — target distribution, transaction-amount
distribution (overall and by class), and a correlation heatmap are
produced in `notebooks/01_payment_transaction_anomaly_analysis.ipynb`.
Observations and implications will be filled in from the actual plots
once the dataset is available; this paper does not pre-write
interpretations of data that has not been examined.

# 8. Feature Engineering

Implemented in `src/features.py`, with every feature documented by
formula, purpose, prediction-time availability, and leakage risk:

- `hour_of_day`, `day_of_week` — derived from the transaction timestamp;
  no leakage risk (deterministic function of the transaction's own time).
- `amount_log` — log1p transform of the amount, addressing right-skew;
  no leakage risk.
- `amount_deviation_from_history` — relative deviation of the current
  amount from the account-history feature; **conditional leakage risk** if
  `account_history` in the real dataset is not strictly computed from
  transactions prior to the current one. This must be verified before
  the feature is trusted.
- `frequency_bucket` — fixed-threshold bucketing of transaction frequency;
  low leakage risk provided bin edges remain fixed constants rather than
  being fit on the evaluation split.

# 9. Methodology

```text
Raw Data → Duplicate Removal → Train/Test Split (stratified, before fitting
anything) → Feature Engineering → Preprocessing (fit on train only) →
Model Training (3 baselines) → Cross-Validated Hyperparameter Tuning →
Test-Set Evaluation → Error Analysis → Final Model Selection → Pipeline
Persistence → Streamlit Inference
```

Random state is fixed (`config.RANDOM_STATE = 42`) throughout for
reproducibility. The train/test split is stratified on the target to
preserve class balance in both splits given the expected class imbalance.

# 10. Model Development

Three foundational models are trained inside one shared
`sklearn.Pipeline` (preprocessor + estimator) each, so preprocessing can
never differ between training and inference for a given model:

- **Logistic Regression** — linear baseline; interpretable via
  coefficients (`src/evaluate.py -> logistic_regression_coefficients`).
- **K-Nearest Neighbors** — non-parametric, distance-based comparison.
- **Decision Tree** — non-linear, interpretable via feature importances
  (`src/evaluate.py -> tree_feature_importances`).

Hyperparameters are tuned with `GridSearchCV` (5-fold cross-validation,
scored on F1 for the anomalous class — see `config.PARAM_GRIDS` and
`config.TUNING_SCORING`), not on the held-out test set, to avoid
test-set tuning leakage.

# 11. Model Evaluation

`[RESULT TO BE GENERATED FROM EXPERIMENT]` — Accuracy, Precision, Recall,
F1-score, ROC-AUC, and confusion matrices for all three models will be
reported here from `reports/metrics/model_comparison.json` once training
has been run on the approved dataset.

# 12. Error Analysis

Implemented in `src/evaluate.py`: true/false positive/negative counts and
rates (`error_breakdown`), a per-row error-type-labeled dataframe
(`error_case_dataframe`), and a false-positive-vs-true-negative pattern
comparison across categorical features (`false_positive_pattern_summary`).
`[RESULT TO BE GENERATED FROM EXPERIMENT]` — specific patterns found among
false positives (e.g. concentration in a particular merchant category or
time window) will be reported here once computed on real predictions.

# 13. Results

`[RESULT TO BE GENERATED FROM EXPERIMENT]`

# 14. Discussion

`[GENERATED AFTER TRAINING]` — a balanced discussion of model trade-offs
(recall vs. precision, interpretability vs. performance) will be written
here based on the actual comparison table, following the selection
reasoning implemented in `src/train.py -> select_final_model()`.

# 15. Limitations

- This project is an educational prototype and has not been audited for
  production or regulatory use.
- The dataset was not finalized at the time this template was written;
  if a synthetic dataset (e.g. PaySim) is ultimately used, findings
  describe simulated rather than real-world transaction behavior.
- Feature importance/coefficients reflect statistical association within
  the trained model, not causal influence (see
  `src/evaluate.py -> interpretation_note`).
- No deep learning models were used, per the capstone's foundational-model
  scope; this may limit performance ceiling relative to more complex
  approaches, which is an accepted trade-off for interpretability and
  the 10-day timeframe.

# 16. Future Scope

- Extend model comparison with Random Forest if error analysis motivates
  it.
- Investigate cost-sensitive threshold tuning once real false-positive /
  false-negative costs are defined by the problem owner.
- Revisit `account_history`-derived features with a confirmed,
  leakage-safe definition once the final dataset's data dictionary is
  available.

# 17. Conclusion

`[GENERATED AFTER TRAINING]` — a concise summary of the final model, its
performance, and its suitability as an educational demonstration of
transaction anomaly classification will be written here once results
exist.

# 18. References

1. E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, "PaySim: A financial
   mobile money simulator for fraud detection," in *Proc. 28th European
   Modeling and Simulation Symposium (EMSS)*, Larnaca, Cyprus, 2016.
2. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python,"
   *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.
3. Streamlit Inc., Streamlit documentation, https://docs.streamlit.io
   (accessed 2026).
4. *[Additional references to be added only as they are actually
   consulted during the real dataset's analysis — no placeholder
   citations are invented here.]*
