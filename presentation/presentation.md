# Presentation: Payment Transaction Anomaly Classification

Format: ~12–15 slides. No numerical results are fabricated below — slides
that depend on training output are marked `[GENERATED AFTER TRAINING]` and
must be filled in from `reports/metrics/model_comparison.json` and the
notebook once the real dataset has been used.

---

### Slide 1 — Title
**Payment Transaction Anomaly Classification**
BS AI Capstone — Problem 26 (FinTech, Binary Classification)
Muhammad (Abrar-ul-hasnain), Air University, Islamabad

*Speaker notes:* Introduce yourself, the course, and the 10-day capstone
format. State up front that this is an educational prototype.

---

### Slide 2 — Problem Statement
- Classify payment transactions as **Routine** or **Anomalous**
- Binary classification using transaction characteristics
- Domain: FinTech

*Speaker notes:* Read the official problem statement verbatim, then
restate it in your own words to show understanding.

---

### Slide 3 — Motivation
- Manual transaction review does not scale
- Automated screening can help prioritize which transactions deserve a
  closer look
- Educational value: demonstrates a full supervised-classification
  workflow on a realistic, imbalanced FinTech problem

*Speaker notes:* Be careful not to overstate real-world impact — this is a
prototype, not a deployed system.

---

### Slide 4 — Objective
- Build a reproducible, end-to-end pipeline: data → EDA → preprocessing →
  features → models → evaluation → error analysis → final model →
  Streamlit app
- No fabricated results — every reported number comes from an actual run

*Speaker notes:* Emphasize reproducibility as a design goal, not just an
outcome.

---

### Slide 5 — Dataset
`[GENERATED AFTER DATASET IS FINALIZED]`
- Name / source / license
- Number of samples, feature summary, target definition
- Known limitations

*Speaker notes:* If using PaySim, mention it is synthetic and explain why
that is actually a strength (safe to publish/share) as well as a
limitation (not real transactions).

---

### Slide 6 — Data Preprocessing
- Duplicate rows dropped before splitting
- Missing values: median (numeric) / most-frequent (categorical)
  imputation
- Scaling: StandardScaler; Encoding: OneHotEncoder(handle_unknown="ignore")
- Outliers reported, not blindly deleted

*Speaker notes:* Be ready to explain *why* each choice was made (see
`paper/technical_paper.md` §6).

---

### Slide 7 — EDA
`[GENERATED AFTER TRAINING]`
- Target class distribution (imbalance ratio)
- Transaction amount distribution, overall and by class
- Correlation heatmap

*Speaker notes:* Show 2–3 of the actual saved figures from
`reports/figures/`. State one observation and one implication per figure.

---

### Slide 8 — Feature Engineering
- `hour_of_day`, `day_of_week` — time-based patterns
- `amount_log` — reduces right-skew
- `amount_deviation_from_history` — deviation from typical spend
  (leakage-risk flagged, verify before trusting)
- `frequency_bucket` — low/medium/high frequency grouping

*Speaker notes:* Proactively mention the leakage caveat on
`amount_deviation_from_history` — this shows methodological maturity.

---

### Slide 9 — Models
- Logistic Regression (interpretable baseline)
- K-Nearest Neighbors (non-parametric comparison)
- Decision Tree (non-linear, interpretable)
- All three share the same preprocessing pipeline for a fair comparison

*Speaker notes:* Explain briefly, in your own words, how each algorithm
works (see `viva/viva_answers.md`).

---

### Slide 10 — Evaluation Metrics
- Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix
- Why accuracy alone is misleading under class imbalance

*Speaker notes:* Use a concrete illustrative example: a model that always
predicts "Routine" can have high accuracy but zero recall on anomalies.

---

### Slide 11 — Model Comparison
`[GENERATED AFTER TRAINING]`
- Side-by-side metrics table for all three models
- Training time per model

*Speaker notes:* Pull the table directly from
`reports/metrics/model_comparison.json` — do not estimate numbers from
memory.

---

### Slide 12 — Error Analysis
`[GENERATED AFTER TRAINING]`
- TP / TN / FP / FN counts and rates
- Patterns found among false positives (e.g. by merchant category or time)
- Why false positives matter in transaction monitoring

*Speaker notes:* This is the most research-oriented slide — spend extra
prep time here since it directly answers the project's research direction.

---

### Slide 13 — Streamlit Application
- Live demo: enter a transaction, get a prediction + probability
- Input validation and clear "educational prototype" disclaimer
- Uses the exact trained pipeline — no separate/duplicated logic

*Speaker notes:* Have the app already running locally before this slide.

---

### Slide 14 — Limitations & Future Scope
- Educational prototype, not production-ready
- Dataset limitations (synthetic or otherwise)
- Future: Random Forest comparison, cost-sensitive thresholds, confirmed
  leakage-safe account-history features

*Speaker notes:* Showing awareness of limitations is itself a strong
signal in a viva — don't rush this slide.

---

### Slide 15 — Conclusion
`[GENERATED AFTER TRAINING]`
- Restate the final model and why it was selected
- One-sentence takeaway on what the project demonstrates

*Speaker notes:* End on the reproducibility and honesty of the workflow —
that every number shown was actually generated, not assumed.
