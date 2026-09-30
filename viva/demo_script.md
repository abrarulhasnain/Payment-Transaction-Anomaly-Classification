# Final Demo Script (5–7 minutes)

Rehearse this end-to-end at least once before the real demo. Have the
dataset already placed and `python -m src.train` already run beforehand —
don't train live unless specifically asked to.

## 1. Open project (30s)
Open the repository in your editor/terminal. Briefly show the folder
structure (`README.md` §13) and mention it separates data, source code,
models, reports, app, tests, and documentation.

## 2. Explain the problem (30s)
State the problem in one sentence: classify payment transactions as
Routine or Anomalous using transaction characteristics. Mention the
FinTech domain and binary-classification framing.

## 3. Show the dataset (30s)
Open `data/README.md` and briefly state: source, size, target column, and
one known limitation. If asked "is this real data?", answer honestly per
what was actually used.

## 4. Show the notebook (45s)
Open `notebooks/01_...ipynb`. Scroll through section headers (Data
Understanding → Cleaning → EDA → Feature Engineering → Training → Error
Analysis → Interpretability) without re-running every cell live — point
out it calls into `src/` rather than duplicating logic.

## 5. Show EDA (45s)
Show 1–2 saved figures from `reports/figures/` (e.g. target distribution,
amount-by-class boxplot). State one real observation from each.

## 6. Show model comparison (45s)
Open `reports/metrics/model_comparison.json` or the notebook's comparison
table. State which three models were compared and, briefly, the actual
metric values — do not estimate numbers from memory.

## 7. Show evaluation (30s)
Point to precision/recall/F1/ROC-AUC/confusion matrix for the final model.
Explain briefly why the final model was chosen (recall/F1 first, not
accuracy alone — reference `src/train.py -> select_final_model()`).

## 8. Show error analysis (45s)
Show the TP/TN/FP/FN breakdown and, if available, the false-positive
pattern summary. This directly answers the project's research direction —
spend real time here.

## 9. Show the saved model (15s)
Point to `models/final_model.joblib` and `models/model_card.json` in the
file explorer — mention these are what the Streamlit app actually loads.

## 10. Open Streamlit (15s)
```bash
streamlit run app/app.py
```

## 11. Enter a sample transaction (30s)
Fill in a plausible amount, time, frequency, merchant category, and
account-history value in the form.

## 12. Run prediction (15s)
Click **Predict**.

## 13. Explain the output (30s)
Read the predicted label and probability aloud. Explain: "This came from
the exact same pipeline shown in the notebook — same preprocessing, same
trained model, loaded via joblib."

## 14. Explain limitations (30s)
State clearly: this is an educational prototype, not production-grade;
name one specific limitation from `README.md` §21 (e.g. dataset
limitation or the account-history leakage caveat).

## 15. Conclusion (20s)
One sentence: "This project demonstrates a complete, reproducible,
leakage-aware ML workflow for transaction anomaly classification, with
every number shown actually generated from the trained pipeline — nothing
was fabricated."

---

**Total: ~6 minutes.** Adjust pacing live based on time remaining; steps
6, 8, and 13 are the highest-value sections if you need to cut time
elsewhere.
