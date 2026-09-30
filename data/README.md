# Dataset Documentation

## Current status

**No dataset is bundled with this repository.** The `src/` code is written
to be dataset-agnostic (see `src/config.py -> FEATURE_CONFIG`) so that it
can be pointed at whichever transaction dataset is ultimately approved for
this capstone. Until a real CSV is placed at `data/raw/transactions.csv`
(or the path is changed in `src/config.py`), `python -m src.train` will
raise a clear `DatasetNotFoundError` rather than fabricating results.

## Recommended free/open dataset (pending your confirmation)

If you have not already been assigned a specific dataset, the following is
a reasonable, well-documented, free option that fits the "payment
transaction anomaly classification" brief:

- **Name:** Synthetic Financial Datasets For Fraud Detection (PaySim)
- **Official source:** Kaggle — `https://www.kaggle.com/datasets/ealaxi/paysim1`
- **Underlying simulator:** PaySim, developed by Edgar Lopez-Rojas, Ahmad
  Elmir and Stefan Axelsson. Citation: E. A. Lopez-Rojas, A. Elmir, and
  S. Axelsson, "PaySim: A financial mobile money simulator for fraud
  detection," in *Proceedings of the 28th European Modeling and Simulation
  Symposium (EMSS)*, Larnaca, Cyprus, 2016.
- **License:** Listed on the Kaggle dataset page. **Verify the exact
  license terms on the Kaggle page itself before use/redistribution** —
  this README intentionally does not assert a license string that has not
  been directly confirmed there.
- **Number of samples:** Approximately 6.3 million simulated mobile-money
  transactions over a 30-day simulation (744 hourly "steps").
- **Feature description (raw columns):**
  - `step` — one step = one simulated hour (time feature)
  - `type` — transaction type: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER
    (merchant/category-style feature)
  - `amount` — transaction amount in local currency
  - `nameOrig` — originating customer ID
  - `oldbalanceOrg` / `newbalanceOrig` — originating account balance before/after
  - `nameDest` — destination customer/merchant ID
  - `oldbalanceDest` / `newbalanceDest` — destination account balance before/after
  - `isFraud` — target label (1 = fraudulent/anomalous, 0 = routine)
  - `isFlaggedFraud` — a rule-based flag from the original simulation (NOT
    the ML target — including it as a feature could leak information about
    a separate detection process; treat it as a potential leakage column
    and exclude it from model features)
- **Target definition:** `isFraud` — binary indicator of a fraudulent
  (anomalous) transaction.
- **Known limitations:**
  - The data is **synthetic**, generated to statistically resemble a real
    mobile-money provider's logs, not real customer transactions. This
    is explicitly why it is safe to publish and use for a student project,
    but conclusions should not be over-generalized to real banking systems.
  - Several Kaggle write-ups of this dataset note that once a transaction
    is detected as fraud in the simulation, it is effectively cancelled —
    meaning the post-transaction balance columns (`oldbalanceOrg`,
    `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) can leak
    information about the outcome and should either be excluded or used
    only to derive lagged/pre-transaction features. **This must be
    verified directly against the dataset before use** — do not assume
    this note is complete or current.
  - The dataset is heavily imbalanced (fraud is a small minority of
    transactions), which is directly relevant to Step 8 (Class Imbalance)
    of this project.
- **Citation:** See the citation above; always cite both the Kaggle
  dataset page and the original PaySim paper.

## How to map PaySim (or any other approved dataset) into this project

Edit `src/config.py`:

```python
FEATURE_CONFIG = {
    "amount": "amount",
    "time": "step",
    "frequency": "<derive or map to an available column>",
    "merchant_category": "type",
    "account_history": "oldbalanceOrg",   # verify leakage risk first!
    "target": "isFraud",
}
```

`frequency` (transaction frequency per account) is not a native PaySim
column — it would need to be engineered (e.g. a rolling count of prior
transactions per `nameOrig`, computed only from transactions strictly
before the current one to avoid leakage). This is exactly the kind of
adaptation `src/features.py` is structured to support — add a new function
there rather than hard-coding it elsewhere.

## Expected file location

Place the approved dataset CSV at:

```text
data/raw/transactions.csv
```

or update `RAW_DATA_FILENAME` in `src/config.py` to match your actual
filename.

## Privacy considerations

Whichever dataset is finally used, confirm it is either:
- fully synthetic (like PaySim), or
- properly anonymized with no way to re-identify real individuals or
  accounts,

before including it in a submitted repository. Do not commit a dataset
containing real, identifiable financial information.

## What still needs to be confirmed before training can produce real results

1. Final approved dataset (PaySim, an alternative such as the
   ULB/Worldline "Credit Card Fraud Detection" dataset, or an
   instructor-provided file).
2. Confirmation of the exact target column and what value represents
   "anomalous".
3. Confirmation of which columns are safe to use as features vs. which
   are potential leakage risks (see the balance-column note above).
4. A decision on how to construct the `frequency` and `account_history`
   features if the chosen dataset does not provide them directly.

Until these are confirmed, all model metrics, figures, and conclusions in
this repository remain placeholders (`[GENERATED AFTER TRAINING]`).
