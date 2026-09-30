# Viva Answers

Simple, technically correct answers a BS AI student can explain naturally.
Where an answer depends on the actual dataset/results, it is marked
`[FILL IN FROM ACTUAL PROJECT RESULTS]` — memorize the *reasoning*, not a
made-up number.

## Basic ML

**1. What is classification?**
Predicting a discrete category (label) for an input, based on patterns
learned from labeled training examples.

**2. What is binary classification?**
Classification where there are only two possible output classes — here,
Routine vs. Anomalous.

**3. What is supervised learning?**
Learning a mapping from inputs to outputs using a dataset where the
correct output (label) is already known for each training example.

**4. What is anomaly detection?**
Identifying data points that deviate significantly from the normal/
expected pattern in the data. In this project it's framed as supervised
binary classification because labeled examples of "anomalous" transactions
are assumed to be available.

## Dataset

**5. What is your target variable?**
The column configured in `src/config.py -> FEATURE_CONFIG["target"]`,
indicating whether a transaction is Routine (0) or Anomalous (1).
`[FILL IN the actual column name once the dataset is finalized]`

**6. What are your features?**
Transaction amount, time, frequency, merchant/category, and
account-history indicator, plus engineered features (hour of day, day of
week, log-transformed amount, deviation from account history, frequency
bucket).

**7. Why did you choose this dataset?**
`[FILL IN — e.g. "it is freely available, has a clear binary fraud/
anomaly label, and its transaction structure matches the assigned problem
statement."]`

**8. What problems did you find in the data?**
`[FILL IN from the actual data_loader.py report output — e.g. missing
values in column X, duplicate rows, class imbalance ratio of Y:1]`

## Preprocessing

**9. Why do we handle missing values?**
Most ML algorithms (including all three used here) cannot process NaN
values directly, and unhandled missing data can bias or break training.

**10. Why encode categorical variables?**
Models like Logistic Regression and KNN operate on numbers, not text
categories, so categories must be converted into a numeric representation
(here, one-hot encoding).

**11. Why scale features?**
Logistic Regression and KNN are sensitive to the scale of input features —
a feature measured in the thousands (like amount) could dominate one
measured in single digits (like frequency) if left unscaled. Scaling puts
all numeric features on comparable ranges.

**12. What is data leakage?**
When information that would not actually be available at prediction time
(including information from the test set, or from the future/target)
leaks into the training process, causing artificially inflated performance
that will not hold up in real use.

## Models

**13. How does Logistic Regression work?**
It models the probability of the positive class as a sigmoid function of a
weighted linear combination of the input features, and predicts the class
whose probability is higher (using a threshold, typically 0.5).

**14. Why use Logistic Regression?**
It's simple, fast, and interpretable — its coefficients directly show how
each feature pushes the prediction toward "anomalous" or "routine" — which
makes it a strong, explainable baseline.

**15. How does KNN work?**
For a new transaction, it looks at the "k" most similar transactions
(by distance in feature space) in the training data and predicts the
majority class among those neighbors.

**16. How does Decision Tree work?**
It splits the data repeatedly on feature thresholds that best separate the
classes (e.g. by Gini impurity or entropy), forming a tree of if/else
rules that leads to a final class prediction at each leaf.

**17. Why compare multiple models?**
No single algorithm is guaranteed to perform best on a given dataset;
comparing several under identical conditions provides evidence for which
approach actually fits the data, rather than assuming one in advance.

## Evaluation

**18. What is accuracy?**
The proportion of all predictions that were correct: (TP + TN) / total.

**19. What is precision?**
Of all transactions predicted as anomalous, the proportion that actually
were anomalous: TP / (TP + FP).

**20. What is recall?**
Of all transactions that were actually anomalous, the proportion the model
correctly caught: TP / (TP + FN).

**21. What is F1-score?**
The harmonic mean of precision and recall — a single number that balances
both, useful when you care about both false positives and false negatives.

**22. What is ROC-AUC?**
The area under the ROC curve (true positive rate vs. false positive rate
across all classification thresholds); a value close to 1.0 indicates the
model separates the two classes well across thresholds, not just at one
fixed cutoff.

**23. What is a confusion matrix?**
A table showing counts of True Positives, True Negatives, False Positives,
and False Negatives — the raw material every other classification metric
is computed from.

## Financial anomaly detection

**24. Why can accuracy be misleading?**
Under class imbalance (few anomalies among many routine transactions), a
model that always predicts "Routine" can score very high accuracy while
catching zero actual anomalies — accuracy alone hides this failure.

**25. Why are false positives important?**
A false positive is a routine transaction wrongly flagged as anomalous.
In a real system this wastes investigator time and can inconvenience
legitimate customers, so a high false-positive rate makes a system
impractical even if it "looks accurate."

**26. Why are false negatives important?**
A false negative is an actual anomaly the model missed — in a fraud
context, this is usually the most costly type of error since the
anomalous activity goes undetected.

**27. What is class imbalance?**
When one class (usually the anomalous/positive class) makes up a small
fraction of the total data compared to the other class — very common in
fraud/anomaly problems.

**28. How did you handle class imbalance?**
Primarily through the model's `class_weight="balanced"` option (tuned via
GridSearchCV alongside other hyperparameters — see
`config.PARAM_GRIDS`), and by choosing recall/F1 on the anomalous class,
rather than accuracy, as the primary metric for model selection and
tuning.

## Feature engineering

**29. Which features did you create?**
`hour_of_day`, `day_of_week`, `amount_log`, `amount_deviation_from_history`,
and `frequency_bucket` — see `src/features.py` for exact formulas.

**30. Why did you create them?**
To surface patterns (time-of-day effects, skew correction, deviation from
personal spending baseline, frequency grouping) that could help the models
separate anomalous from routine transactions more effectively than the
raw features alone.

**31. Could your features cause leakage?**
`amount_deviation_from_history` carries a documented conditional leakage
risk if the underlying `account_history` value in the real dataset is not
strictly computed from transactions before the current one — this is
explicitly flagged in `src/features.py` and must be verified before the
feature is trusted.

## Deployment

**32. How does Streamlit work?**
It's a Python framework that turns a script with special UI calls
(`st.number_input`, `st.button`, etc.) into an interactive web app,
re-running the script top-to-bottom on each interaction.

**33. How does the app use the trained model?**
It loads the saved `final_model.joblib` pipeline once
(`@st.cache_resource`) and calls the same `predict_transaction()` function
used by the automated tests — there is no separate prediction logic in the
app itself.

**34. Why save the preprocessing pipeline?**
So that the exact transformations fit on the training data (imputation
values, scaling parameters, one-hot categories) are reused identically at
prediction time — refitting preprocessing on new data would break
consistency and could silently corrupt predictions.

**35. How do you ensure training and inference preprocessing are identical?**
By saving one combined `sklearn.Pipeline` object (preprocessor + model)
with `joblib` rather than saving preprocessing code and model weights
separately — loading the pipeline back guarantees identical behavior.

## Research

**36. What is your research direction?**
Investigating behavioral patterns in transaction anomalies and, in
particular, studying classification errors — especially false positives —
to understand where and why the model gets things wrong.

**37. What behavioral patterns did you investigate?**
`[FILL IN from the actual false_positive_pattern_summary() output — e.g.
whether false positives cluster in a particular merchant category or time
window]`

**38. What were your major findings?**
`[FILL IN from actual results — do not state a finding before the
experiment has been run]`

## Limitations

**39. What are the limitations of your project?**
Educational-prototype scope, dependence on the final dataset's quality and
representativeness, a documented conditional leakage risk in one
engineered feature, and the absence of deep-learning approaches (out of
scope for this capstone).

**40. Can this system be used directly by a bank?**
No — it has not been audited, tested at production scale, or validated
against real regulatory/operational requirements. It's a demonstration of
the ML workflow, not a deployable fraud system.

**41. What would you improve with more time?**
`[FILL IN — reasonable answers: more models e.g. Random Forest, deeper
error analysis, cost-sensitive thresholding, confirmed leakage-safe
account-history feature]`

## Extra

**42. Why did you use GridSearchCV instead of manually trying parameters?**
It systematically and reproducibly searches a defined hyperparameter grid
using cross-validation, avoiding both manual guesswork and accidental
overfitting to a single validation split.

**43. Why is your test set never used to fit the preprocessing pipeline?**
Fitting on the test set (e.g. computing scaling means/variances or
one-hot categories from it) would let information from the test set
influence training, inflating evaluation metrics and misrepresenting how
the model would perform on truly unseen data.

**44. What would happen if you scaled the full dataset before splitting it?**
The scaler would learn statistics (mean/variance) that include test-set
values, so the "unseen" test set would no longer be fully independent of
training — a subtle form of data leakage that can make test performance
look better than it really would be in production.

**45. Why didn't you use a neural network for this project?**
The capstone scope explicitly calls for foundational models
(Logistic Regression, KNN, Decision Tree); with a 10-day timeframe and a
tabular dataset of this size, foundational models are typically
sufficient, more interpretable, and easier to defend in a viva than an
unexplained deep-learning black box.
