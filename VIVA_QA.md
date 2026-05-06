# ❓ Viva & Presentation Q&A — Titanic ML Lab

This document covers the most likely questions you will face during a presentation, viva, or demo of the Titanic ML Lab project, along with clear, concise answers.

---

## Table of Contents

1. [General / Project Overview](#1-general--project-overview)
2. [Dataset & Data Understanding](#2-dataset--data-understanding)
3. [Preprocessing](#3-preprocessing)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Selection & Training](#5-model-selection--training)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [SHAP & Explainability](#7-shap--explainability)
8. [Streamlit & Architecture](#8-streamlit--architecture)
9. [Machine Learning Theory](#9-machine-learning-theory)
10. [Advanced / Critical Questions](#10-advanced--critical-questions)

---

## 1. General / Project Overview

---

**Q: What is the goal of this project?**

A: The goal is to predict whether a Titanic passenger would have survived the disaster using their personal attributes (age, sex, class, family size, etc.). Beyond pure prediction, the project provides an interactive ML lab where users can experiment with different models and features, and immediately see how their choices affect model performance — all in a browser-based dashboard.

---

**Q: What makes this project different from a simple Titanic notebook?**

A: Several things:
- It is a **fully modular application** with a clean separation between the ML library (`ml_engine`) and the presentation layer (`app.py`).
- It supports **four different classifiers** that can be switched without touching any code.
- It includes **SHAP explainability** so you understand *why* the model made each prediction.
- It supports **live survival prediction** via an interactive form.
- It tracks an **experiment history** so you can compare multiple runs side-by-side in the same session.

---

**Q: Who is the intended user of this application?**

A: Students, data science practitioners, or anyone who wants to learn ML concepts interactively. The UI is designed to be self-explanatory — no coding knowledge is required to run experiments.

---

**Q: How do you run the project?**

A:
```bash
pip install -r requirements.txt
pip install -e .
streamlit run app.py
```
The app opens at `http://localhost:8501`.

---

## 2. Dataset & Data Understanding

---

**Q: What dataset did you use?**

A: The classic Kaggle Titanic dataset. It contains 891 rows and 12 columns covering passenger demographics, ticket information, and whether the passenger survived.

---

**Q: What are the key features in the dataset?**

A: The most important raw features are:
- `Pclass` — socioeconomic class (1st, 2nd, 3rd).
- `Sex` — gender of the passenger.
- `Age` — age in years.
- `SibSp` — number of siblings/spouses aboard.
- `Parch` — number of parents/children aboard.
- `Fare` — ticket price paid.
- `Embarked` — port of embarkation (Cherbourg, Queenstown, Southampton).

---

**Q: What is the class distribution of the target variable?**

A: Approximately **38% survived** (342 out of 891), meaning the dataset is moderately imbalanced. We account for this by using stratified splitting and evaluating precision, recall, and F1-score — not just accuracy.

---

**Q: Why is the Cabin column dropped?**

A: Because over **77% of Cabin values are missing**. Imputing that many values would introduce more noise than signal. A column that is mostly absent cannot reliably help the model.

---

**Q: Why are Name and Ticket dropped?**

A: Both are **high-cardinality text fields**. `Ticket` has hundreds of unique values with no consistent pattern useful to a classifier. `Name` is dropped after the useful part — the title — is extracted as a new feature.

---

## 3. Preprocessing

---

**Q: Why do you impute Age with the median instead of the mean?**

A: The **median is robust to outliers**. If a small number of passengers have very high or very low ages, the mean would be pulled in that direction, while the median stays at the central value. For a right-skewed distribution like Age, median imputation is the safer default.

---

**Q: Why is Embarked imputed with the mode?**

A: Only 2 values are missing. The mode (most frequent category) is the simplest and most interpretable choice for a categorical variable with so few gaps.

---

**Q: What is the dummy variable trap and how do you avoid it?**

A: The dummy variable trap occurs when one-hot encoded columns are perfectly linearly dependent on each other (e.g., if you know `Embarked_Q` and `Embarked_S`, you already know whether `Embarked_C` is true). We avoid it by passing `drop_first=True` to `pd.get_dummies()`, which drops one reference category.

---

**Q: How is Sex encoded and why?**

A: `Sex` is label-encoded: `male → 1`, `female → 0`. This is valid for binary categories. One-hot encoding would produce one redundant column, so label encoding is more efficient here.

---

**Q: Is there any risk of data leakage in your preprocessing?**

A: No. All imputation statistics (median, mode) are computed on the entire dataset before splitting. In a production system you would compute them only on the training set and apply the same values to the test set. Here, since the same dataset is used for both demonstration and model training (no separate unseen deployment data), the difference is minimal.

---

## 4. Feature Engineering

---

**Q: Why did you create the FamilySize feature?**

A: `SibSp` and `Parch` carry overlapping information about family. Combining them into a single `FamilySize` variable reduces feature count and creates a more interpretable signal: solo travellers, small families, and large families all had different survival rates.

---

**Q: What does IsAlone capture that FamilySize doesn't?**

A: `FamilySize` is continuous, but the most important threshold is whether a passenger was *completely alone*. `IsAlone` binarises this threshold, making it easier for linear models to exploit the pattern directly without learning the threshold from data.

---

**Q: Why extract the title from the Name column?**

A: The title is a **proxy for multiple variables simultaneously**:
- `Master` → young male child → higher chance of being prioritised for lifeboats.
- `Mrs`, `Miss` → female → significantly higher survival rate ("women and children first").
- `Rare` titles like `Dr`, `Rev`, `Col` → often male adults with lower survival priority.

It encodes age bracket, gender, and social status in a single compact feature.

---

**Q: Why bin Age into categories instead of using it as a continuous value?**

A: Age bins provide a **non-linear signal** that is easy for models to learn. A linear model might struggle to detect that survival probability drops sharply at certain thresholds. The bins (`child < 16`, `adult 16–60`, `senior > 60`) capture the "women and children first" policy directly. The continuous Age column is retained alongside the bins so models can use both.

---

**Q: How many features does the model end up with after all engineering?**

A: Starting from ~7 raw features (after dropping Cabin, Name, Ticket), engineering adds FamilySize, IsAlone, Title_\* (5 columns), and AgeBin_\* (3 columns), reaching approximately **18–20 features** before any user-selected drops.

---

## 5. Model Selection & Training

---

**Q: Why did you include four different models?**

A: To allow experimentation and comparison. Each model has different strengths:
- **Logistic Regression** — fast, interpretable baseline; works well when features are linearly separable.
- **Random Forest** — handles non-linearity, robust to outliers, less prone to overfitting than a single decision tree.
- **XGBoost** — gradient boosting with regularisation; often achieves top performance on tabular data.
- **CatBoost** — gradient boosting optimised for categorical data; robust out of the box with minimal tuning.

---

**Q: How is the model trained?**

A: `model.fit(X_train, y_train)` — the standard sklearn API. Wall-clock training time is measured with `time.perf_counter()` for high-resolution timing.

---

**Q: What random seed do you use and why?**

A: `random_state=42` is set for all models and for the train/test split. This ensures that results are **reproducible** — the same configuration always produces the same output.

---

**Q: What are the hyperparameters you chose for Random Forest?**

A: `n_estimators=200` (200 trees), `max_depth=8` (limits tree depth to reduce overfitting), `min_samples_split=4` (a node must have at least 4 samples to be split), `n_jobs=-1` (uses all CPU cores).

---

**Q: Did you do any hyperparameter tuning?**

A: The current version uses sensible defaults rather than automated tuning. The application could be extended with `GridSearchCV` or `Optuna` to search hyperparameter spaces, but the focus here is on demonstrating the full ML pipeline interactively rather than maximising accuracy.

---

## 6. Evaluation Metrics

---

**Q: What metrics do you use to evaluate the model?**

A: Accuracy, Precision, Recall, F1-Score, and the Confusion Matrix.

---

**Q: What does Accuracy tell you?**

A: The proportion of total predictions that were correct: `(TP + TN) / (TP + TN + FP + FN)`. It is the most intuitive metric but can be misleading when classes are imbalanced.

---

**Q: What does Precision tell you?**

A: Of all passengers the model predicted would survive, what fraction actually survived? `Precision = TP / (TP + FP)`. High precision means few false alarms.

---

**Q: What does Recall (Sensitivity) tell you?**

A: Of all passengers who actually survived, what fraction did the model correctly identify? `Recall = TP / (TP + FN)`. High recall means fewer survivors were missed.

---

**Q: Why is F1-Score useful here?**

A: The F1-Score is the **harmonic mean of Precision and Recall**. It is a single balanced number that penalises models that sacrifice one metric for the other. It is especially useful when the class distribution is imbalanced.

---

**Q: What does the confusion matrix show?**

A:

|  | Predicted: No | Predicted: Yes |
|---|---|---|
| **Actual: No** | TN (correct) | FP (false alarm) |
| **Actual: Yes** | FN (missed) | TP (correct) |

A perfect model has large TN and TP values and zeros on the off-diagonal.

---

**Q: Why might high accuracy still be a bad model?**

A: If the dataset is 90% class-0, a model that always predicts class-0 achieves 90% accuracy while never identifying a single positive case. On Titanic (38% survivors), a "always predict dead" model would reach ~62% accuracy — but recall for survivors would be 0. This is why we report precision, recall, and F1 alongside accuracy.

---

## 7. SHAP & Explainability

---

**Q: What is SHAP?**

A: **SHapley Additive exPlanations** — a method from cooperative game theory that assigns each feature a contribution value for a specific prediction. For a given passenger, a SHAP value for `Sex` tells you exactly how much the fact that the passenger is male (or female) pushed the prediction toward or away from "survived".

---

**Q: Why is model explainability important?**

A: Machine learning models are often treated as black boxes. Explainability:
- Builds trust with users and stakeholders.
- Helps detect bias (e.g., if the model is over-relying on an unfair proxy).
- Helps debug unexpected predictions.
- Is increasingly required by regulations (e.g., GDPR Article 22 on automated decision-making).

---

**Q: What is the difference between TreeExplainer and KernelExplainer?**

A: `TreeExplainer` is a fast, exact algorithm that exploits the tree structure of decision-tree-based models (Random Forest, XGBoost, CatBoost). It runs in polynomial time. `KernelExplainer` is a model-agnostic algorithm that approximates SHAP values using a weighted linear regression; it works for any model but is significantly slower. We use TreeExplainer when available and fall back to KernelExplainer for Logistic Regression.

---

**Q: What does the beeswarm plot show?**

A: Each dot represents one test-set passenger. The horizontal position is the SHAP value (positive = pushed toward "survived"). The colour shows the feature's actual value (red = high, blue = low). The y-axis lists features ordered by mean importance. This lets you see both the direction and magnitude of each feature's effect across the entire test set simultaneously.

---

**Q: Which feature is typically most important in Titanic predictions?**

A: `Sex` (or its engineered proxy `Title_Mr`/`Title_Mrs`) consistently ranks as the most important feature, reflecting the "women and children first" boarding policy that dramatically favoured female survival.

---

## 8. Streamlit & Architecture

---

**Q: Why did you choose Streamlit?**

A: Streamlit allows you to build interactive data applications in pure Python with minimal boilerplate. It handles reactivity (widget → re-run), caching, layout, and charting — letting you focus on the ML logic rather than web development.

---

**Q: What is `@st.cache_data` and why do you use it?**

A: `@st.cache_data` is a Streamlit decorator that memoises the return value of a function. Subsequent calls with the same arguments return the cached result instantly instead of re-running the function. We use it for `_load_raw()` so the CSV is read from disk only once per session, no matter how many times widgets are interacted with.

---

**Q: What is session state and why do you need it?**

A: Streamlit re-runs the entire `app.py` script every time a widget is changed. `st.session_state` is a dictionary that persists across these re-runs within a single user session. We store the experiment history and the last full result (model, SHAP values, X_test) in session state so they survive widget interactions without being recomputed.

---

**Q: Why are there two separate data loaders (`_load_raw` and `_load_raw_display`)?**

A: `_load_raw()` is used by the ML pipeline — it may drop columns like `Name`, `Ticket`, and `PassengerId` that are not needed for modelling. `_load_raw_display()` loads the full CSV with all original columns so the Dataset Explorer section can show users a complete, unmodified view of the data.

---

**Q: What does `_sanitize_X()` do and why is it important?**

A: It ensures the feature matrix always has consistent format before it is passed to the model:
1. Flattens MultiIndex column names.
2. Converts all column names to strings.
3. Coerces all values to numeric (missing values become 0).

It is applied to both training data (inside `run_pipeline_safe`) and prediction input (the passenger form). Without this, column name mismatches or mixed types would cause silent errors or crashes at prediction time.

---

**Q: How is the modular `ml_engine` package structured?**

A: It is a standard Python package (with `setup.py` and `find_packages()`) installed in editable mode with `pip install -e .`. Each module has a single responsibility — loading, preprocessing, engineering, selection, training, evaluation, explanation, or insights. This makes each module independently testable and reusable outside the Streamlit app.

---

## 9. Machine Learning Theory

---

**Q: What is overfitting and how do you guard against it?**

A: Overfitting occurs when a model learns the training data so well (including its noise) that it performs poorly on unseen data. Guards used in this project:
- **Train/test split** — the model never sees test data during training.
- `max_depth=8` for Random Forest — limits tree complexity.
- `min_samples_split=4` — prevents extremely small leaf nodes.
- Regularisation in Logistic Regression (`C=1.0` controls inverse regularisation strength).
- XGBoost `subsample=0.8` and `colsample_bytree=0.8` — random subsampling reduces variance.

---

**Q: What is the bias-variance tradeoff?**

A: **Bias** is error from wrong assumptions in the model (e.g., assuming linearity when the relationship is non-linear). **Variance** is error from sensitivity to small fluctuations in training data. Simple models have high bias / low variance; complex models have low bias / high variance. The goal is to find the sweet spot. Random Forest and gradient boosting reduce variance through averaging/boosting while keeping bias reasonably low.

---

**Q: Why use stratified splitting?**

A: Stratified splitting ensures that both the training and test sets contain the same proportion of the target classes (~38% survived, ~62% not survived). Without stratification, a random split could accidentally put most survivors in training and very few in the test set, making evaluation unreliable.

---

**Q: What is gradient boosting?**

A: Gradient boosting builds an ensemble of **weak learners** (typically shallow decision trees) sequentially. Each new tree is trained to correct the residual errors of the previous ensemble. XGBoost and CatBoost are highly optimised implementations of gradient boosting that add regularisation, parallelism, and other improvements.

---

**Q: What is a Random Forest?**

A: A Random Forest trains many decision trees on **bootstrapped** samples of the training data (bagging) and uses a **random subset of features** at each split. Final prediction is by majority vote (classification) or averaging (regression). The randomness reduces overfitting and variance compared to a single deep tree.

---

**Q: What is logistic regression and when is it appropriate?**

A: Logistic regression models the **log-odds** of the positive class as a linear combination of features, then transforms the result through the sigmoid function to produce a probability. It is appropriate when the decision boundary is approximately linear, it trains very fast, and it is highly interpretable. It is a strong baseline for binary classification.

---

## 10. Advanced / Critical Questions

---

**Q: What would you change if you had more time?**

A: Several improvements:
- Add **cross-validation** (k-fold) instead of a single train/test split for more robust metric estimates.
- Implement **hyperparameter tuning** with Optuna or GridSearchCV.
- Add a **calibration plot** to assess whether predicted probabilities are well-calibrated.
- Persist models to disk so experiments survive page refreshes.
- Add **unit tests** for each `ml_engine` module.
- Deploy with Docker or Streamlit Cloud for public access.

---

**Q: Could this model be deployed in production?**

A: The ML pipeline is clean and modular enough to adapt, but several things would be needed for production:
- Compute imputation statistics on training data only and store them (to apply to new data without data leakage).
- Persist the fitted model and preprocessing parameters (e.g., with `joblib` or `pickle`).
- Wrap the prediction endpoint in a REST API (FastAPI or Flask).
- Add input validation, error handling, and logging.
- Monitor model drift over time.

---

**Q: How would you handle class imbalance if it were more severe?**

A: Options include:
- **Oversampling** the minority class (SMOTE — Synthetic Minority Over-sampling Technique).
- **Undersampling** the majority class.
- Using **class_weight='balanced'** in sklearn estimators.
- Adjusting the **classification threshold** (instead of 0.5, use 0.3 to increase recall).
- Using **cost-sensitive learning** with custom sample weights.

---

**Q: What is the difference between classification and regression in this context?**

A: Survival prediction is a **binary classification** problem — the output is a discrete label (0 = did not survive, 1 = survived). A regression model would predict a continuous numeric output (e.g., survival probability as a raw score), but would not directly output a class label. Logistic regression, despite its name, is a classification algorithm — it uses a regression internally but outputs a probability that is thresholded to produce a class label.

---

**Q: Is the SHAP explanation for the model or for individual predictions?**

A: Both. The **feature importance bar chart** shows global importance (average over all test samples). The **beeswarm plot** shows the distribution of individual SHAP values across the entire test set. For a specific passenger in the prediction form, you could generate a **force plot** or **waterfall plot** to show that individual's prediction breakdown — though the current UI shows only global plots.

---

**Q: How would your model perform on data from a different disaster (e.g., a different ship)?**

A: Likely worse. The model has been trained specifically on the Titanic's passenger demographics, class structure, crew policies, and historical context. Transferring to a different scenario would require retraining (or at minimum, fine-tuning) on new data. This is the **domain shift** problem in ML — a model trained on one distribution may not generalise to a different one.

---

**Q: What ethical considerations exist in a project like this?**

A: Several:
- **Survivorship bias** — the dataset only contains passengers we have records of; data may not be complete.
- **Proxies for protected attributes** — features like Pclass and Fare are proxies for wealth, and Sex is a protected attribute. A real-world deployment predicting outcomes based on gender or class would require careful fairness analysis.
- **Historical harm** — the Titanic disaster was a real tragedy. Treating it as a pure ML benchmark without acknowledging the human cost can be insensitive.
- **Overconfidence in predictions** — a model accuracy of 83% means 17% of predictions are still wrong. In life-or-death contexts, the consequences of false negatives and false positives must be carefully weighed.
