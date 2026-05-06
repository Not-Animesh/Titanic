# 🔬 Step-by-Step Walkthrough — Titanic ML Lab

This document explains **every step** of the codebase, from loading raw data all the way to displaying predictions and insights in the browser.

---

## Table of Contents

1. [High-Level Flow](#1-high-level-flow)
2. [Step 1 — Data Loading (`data_loader.py`)](#step-1--data-loading)
3. [Step 2 — Preprocessing (`preprocessing.py`)](#step-2--preprocessing)
4. [Step 3 — Feature Engineering (`feature_engineering.py`)](#step-3--feature-engineering)
5. [Step 4 — Feature Selection (`feature_selector.py`)](#step-4--feature-selection)
6. [Step 5 — Train / Test Split](#step-5--train--test-split)
7. [Step 6 — Model Instantiation (`model_factory.py`)](#step-6--model-instantiation)
8. [Step 7 — Training (`trainer.py`)](#step-7--training)
9. [Step 8 — Evaluation (`evaluator.py`)](#step-8--evaluation)
10. [Step 9 — SHAP Explanation (`explainer.py`)](#step-9--shap-explanation)
11. [Step 10 — Insights (`insights.py`)](#step-10--insights)
12. [Step 11 — Pipeline Orchestration (`pipeline.py`)](#step-11--pipeline-orchestration)
13. [Step 12 — Streamlit UI (`app.py`)](#step-12--streamlit-ui)
14. [Step 13 — Survival Prediction Form](#step-13--survival-prediction-form)

---

## 1. High-Level Flow

```
User opens the browser
        │
        ▼
  Streamlit app.py starts
        │
  User clicks "Run Pipeline"
        │
        ▼
  pipeline.run_pipeline()
  ┌───────────────────────────────────────────┐
  │  1. data_loader  → load raw CSV           │
  │  2. preprocessing → clean & encode        │
  │  3. feature_engineering → create features │
  │  4. feature_selector → drop features      │
  │  5. split_data → train/test split         │
  │  6. model_factory → get model             │
  │  7. trainer → fit model                   │
  │  8. evaluator → compute metrics           │
  │  9. explainer → SHAP values               │
  │ 10. insights → human-readable summaries   │
  └───────────────────────────────────────────┘
        │
        ▼
  Results rendered in Streamlit
  (metrics, confusion matrix, SHAP charts, insights)
```

---

## Step 1 — Data Loading

**File:** `ml_engine/data_loader.py`

### What happens

```python
raw_df = load_titanic()
```

1. Tries to open `titanic.csv` (or `ml_engine/train.csv`) from disk.
2. If the file is not found, downloads the dataset from **seaborn's built-in datasets** and renames the columns to PascalCase (`survived` → `Survived`, `pclass` → `Pclass`, …) so that the rest of the pipeline always sees consistent column names.
3. `split_data(df, target="Survived")` performs a stratified 80/20 train-test split using `sklearn.model_selection.train_test_split` with `random_state=42`, ensuring reproducible splits and balanced class distribution in both sets.

### Why this matters

Having a fixed random seed (`random_state=42`) means every run with the same data and model produces identical train/test splits, making experiments reproducible and fair to compare.

---

## Step 2 — Preprocessing

**File:** `ml_engine/preprocessing.py`

### What happens — three sub-steps

#### 2a. Handle Missing Values (`handle_missing_values`)

| Column | Strategy | Reason |
|---|---|---|
| `Age` | Fill with **median** | Robust to outliers; ~20 % of values are missing |
| `Embarked` | Fill with **mode** (most common port) | Only 2 values missing |
| `Fare` | Fill with **median** | Handles rare missing values robustly |
| `Cabin` | **Drop the column** | >77 % missing — too sparse to be useful |

#### 2b. Encode Categorical Variables (`encode_categoricals`)

| Column | Encoding | Result |
|---|---|---|
| `Sex` | Label encoding | `male → 1`, `female → 0` |
| `Embarked` | One-hot encoding (`drop_first=True`) | Creates `Embarked_Q`, `Embarked_S` (C is the reference) |

`drop_first=True` avoids the **dummy variable trap** (perfect multicollinearity).

#### 2c. Drop Unused Columns (`drop_unused_columns`)

`Name` and `Ticket` are dropped because they are high-cardinality text fields with no direct predictive signal (the title inside `Name` is extracted in the next step).

### Code flow

```python
df = handle_missing_values(df)   # imputation
df = encode_categoricals(df)     # Sex + Embarked
df = drop_unused_columns(df)     # Name, Ticket
```

---

## Step 3 — Feature Engineering

**File:** `ml_engine/feature_engineering.py`

New features are derived from existing ones to give models stronger signals.

### 3a. `FamilySize`

```python
FamilySize = SibSp + Parch + 1
```

Counts the total number of people travelling with the passenger (siblings, spouses, parents, children) plus the passenger themselves. Research shows travelling alone vs. in a large group significantly affects survival odds.

### 3b. `IsAlone`

```python
IsAlone = 1 if FamilySize == 1 else 0
```

A binary flag derived from `FamilySize`. Passengers travelling alone had lower survival rates than those in small families (but very large families also fared poorly).

### 3c. `Title_*` (from passenger `Name`)

The raw `Name` field contains titles like "Mr.", "Mrs.", "Miss.", "Master.", "Dr.", etc.

```
"Braund, Mr. Owen Harris"  →  "Mr"
"Futrelle, Mrs. Jacques Heath"  →  "Mrs"
"Palsson, Master. Gosta Leonard"  →  "Master"
"Capt. Edward Gifford Crosby"  →  "Rare"
```

Titles are mapped to: `Mr`, `Miss`, `Mrs`, `Master`, or `Rare` (for all uncommon titles). These are then one-hot encoded into `Title_Mr`, `Title_Miss`, `Title_Mrs`, `Title_Master`, `Title_Rare`.

**Why:** Title is a proxy for age, gender, and social class simultaneously — all strong survival predictors.

### 3d. `AgeBin_*`

Age is binned into three groups:

| Bin | Condition | Column |
|---|---|---|
| child | Age < 16 | `AgeBin_child` |
| adult | 16 ≤ Age ≤ 60 | `AgeBin_adult` |
| senior | Age > 60 | `AgeBin_senior` |

The continuous `Age` column is **retained** alongside the bins so models can use both.

---

## Step 4 — Feature Selection

**File:** `ml_engine/feature_selector.py`

### What happens

The user selects features to drop via the Streamlit sidebar. `drop_features()` enforces two safety rules:

1. **The target column (`Survived`) is never dropped**, regardless of what the user selects.
2. **Columns that do not exist in the DataFrame are silently ignored** — preventing crashes if column names change.

```python
reduced_df = drop_features(engineered_df, drop_list=drop_feature_list)
```

`get_droppable_features()` returns a sorted list of all non-target columns, which is used to populate the multiselect widget in the UI.

---

## Step 5 — Train / Test Split

**Called inside:** `ml_engine/data_loader.split_data`

```python
X_train, X_test, y_train, y_test = split_data(reduced_df, target="Survived")
```

- **80 % training**, **20 % testing**.
- `stratify=y` ensures both splits contain roughly the same proportion of survivors (~38 %).
- `random_state=42` for reproducibility.

---

## Step 6 — Model Instantiation

**File:** `ml_engine/model_factory.py`

`get_model(model_name)` takes a string and returns a pre-configured, unfitted sklearn-compatible estimator:

| String passed | Returned object |
|---|---|
| `"Logistic Regression"` | `LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')` |
| `"Random Forest"` | `RandomForestClassifier(n_estimators=200, max_depth=8)` |
| `"XGBoost"` | `XGBClassifier(n_estimators=200, max_depth=5, lr=0.05)` |
| `"CatBoost"` | `CatBoostClassifier(iterations=200, depth=6, lr=0.05)` |

If XGBoost or CatBoost is not installed, a helpful `ImportError` with an install command is raised.

---

## Step 7 — Training

**File:** `ml_engine/trainer.py`

```python
trained_model, train_time = train_model(model, X_train, y_train)
```

1. Validates that `X_train` and `y_train` are non-empty.
2. Ensures all column names are strings (required by XGBoost and CatBoost).
3. Records the start time with `time.perf_counter()` (high-resolution wall-clock).
4. Calls `model.fit(X_train, y_train)`.
5. Returns the **fitted model** and the **elapsed time in seconds** (rounded to 4 decimal places).

---

## Step 8 — Evaluation

**File:** `ml_engine/evaluator.py`

```python
metrics = evaluate_model(trained_model, X_test, y_test)
```

Computes the following using sklearn:

| Metric | What it measures |
|---|---|
| **Accuracy** | `(TP + TN) / total` — overall correct predictions |
| **Precision** | `TP / (TP + FP)` — of those predicted to survive, how many actually did? |
| **Recall** | `TP / (TP + FN)` — of those who actually survived, how many did we catch? |
| **F1 Score** | `2 × (Precision × Recall) / (Precision + Recall)` — harmonic mean |
| **Confusion Matrix** | 2×2 matrix: [[TN, FP], [FN, TP]] |

`zero_division=0` is passed to precision/recall/F1 to avoid divide-by-zero warnings if one class is never predicted.

---

## Step 9 — SHAP Explanation

**File:** `ml_engine/explainer.py`

SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for each prediction.

### Which explainer is used?

```
Is model a tree-based model?
    ├── YES → shap.TreeExplainer   (fast, exact)
    └── NO  → shap.KernelExplainer (slower, model-agnostic)
```

Tree-based models: `RandomForestClassifier`, `XGBClassifier`, `CatBoostClassifier`, `GradientBoostingClassifier`, `DecisionTreeClassifier`, `ExtraTreesClassifier`.

For `KernelExplainer`, a k-means summary of the training set (≤ 50 representative points) is used as the background distribution to keep latency acceptable.

### What is returned?

- **`shap_values`** — NumPy array of shape `(n_test_samples, n_features)`. Positive values push the prediction toward "survived"; negative values push toward "did not survive".
- **`feature_importance`** — DataFrame with columns `['feature', 'importance']`, where importance = mean absolute SHAP value across all test samples, sorted descending.

---

## Step 10 — Insights

**File:** `ml_engine/insights.py`

`generate_insights(current_run, history)` generates up to **5 plain-English bullet points**:

| Insight | Condition |
|---|---|
| Overall accuracy snapshot | Always generated |
| Impact of dropping features | Compares against an earlier all-features run of the same model if available |
| Top influential feature | Taken from SHAP feature importance |
| Precision vs. recall balance | Flags if the gap is > 0.10 |
| Best run so far | Shown when ≥ 2 experiments have been run |

---

## Step 11 — Pipeline Orchestration

**File:** `ml_engine/pipeline.py`

`run_pipeline()` is the single public function that wires together all the steps above in order. It accepts:

- `drop_feature_list` — features to exclude.
- `model_name` — which algorithm to use.
- `history` (optional) — previous experiment results for insights comparison.

It returns a dictionary containing the model, metrics, SHAP values, feature importance, insights, training time, and test data.

---

## Step 12 — Streamlit UI

**File:** `app.py`

The Streamlit app renders seven main sections:

### 12a. Page Config & Session State

```python
st.set_page_config(page_title="Titanic ML Lab & Predictor", page_icon="🚢", layout="wide")
```

Session state keys are initialised once at startup:
- `history` — lightweight list of metrics per run.
- `last_result` — full result dict of the most recent run (model, SHAP values, X_test).
- `splash_done` — one-shot guard for the loading animation.

### 12b. Data Loading (Cached)

Two separate cached loaders are used:

| Loader | Purpose |
|---|---|
| `_load_raw()` | Used by the ML pipeline — may strip non-ML columns |
| `_load_raw_display()` | Used by the Dataset Explorer — retains Name, Ticket, Cabin, PassengerId |

`@st.cache_data` ensures the CSV is read from disk only once per session.

### 12c. Dataset Overview

The full DataFrame is displayed with interactive row filtering. Users can select specific columns and filter by value ranges or categories using Streamlit widgets.

### 12d. Summary Statistics

Descriptive statistics (`df.describe()`) plus a survival-rate breakdown by gender, class, and embarkation port.

### 12e. ML Experimentation Lab

1. **Sidebar:** model selector + feature multiselect.
2. **"Run Pipeline" button** → calls `run_pipeline_safe()`.
3. **Results panel:** metric bar chart, confusion matrix heatmap, SHAP feature importance chart, beeswarm plot, insights bullets.
4. **Experiment history table** listing all previous runs in the session.

### 12f. `_sanitize_X()`

A critical utility applied to training data AND prediction input to prevent train/predict mismatches:

1. Flattens MultiIndex column names (joins levels with `_`).
2. Converts all column names to strings.
3. Coerces every column to numeric (`pd.to_numeric(..., errors="coerce")`).
4. Fills remaining NaNs with `0`.

This ensures the model always sees the same data format it was trained on.

---

## Step 13 — Survival Prediction Form

The user fills in a passenger profile via Streamlit widgets:

| Field | Widget | Notes |
|---|---|---|
| Pclass | Selectbox (1 / 2 / 3) | |
| Sex | Selectbox (male / female) | |
| Age | Slider (0–80) | |
| SibSp | Number input | |
| Parch | Number input | |
| Fare | Slider (0–512) | |
| Embarked | Selectbox (C / Q / S) | |

On submit:

1. Input is assembled into a single-row DataFrame with the same column names used during training.
2. **The same preprocessing and feature-engineering steps are applied** to this one row (FamilySize, IsAlone, Title, AgeBins).
3. `_sanitize_X()` aligns the input columns exactly with those seen during training (using `reindex`, filling missing columns with 0).
4. `model.predict()` returns 0 or 1.
5. `model.predict_proba()` returns the probability of survival.
6. A celebratory animation plays if survival is predicted, or a somber message if not.

---

## Summary

Every piece of the pipeline is **modular**, **tested independently**, and **documented with docstrings**. The Streamlit layer acts purely as a presentation and interaction layer — all ML logic lives in `ml_engine/` and can be used independently of the UI.
