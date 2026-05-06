# 🚢 Titanic ML Lab & Survival Predictor

An interactive machine-learning web application built with **Streamlit** that lets you explore the Titanic dataset, run classification experiments with multiple models, and predict whether a passenger would have survived — all with live SHAP-powered explainability.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [Installation & Setup](#-installation--setup)
6. [Running the App](#-running-the-app)
7. [ML Pipeline](#-ml-pipeline)
8. [Supported Models](#-supported-models)
9. [Feature Engineering](#-feature-engineering)
10. [Model Explainability (SHAP)](#-model-explainability-shap)
11. [Dataset](#-dataset)
12. [Screenshots & Sections](#-screenshots--sections)
13. [Contributing](#-contributing)

---

## 🔍 Project Overview

The **Titanic ML Lab** turns the classic Titanic survival prediction problem into an end-to-end interactive data-science playground. Users can:

- Browse and filter the raw Titanic dataset.
- Choose which features to include or exclude before training.
- Select a classification algorithm (Logistic Regression, Random Forest, XGBoost, or CatBoost).
- View metrics (Accuracy, Precision, Recall, F1) and a confusion matrix in real time.
- Inspect SHAP feature-importance charts and beeswarm plots.
- Fill a passenger form and get an instant survival prediction with probability.
- Track multiple experiment runs and compare them side-by-side.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 Dataset Explorer | Browse all columns, apply row-level filters, and view summary statistics |
| 🧪 ML Experiment Lab | Choose model + features, train instantly, compare across runs |
| 📈 Rich Metrics | Accuracy, Precision, Recall, F1 Score with interactive bar chart |
| 🧩 Confusion Matrix | Interactive heatmap of true/false positives and negatives |
| 🔍 SHAP Explainability | Feature importance ranking + beeswarm summary plot |
| 💡 Auto Insights | Plain-English observations generated after every run |
| 🧍 Survival Predictor | Custom passenger form → instant prediction with probability |
| 📜 Run History | Lightweight history log comparing all experiments in a session |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| UI Framework | [Streamlit](https://streamlit.io) |
| Data Handling | [Pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) |
| Machine Learning | [scikit-learn](https://scikit-learn.org), [XGBoost](https://xgboost.readthedocs.io), [CatBoost](https://catboost.ai) |
| Explainability | [SHAP](https://shap.readthedocs.io) |
| Visualisation | [Plotly](https://plotly.com/python/), [Matplotlib](https://matplotlib.org), [Seaborn](https://seaborn.pydata.org) |
| Packaging | [setuptools](https://setuptools.pypa.io) |

Python **≥ 3.8** is required.

---

## 📂 Project Structure

```
Titanic/
├── app.py                   # Streamlit entry-point (UI layer)
├── data_loader.py           # Top-level thin wrapper (legacy)
├── explainer.py             # Top-level thin wrapper (legacy)
├── model_factory.py         # Top-level thin wrapper (legacy)
├── requirements.txt         # Python dependency list
├── setup.py                 # Package definition for ml_engine
│
└── ml_engine/               # Core reusable ML library
    ├── data_loader.py       # Load & split the Titanic dataset
    ├── preprocessing.py     # Missing-value imputation + encoding
    ├── feature_engineering.py  # FamilySize, IsAlone, Title, AgeBins
    ├── feature_selector.py  # Drop user-selected features safely
    ├── model_factory.py     # Instantiate classifiers by name
    ├── trainer.py           # Fit model + measure training time
    ├── evaluator.py         # Compute Accuracy/Precision/Recall/F1/CM
    ├── explainer.py         # SHAP values (TreeExplainer / KernelExplainer)
    ├── insights.py          # Auto-generate plain-English insights
    ├── pipeline.py          # Orchestrator — wires all modules together
    ├── train.csv            # Titanic training data
    ├── test.csv             # Titanic test data
    └── result.csv           # Sample result output
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Not-Animesh/Titanic.git
cd Titanic
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the `ml_engine` Package (Editable)

```bash
pip install -e .
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Streamlit will open the app in your default browser at `http://localhost:8501`.

> **Note:** If `titanic.csv` is not present in the root directory the app falls back to downloading the dataset automatically via the `seaborn` built-in datasets loader.

---

## 🔄 ML Pipeline

The pipeline is fully modular and lives inside `ml_engine/pipeline.py`. Each run executes the following 9 steps in order:

```
Raw CSV
  │
  ▼
1. Load         → ml_engine/data_loader.py   (load_titanic, split_data)
2. Preprocess   → ml_engine/preprocessing.py (imputation, encoding, column cleanup)
3. Engineer     → ml_engine/feature_engineering.py (FamilySize, IsAlone, Title, AgeBins)
4. Select       → ml_engine/feature_selector.py (drop user-selected features)
5. Split        → train / test (80 / 20, stratified)
6. Train        → ml_engine/trainer.py (fit + wall-clock time)
7. Evaluate     → ml_engine/evaluator.py (accuracy, precision, recall, F1, CM)
8. Explain      → ml_engine/explainer.py (SHAP values)
9. Insights     → ml_engine/insights.py (plain-English summaries)
```

---

## 🤖 Supported Models

| Model | Library | Key Hyperparameters |
|---|---|---|
| Logistic Regression | scikit-learn | `C=1.0`, `solver=lbfgs`, `max_iter=1000` |
| Random Forest | scikit-learn | `n_estimators=200`, `max_depth=8` |
| XGBoost | xgboost | `n_estimators=200`, `max_depth=5`, `lr=0.05` |
| CatBoost | catboost | `iterations=200`, `depth=6`, `lr=0.05` |

All models expose the standard sklearn API (`fit` / `predict` / `predict_proba`).

---

## 🧬 Feature Engineering

On top of the raw Titanic columns the pipeline creates four groups of derived features:

| Engineered Feature | Formula / Logic |
|---|---|
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | `1` if `FamilySize == 1` else `0` |
| `Title_*` | One-hot encoded title extracted from passenger name (`Mr`, `Miss`, `Mrs`, `Master`, `Rare`) |
| `AgeBin_child/adult/senior` | Age binned into `<16`, `16–60`, `>60` |

Categorical encoding applied during preprocessing:
- **Sex** → label-encoded (`male=1`, `female=0`)
- **Embarked** → one-hot encoded (drop_first=True)
- **Cabin** → dropped (>77 % missing)
- **Name**, **Ticket** → dropped (high-cardinality text)

---

## 🔬 Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) provides model-agnostic feature importance:

- **Tree-based models** (Random Forest, XGBoost, CatBoost, Gradient Boosting) → fast `TreeExplainer`.
- **Other models** (Logistic Regression, etc.) → `KernelExplainer` with a k-means background summary (≤ 100 samples for speed).

The app displays:
1. **Bar chart** — top-N features by mean absolute SHAP value.
2. **Beeswarm plot** — distribution of SHAP values across test samples.

---

## 📁 Dataset

The project uses the classic [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic/data).

| Column | Type | Description |
|---|---|---|
| `PassengerId` | int | Unique passenger identifier |
| `Survived` | int | Target: 0 = No, 1 = Yes |
| `Pclass` | int | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Name` | str | Passenger full name |
| `Sex` | str | Gender |
| `Age` | float | Age in years |
| `SibSp` | int | # siblings / spouses aboard |
| `Parch` | int | # parents / children aboard |
| `Ticket` | str | Ticket number |
| `Fare` | float | Passenger fare |
| `Cabin` | str | Cabin number (mostly missing) |
| `Embarked` | str | Port of embarkation (C / Q / S) |

---

## 📸 Screenshots & Sections

The Streamlit app is organised into these sections (top-to-bottom):

1. **Page header & introduction** — title, description, and welcome text.
2. **Data dictionary** — table explaining every column.
3. **Dataset overview + interactive filter** — browse and filter rows.
4. **Summary statistics** — descriptive stats with survival breakdown.
5. **ML Experimentation Lab** — model selector, feature selector, train button, metrics, SHAP charts, and insights.
6. **Survival prediction form** — manual passenger input → live prediction.
7. **Footer** — credits and links.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m "feat: your feature description"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a Pull Request.

Please keep the modular `ml_engine` structure intact and add docstrings to any new functions.
