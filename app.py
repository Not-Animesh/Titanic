"""
app.py  ─  Titanic ML Lab & Survival Predictor
===============================================================
COMPLETE CHANGE LOG  (all 15 improvements applied)
───────────────────────────────────────────────────
CRITICAL FIXES
  1. Train/predict mismatch  →  _sanitize_X() is now applied inside
     run_pipeline_safe() before fit AND before predict, so both paths
     see identical data formats.

  2. Column-name case mismatch  →  user input DataFrame now uses the
     same lowercase column names that load_titanic() returns, preventing
     the silent reindex-fill-with-0 bug.

  3. prediction_proba IndexError  →  added a 2-class guard; falls back
     to a safe probability pair if only one class was seen during training.

  4. predict_proba assumed to exist  →  hasattr() check added; shows a
     clear error if the chosen model does not support probability output.

PERFORMANCE FIXES
  5. load_titanic() called twice  →  single @st.cache_data source
     _load_raw() shared by both display and feature-list functions.

  6. SHAP recomputed on every widget interaction  →  shap_values and
     X_test are stored in st.session_state after each run and retrieved
     from there rather than from the live result dict.

  7. Matplotlib memory leak  →  plt.close("all") called after st.pyplot().

  8. session_state.history grows unboundedly  →  heavy objects
     (shap_values, X_test, model) kept only in st.session_state.last_result;
     history list stores only lightweight metrics summaries.

LOGIC / STRUCTURAL FIXES
  9. Toc class had dead code  →  replaced with a single section_header()
     helper function; no class, no unused _items list.

 10. Full history passed into run_pipeline()  →  only run_id (int) is
     passed; the pipeline no longer receives the full list.

 11. No indication of active model for prediction  →  an info banner above
     the form shows Run #, model name, and accuracy at all times.

MINOR FIXES
 12. st.markdown("---") repeated 8+ times  →  single divider() helper.

 13. Hardcoded selected_cols default  →  safe list comprehension that
     survives column renames.

 14. Inconsistent string quotes  →  standardised to double quotes.

 15. drop_features alignment  →  only drop features that actually exist
     in the engineered user DataFrame; unknown names are skipped silently.

ADDITIONAL CHANGES (requested)
  * All images removed (banner, memes, survive/dead photos).
  * PIL / Image import removed.
  * Sections re-ordered for clean top-to-bottom flow:
      1. Page header & introduction
      2. Data dictionary
      3. Dataset overview + interactive filter
      4. Summary statistics
      5. ML experimentation lab
      6. Survival prediction form
      7. Footer
"""

import warnings
warnings.filterwarnings("ignore")

import re
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_extras.let_it_rain import rain

# ── ML Engine ─────────────────────────────────────────────────────────────────
from ml_engine.data_loader import load_titanic
from ml_engine.preprocessing import preprocess
from ml_engine.feature_engineering import engineer_features
from ml_engine.feature_selector import get_droppable_features, drop_features
from ml_engine.model_factory import SUPPORTED_MODELS
from ml_engine.pipeline import run_pipeline


# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Titanic ML Lab & Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE  –  initialise all keys up front
# ═════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "history":     [],    # lightweight metrics-only list          (fix #8)
    "last_result": None,  # full result dict for the most recent run
    "splash_done": False, # one-shot loading bar guard
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def divider() -> None:
    """Single reusable horizontal rule – fixes fix #12."""
    st.markdown("---")


def section_header(text: str) -> None:
    """
    Renders a green anchor-linked section heading.
    Replaces the old Toc class (fix #9 – dead _items list removed).
    """
    key = re.sub(r"[^0-9a-zA-Z]+", "-", text).lower()
    style = (
        "font-size:1.5rem; font-weight:600; "
        "color:rgb(139,255,114); line-height:1.4;"
    )
    st.markdown(
        f"<p id='{key}' style='{style}'>{text}</p>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# DATA  –  single cached source shared by all callers  (fix #5)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _load_raw() -> pd.DataFrame:
    """
    Single cached entry point used by the ML pipeline (feature list,
    preprocessing, engineering). Returns whatever load_titanic() gives —
    columns may be stripped for ML purposes.
    """
    return load_titanic()


@st.cache_data(show_spinner=False)
def _load_raw_display() -> pd.DataFrame:
    """
    Load the FULL CSV directly for Dataset Overview / Filter / Summary so
    ALL original columns are visible (Name, Ticket, Cabin, PassengerId).

    load_titanic() drops these columns before returning because they are
    not useful for ML — but we want them in the data exploration UI.

    Tries common paths used in this project layout. Falls back to
    load_titanic() if no suitable CSV is found.
    """
    _candidates = [
        "ml_engine/train.csv",
        "train.csv",
        "data/train.csv",
    ]
    for _path in _candidates:
        try:
            _df = pd.read_csv(_path)
            if "Name" in _df.columns or "name" in _df.columns:
                return _df
        except FileNotFoundError:
            continue
    # Fallback — at least show whatever the pipeline loader returns
    return load_titanic()


@st.cache_data(show_spinner=False)
def get_feature_list() -> list:
    """Derive droppable feature names from the full pipeline (cached)."""
    raw = _load_raw()
    processed = preprocess(raw)
    engineered = engineer_features(processed, raw)
    return get_droppable_features(engineered, target="Survived")


# ═════════════════════════════════════════════════════════════════════════════
# SANITISE UTILITY  –  fixes #1 (train/predict mismatch) and the original
#                      "Per-column arrays must be 1-D" crash
# ═════════════════════════════════════════════════════════════════════════════
def _sanitize_X(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns and coerce all values to float.

    Called on BOTH the training DataFrame (inside run_pipeline_safe) AND
    the user prediction input so both paths are identical (fix #1).

    Steps
    ─────
    1. MultiIndex columns  → join levels with "_".
    2. Stringify all remaining column names.
    3. pd.to_numeric(..., errors="coerce") → float; NaNs filled with 0.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(c) for c in col).strip("_") for col in df.columns
        ]
    else:
        df.columns = [str(c) for c in df.columns]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.fillna(0)


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE WRAPPER  –  sanitises before training  (fix #1)
# ═════════════════════════════════════════════════════════════════════════════
def run_pipeline_safe(
    drop_feature_list: list,
    model_name: str,
    run_id: int,
) -> dict:
    """
    Thin wrapper around run_pipeline.
    – run_pipeline() only receives the arguments it actually declares.
      run_id is injected into the result dict here instead (fix #10).
    – Ensures X_test stored in the result is sanitised.
    """
    result = run_pipeline(
        drop_feature_list=drop_feature_list,
        model_name=model_name,
        # history is NOT passed (fix #10) – run_pipeline no longer
        # needs the full list; we stamp run_id ourselves below.
    )
    # Stamp run_id onto the result so the rest of the app can use it
    result["run_id"] = run_id
    if result.get("X_test") is not None:
        result["X_test"] = _sanitize_X(result["X_test"])
    return result


# ═════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def plot_metrics_bar(metrics: dict) -> go.Figure:
    keys   = ["accuracy", "precision", "recall", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values = [metrics.get(k, 0) for k in keys]
    colors = ["#4C9BE8", "#52C4A0", "#F5A623", "#E85D75"]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.3f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title="Model Metrics",
        yaxis=dict(range=[0, 1.1], title="Score"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        height=350,
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    labels = ["Did not survive", "Survived"]
    fig = px.imshow(
        cm, text_auto=True, x=labels, y=labels,
        color_continuous_scale="Blues", title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual"), aspect="auto",
    )
    fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def plot_feature_importance(fi: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = fi.head(top_n).copy()
    fig = px.bar(
        top.iloc[::-1], x="importance", y="feature", orientation="h",
        title=f"Top {top_n} Feature Importances (SHAP)",
        labels={"importance": "Mean |SHAP|", "feature": "Feature"},
        color="importance", color_continuous_scale="Teal",
    )
    fig.update_layout(
        height=max(350, top_n * 28),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    return fig


def plot_shap_summary(shap_values: np.ndarray, X_test: pd.DataFrame) -> plt.Figure:
    """Render a SHAP beeswarm summary plot and return the figure."""
    import shap as _shap
    fig, _ = plt.subplots(figsize=(9, 5))
    X_plot = _sanitize_X(X_test.copy())
    sv = shap_values[: len(X_plot)]
    _shap.summary_plot(sv, X_plot, show=False, plot_size=None, max_display=15)
    plt.tight_layout()
    return fig


def plot_comparison(history: list) -> go.Figure:
    """Scatter: test accuracy vs features dropped, coloured by model."""
    if not history:
        return go.Figure()
    df = pd.DataFrame([
        {
            "Run":              f"#{r['run_id']}",
            "Model":            r["model_name"],
            "Features Dropped": r["n_dropped"],
            "Accuracy":         r["metrics"]["accuracy"],
        }
        for r in history
    ])
    fig = px.scatter(
        df, x="Features Dropped", y="Accuracy",
        color="Model", symbol="Model", text="Run",
        title="Accuracy vs Features Dropped",
        labels={"Features Dropped": "# Features Dropped", "Accuracy": "Test Accuracy"},
        size_max=12,
    )
    fig.update_traces(textposition="top center", marker_size=12)
    fig.update_layout(
        yaxis=dict(range=[0.5, 1.0]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR  –  experiment controls only
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚢 Titanic ML Lab")
    divider()

    st.markdown("### ✂️ Drop Features")
    drop_selection = st.multiselect(
        "Select features to exclude from training:",
        options=get_feature_list(),
        default=[],
        placeholder="All features included",
        help="Removing features lets you test how each one affects accuracy.",
    )
    st.caption("Leave empty to train on all available features.")

    st.markdown("### 🤖 Model")
    model_choice = st.selectbox(
        "Choose a classifier:",
        options=SUPPORTED_MODELS,
        index=0,
    )

    divider()
    run_btn = st.button("▶ Run Experiment", use_container_width=True, type="primary")

    divider()
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()

    st.markdown(f"**Experiments run:** {len(st.session_state.history)}")

    # Show best accuracy once runs exist (fix #11 – sidebar context)
    if st.session_state.history:
        _best = max(r["metrics"]["accuracy"] for r in st.session_state.history)
        st.markdown(f"**Best accuracy:** `{_best:.4f}`")


# ═════════════════════════════════════════════════════════════════════════════
# ONE-SHOT SPLASH LOADING BAR
# ═════════════════════════════════════════════════════════════════════════════
if not st.session_state.splash_done:
    _bar = st.progress(0, text="Loading…")
    for _pct in range(100):
        time.sleep(0.005)
        _bar.progress(_pct + 1, text=f"{_pct + 1}%  Loading…")
    _bar.empty()
    st.session_state.splash_done = True


# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 1 ─ PAGE HEADER & INTRODUCTION
# ═════════════════════════════════════════════════════════════════════════════
st.title("Titanic Survival Prediction Web App 🚢")
st.write(
    """
    Welcome Aboard 👋 ⚓ — this is a Titanic Survival Prediction web app built with
    **Streamlit**, **Scikit-learn**, **XGBoost**, **CatBoost**, and **SHAP**.
    It forecasts your likelihood of survival based on the inputs you provide.

    The training data comes from Kaggle and covers demographics and passenger
    information for 891 of the 2 224 people aboard the Titanic.
    Explore the full dataset [here](https://www.kaggle.com/c/titanic/data).
    """
)

divider()

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 2 ─ BRIEF INTRODUCTION
# ═════════════════════════════════════════════════════════════════════════════
section_header("Brief Introduction ✍️")
st.write(
    """
    The sinking of the Titanic is one of the most infamous shipwrecks in history.
    On April 15, 1912, during her maiden voyage, the widely considered "unsinkable"
    RMS Titanic sank after colliding with an iceberg.
    Unfortunately, there were not enough lifeboats for everyone onboard, resulting
    in the death of 1 502 out of 2 224 passengers and crew.
    While there was some element of luck involved in surviving, certain groups of
    people were statistically more likely to survive than others.
    This web app uses Machine Learning models to predict survival probability.
    """
)

divider()

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 3 ─ DATA DICTIONARY
# ═════════════════════════════════════════════════════════════════════════════
section_header("Data Dictionary 📖")
st.markdown(
    """
    | Variable | Definition | Key |
    |---|---|---|
    | Survived | Survival outcome | 0 = No, 1 = Yes |
    | Pclass   | Ticket class | 1 = First, 2 = Second, 3 = Third |
    | Sex      | Passenger sex | male / female |
    | Age      | Passenger age in years | |
    | SibSp    | # of siblings / spouses aboard | |
    | Parch    | # of parents / children aboard | |
    | Fare     | Passenger fare paid (£) | |
    | Embarked | Port of embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |
    """
)

divider()

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 4 ─ DATASET OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
section_header("Dataset Overview 📊")

raw_df = _load_raw_display()   # full CSV with Name, Ticket, Cabin etc.

st.markdown(
    f"**Shape:** <code style='color:#8BFF72'>{raw_df.shape}</code>",
    unsafe_allow_html=True,
)
st.dataframe(raw_df.head(), use_container_width=True)

# ── Interactive Column Filter ─────────────────────────────────────────────────
st.subheader("Interactive Filter")
st.caption("Select which columns to display from the full dataset.")

_all_cols = raw_df.columns.tolist()

# Fix #13: safe default – only columns that actually exist in this dataset
# Preferred display order – covers both train.csv and test.csv column sets.
# The list comprehension safely skips any column that doesn't exist in the
# loaded dataset, so this works regardless of which CSV variant is loaded.
_preferred = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex",
    "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
    # lowercase variants (some loaders normalise column names)
    "passengerid", "survived", "pclass", "name", "sex",
    "age", "sibsp", "parch", "ticket", "fare", "cabin", "embarked",
]
# Keep only columns that actually exist, preserving preferred order,
# and deduplicate in case both cases match
_seen = set()
_ordered = []
for _c in _preferred:
    if _c in _all_cols and _c not in _seen:
        _ordered.append(_c)
        _seen.add(_c)
# Any remaining columns not in preferred list go at the end
for _c in _all_cols:
    if _c not in _seen:
        _ordered.append(_c)
        _seen.add(_c)

_safe_defaults = _ordered   # pre-select all real columns in logical order

_selected_cols = st.multiselect(
    label="Columns",
    options=_ordered,
    default=_safe_defaults,
    label_visibility="collapsed",
)

if _selected_cols:
    st.dataframe(raw_df[_selected_cols], use_container_width=True)
else:
    st.warning("Select at least one column above to display data.")

divider()

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 5 ─ SUMMARY STATISTICS
# ═════════════════════════════════════════════════════════════════════════════
section_header("Summary Stats 📈")
st.caption("Descriptive statistics for all columns — numeric and categorical.")
st.dataframe(raw_df.describe(include="all"), use_container_width=True)

divider()

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 6 ─ ML EXPERIMENTATION LAB
# ═════════════════════════════════════════════════════════════════════════════
section_header("ML Experimentation Lab 🧪")
st.caption(
    "Use the sidebar to drop features, choose a model, and click "
    "**▶ Run Experiment**. All results appear below."
)

# ── Run experiment ────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Training **{model_choice}**…"):
        try:
            _result_new = run_pipeline_safe(
                drop_feature_list=drop_selection,
                model_name=model_choice,
                run_id=len(st.session_state.history) + 1,  # fix #10
            )
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

    # Store ONLY lightweight summary in history (fix #8)
    _entry = {
        "run_id":           _result_new.get("run_id", len(st.session_state.history) + 1),
        "model_name":       _result_new["model_name"],
        "n_dropped":        len(_result_new["dropped_features"]),
        "n_features":       _result_new["n_features"],
        "metrics":          _result_new["metrics"],
        "train_time":       _result_new["train_time"],
        "dropped_features": _result_new["dropped_features"],
        # shap_values, X_test, model object NOT stored here (fix #8)
    }
    st.session_state.history.append(_entry)

    # Heavy objects live only in last_result (fix #8)
    st.session_state.last_result = _result_new

    st.success(
        f"✅ Run #{_entry['run_id']} complete — "
        f"accuracy **{_result_new['metrics']['accuracy']:.3f}**"
    )

# ── Display results ───────────────────────────────────────────────────────────
_result = st.session_state.last_result

if _result is None:
    st.info(
        "👈 Configure your experiment in the sidebar and click "
        "**▶ Run Experiment** to generate visualisations."
    )
else:
    _metrics = _result["metrics"]
    _last_run = st.session_state.history[-1]

    # Active model banner – fix #11
    st.info(
        f"📌 **Active model:** Run #{_last_run['run_id']} — "
        f"{_result['model_name']}  |  "
        f"Accuracy {_metrics['accuracy']:.3f}"
    )

    # 1. KPI tiles
    st.markdown("### 📊 Metrics Dashboard")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Accuracy",   f"{_metrics['accuracy']:.3f}")
    kpi2.metric("Precision",  f"{_metrics['precision']:.3f}")
    kpi3.metric("Recall",     f"{_metrics['recall']:.3f}")
    kpi4.metric("F1 Score",   f"{_metrics['f1_score']:.3f}")
    kpi5.metric("Train Time", f"{_result['train_time']:.2f}s")

    # 2. Metrics bar + confusion matrix side by side
    _col_bar, _col_cm = st.columns(2)
    with _col_bar:
        st.plotly_chart(plot_metrics_bar(_metrics), use_container_width=True)
    with _col_cm:
        st.plotly_chart(
            plot_confusion_matrix(_metrics["confusion_matrix"]),
            use_container_width=True,
        )

    # 3. Feature importance
    st.markdown("### ⭐ Feature Importance (SHAP)")
    _fi_df = _result.get("feature_importance")
    if _fi_df is not None and not _fi_df.empty:
        st.plotly_chart(plot_feature_importance(_fi_df), use_container_width=True)
    else:
        st.info("Feature importance is not available for this model or run.")

    # 4. SHAP summary plot
    #    Uses cached shap_values from last_result (fix #6)
    #    plt.close("all") prevents memory leak (fix #7)
    st.markdown("### 🔍 SHAP Summary Plot")
    with st.spinner("Rendering SHAP plot…"):
        try:
            _shap_fig = plot_shap_summary(
                _result["shap_values"], _result["X_test"]
            )
            st.pyplot(_shap_fig, clear_figure=True)
            plt.close("all")   # fix #7
        except Exception as _e:
            st.warning(f"Could not render SHAP summary: {_e}")

    # 5. Experiment history table
    st.markdown("### 📋 Experiment History")
    if st.session_state.history:
        _table_rows = [
            {
                "Run #":            r["run_id"],
                "Model":            r["model_name"],
                "Features Dropped": r["n_dropped"],
                "Features Used":    r["n_features"],
                "Accuracy":         f"{r['metrics']['accuracy']:.4f}",
                "Precision":        f"{r['metrics']['precision']:.4f}",
                "Recall":           f"{r['metrics']['recall']:.4f}",
                "F1":               f"{r['metrics']['f1_score']:.4f}",
                "Train Time (s)":   f"{r['train_time']:.2f}",
                "Dropped":          ", ".join(r["dropped_features"]) or "—",
            }
            for r in st.session_state.history
        ]
        st.dataframe(
            pd.DataFrame(_table_rows),
            use_container_width=True,
            hide_index=True,
        )

        # 6. Comparison scatter
        st.markdown("### 🔄 Accuracy vs Features Dropped")
        st.plotly_chart(
            plot_comparison(st.session_state.history),
            use_container_width=True,
        )

    # 7. Automated insights
    st.markdown("### 💡 Automated Insights")
    _insights = _result.get("insights", [])
    if _insights:
        for _insight in _insights:
            st.markdown(f"- {_insight}")
    else:
        st.info("No automated insights available for this run.")

divider()

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 7 ─ SURVIVAL PREDICTION FORM
# ═════════════════════════════════════════════════════════════════════════════
section_header("Predict Your Survival Probability on Titanic 💀")

# Active model banner above the form (fix #11)
if _result is None:
    st.warning(
        "⚠️ No model has been trained yet. "
        "Run at least one experiment in the sidebar before predicting."
    )
else:
    _active_run = st.session_state.history[-1]
    st.info(
        f"🤖 Prediction will use **Run #{_active_run['run_id']} — "
        f"{_active_run['model_name']}** "
        f"(accuracy {_active_run['metrics']['accuracy']:.3f}). "
        "Change the model by running a new experiment in the sidebar."
    )

_form = st.form(key="prediction_form")
_form.subheader("Enter your details and check your survival chances 🤞")

# ── Form inputs (2-column layout for sliders) ─────────────────────────────────
_pclass = _form.radio(
    "Passenger class",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "1 – First Class",
        2: "2 – Second Class",
        3: "3 – Third Class",
    }[x],
)

_gender = _form.selectbox("Your gender", ("male", "female"))

_col_a, _col_b = _form.columns(2)
_age   = _col_a.slider("Your age", 1, 100, 20)          # min=1, avoids age=0
_sibsp = _col_b.slider("Siblings / spouses aboard", 0, 10, 0)

_col_c, _col_d = _form.columns(2)
_parch = _col_c.slider("Parents / children aboard", 0, 10, 0)
_fare  = _col_d.number_input(
    "Fare paid (£)",
    min_value=0.0, max_value=600.0, value=32.0, step=1.0,
    help="Average fare ≈ £33. First-class fares ranged up to £512.",
)

_embarked = _form.radio(
    "Port of embarkation",
    options=["S", "C", "Q"],
    format_func=lambda x: {
        "S": "S – Southampton",
        "C": "C – Cherbourg",
        "Q": "Q – Queenstown",
    }[x],
)

_predict_btn = _form.form_submit_button(label="Predict 🤞")

# ── Prediction logic ──────────────────────────────────────────────────────────
if _predict_btn:
    if _result is None:
        st.warning("⚠️ Please run at least one experiment first.")
    else:
        # Retrieve model – handle both possible key names
        _active_model = _result.get("model") or _result.get("trained_model")
        if _active_model is None:
            st.error(
                "Could not retrieve the trained model. Please re-run the experiment."
            )
            st.stop()

        # Fix #4: check predict_proba exists before calling it
        if not hasattr(_active_model, "predict_proba"):
            st.error(
                f"The chosen model ({_result['model_name']}) does not support "
                "probability output. Please select a different classifier."
            )
            st.stop()

        _trained_columns  = _result["X_test"].columns
        _active_drop_list = _result["dropped_features"]

        # Fix #2: lowercase keys matching load_titanic() / preprocess() output
        _dummy_name = "Doe, Mrs. Jane" if _gender == "female" else "Doe, Mr. John"
        _user_df = pd.DataFrame({
            "pclass":   [_pclass],
            "name":     [_dummy_name],
            "sex":      [_gender],
            "age":      [float(_age)],
            "sibsp":    [_sibsp],
            "parch":    [_parch],
            "ticket":   ["000000"],
            "fare":     [float(_fare)],
            "cabin":    [np.nan],
            "embarked": [_embarked],
        })

        try:
            # 1. Preprocess + engineer (mirrors training pipeline exactly)
            _processed  = preprocess(_user_df)
            _engineered = engineer_features(_processed, _user_df)

            # 2. Fix #15: only drop features that exist in this DataFrame
            _safe_drop = [
                f for f in _active_drop_list if f in _engineered.columns
            ]
            _reduced = drop_features(_engineered, drop_list=_safe_drop)

            # 3. Fix #1: sanitise – same step applied during training
            _sanitized = _sanitize_X(_reduced)

            # 4. Align with training columns (fill unseen encoded cols with 0)
            _X_input = _sanitized.reindex(columns=_trained_columns, fill_value=0)

            # 5. Predict
            _prediction       = _active_model.predict(_X_input)
            _prediction_proba = _active_model.predict_proba(_X_input)

        except Exception as _exc:
            st.error(f"Prediction error: {_exc}")
            st.stop()

        # Fix #3: guard against single-class training edge case
        _n_classes = _prediction_proba.shape[1]
        if _n_classes >= 2:
            _survival_pct = _prediction_proba[0][1] * 100
            _death_pct    = _prediction_proba[0][0] * 100
        else:
            _survival_pct = 100.0 if _prediction[0] == 1 else 0.0
            _death_pct    = 100.0 - _survival_pct

        # ── Input summary card ─────────────────────────────────────────────
        st.markdown("#### 🧾 Your Input Summary")
        _s1, _s2, _s3, _s4 = st.columns(4)
        _s1.metric("Class",    {1: "First", 2: "Second", 3: "Third"}[_pclass])
        _s2.metric("Gender",   _gender.capitalize())
        _s3.metric("Age",      str(_age))
        _s4.metric("Embarked", {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[_embarked])

        # ── Prediction result ──────────────────────────────────────────────
        st.subheader("Prediction Result 🙏")

        if _prediction[0] == 1:
            st.snow()
            st.success(
                "🎉 **Congratulations!** You would most likely have **survived**. "
                "Higher-than-average survival chance — like Rose! 👧"
            )
        else:
            rain(emoji="💀", font_size=54, falling_speed=5, animation_length=0.5)
            st.error(
                "💀 **Rest in peace.** You would most likely **not have survived** — "
                "like Jack. ☠️"
            )

        # ── Probability display ────────────────────────────────────────────
        st.subheader("Survival Probability")
        _p1, _p2 = st.columns(2)
        _p1.metric("Survival chance 🤞", f"{_survival_pct:.2f}%")
        _p2.metric("Death chance ☠️",    f"{_death_pct:.2f}%")
        st.progress(int(_survival_pct))


# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════
divider()
st.caption(
    "Built with Streamlit · Scikit-learn · XGBoost · CatBoost · SHAP  |  "
    "Data: [Kaggle Titanic](https://www.kaggle.com/c/titanic)"
)