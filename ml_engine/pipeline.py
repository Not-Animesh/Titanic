"""
pipeline.py
Core orchestrator: wires together all ml_engine modules into a single
run_pipeline() call consumed by the Streamlit app.
"""

from typing import Any, Dict, List

from ml_engine.data_loader import load_titanic, split_data
from ml_engine.preprocessing import preprocess
from ml_engine.feature_engineering import engineer_features
from ml_engine.feature_selector import drop_features
from ml_engine.model_factory import get_model
from ml_engine.trainer import train_model
from ml_engine.evaluator import evaluate_model
from ml_engine.explainer import compute_shap
from ml_engine.insights import generate_insights


def run_pipeline(
    drop_feature_list: List[str],
    model_name: str,
    history: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute the full ML pipeline end-to-end.

    Steps:
        1. Load raw Titanic data
        2. Preprocess (imputation + encoding)
        3. Feature engineering (FamilySize, IsAlone, Title, AgeBins)
        4. Drop user-selected features
        5. Train / test split
        6. Instantiate + train the model
        7. Evaluate on test set
        8. Compute SHAP explanations
        9. Generate human-readable insights

    Args:
        drop_feature_list (List[str]): Column names to exclude before training.
        model_name (str): Model identifier understood by model_factory.get_model().
        history (List[dict]): Accumulated experiment history from session state.

    Returns:
        dict with keys:
            model_name       (str)
            metrics          (dict)  – accuracy, precision, recall, f1_score, confusion_matrix
            model            (estimator)
            train_time       (float) – seconds
            feature_importance (pd.DataFrame) – ['feature', 'importance']
            shap_values      (np.ndarray)
            insights         (List[str])
            dropped_features (List[str])
            n_features       (int)  – number of features after dropping
    """
    # ── 1. Load ──────────────────────────────────────────────────────────────
    raw_df = load_titanic()

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    processed_df = preprocess(raw_df)

    # ── 3. Feature engineering ───────────────────────────────────────────────
    engineered_df = engineer_features(processed_df, raw_df)

    # ── 4. Drop user-selected features ───────────────────────────────────────
    reduced_df = drop_features(engineered_df, drop_list=drop_feature_list)

    # ── 5. Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(reduced_df, target="Survived")
    n_features = X_train.shape[1]

    # ── 6. Train ─────────────────────────────────────────────────────────────
    model = get_model(model_name)
    trained_model, train_time = train_model(model, X_train, y_train)

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    metrics = evaluate_model(trained_model, X_test, y_test)

    # ── 8. Explain (SHAP) ────────────────────────────────────────────────────
    shap_values, feature_importance = compute_shap(trained_model, X_train, X_test)

    # ── 9. Insights ───────────────────────────────────────────────────────────
    current_run_meta = {
        "model_name": model_name,
        "metrics": metrics,
        "train_time": train_time,
        "dropped_features": list(drop_feature_list),
        "n_features": n_features,
        "feature_importance": feature_importance,
    }
    insights = generate_insights(current_run_meta, history or [])

    return {
        "model_name": model_name,
        "metrics": metrics,
        "model": trained_model,
        "train_time": train_time,
        "feature_importance": feature_importance,
        "shap_values": shap_values,
        "insights": insights,
        "dropped_features": list(drop_feature_list),
        "n_features": n_features,
        # Keep a serialisable snapshot for history storage
        "X_test": X_test,
        "y_test": y_test,
    }
