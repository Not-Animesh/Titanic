"""
explainer.py
Computes SHAP values and derives a feature-importance ranking.

Tree-based models (Random Forest, XGBoost, CatBoost) use TreeExplainer.
All others fall back to a lightweight KernelExplainer with a small background
sample to keep latency acceptable inside a Streamlit app.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

import shap

# Model types that support SHAP TreeExplainer
_TREE_MODEL_CLASSES = (
    "RandomForestClassifier",
    "XGBClassifier",
    "CatBoostClassifier",
    "GradientBoostingClassifier",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
)


def _is_tree_model(model: Any) -> bool:
    return type(model).__name__ in _TREE_MODEL_CLASSES


def compute_shap(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_background: int = 100,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values for the test set and return a feature-importance ranking.

    Args:
        model: Fitted sklearn-compatible estimator.
        X_train (pd.DataFrame): Training data (used as background for KernelExplainer).
        X_test (pd.DataFrame): Test data to explain.
        max_background (int): Max background samples for KernelExplainer.

    Returns:
        Tuple:
            shap_values (np.ndarray): SHAP values array for the positive class.
            feature_importance (pd.DataFrame): DataFrame with columns
                ['feature', 'importance'] sorted descending.
    """
    # Ensure string column names
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train.columns = [str(c) for c in X_train.columns]
    X_test.columns = [str(c) for c in X_test.columns]

    feature_names = list(X_test.columns)

    if _is_tree_model(model):
        explainer = shap.TreeExplainer(model)
        shap_output = explainer.shap_values(X_test)

        # For binary classifiers, shap_values may be:
        #   - a list [class0_array, class1_array]  (older SHAP)
        #   - a 3-D ndarray of shape (n_samples, n_features, n_classes)  (newer SHAP)
        if isinstance(shap_output, list):
            sv = shap_output[1]  # positive class
        elif isinstance(shap_output, np.ndarray) and shap_output.ndim == 3:
            sv = shap_output[:, :, 1]  # positive class
        else:
            sv = shap_output
    else:
        # KernelExplainer — use a small summary of the training set as background
        n_bg = min(max_background, len(X_train))
        background = shap.kmeans(X_train, min(50, n_bg))

        def predict_proba_pos(x):
            return model.predict_proba(x)[:, 1]

        explainer = shap.KernelExplainer(predict_proba_pos, background)
        # Limit test samples to keep it fast
        X_explain = X_test.iloc[:min(100, len(X_test))]
        sv = explainer.shap_values(X_explain, nsamples=100)

    # Mean absolute SHAP across all test samples → feature importance
    mean_abs = np.abs(sv).mean(axis=0)
    feature_importance = (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return sv, feature_importance
