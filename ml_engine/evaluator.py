"""
evaluator.py
Computes classification metrics for a trained model on a held-out test set.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Generate a metrics dictionary for the fitted model on the test set.

    Args:
        model: A fitted sklearn-compatible estimator.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Ground-truth test labels.

    Returns:
        dict with keys:
            accuracy       (float)
            precision      (float)
            recall         (float)
            f1_score       (float)
            confusion_matrix (np.ndarray, shape [2, 2])

    Raises:
        ValueError: If X_test or y_test is empty.
    """
    if X_test.empty or len(y_test) == 0:
        raise ValueError("Test data must not be empty.")

    # Ensure column names are strings
    X_test = X_test.copy()
    X_test.columns = [str(c) for c in X_test.columns]

    y_pred = model.predict(X_test)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
