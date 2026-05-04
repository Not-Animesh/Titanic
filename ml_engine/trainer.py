"""
trainer.py
Fits a model on training data and tracks elapsed training time.
"""

import time
import pandas as pd
import numpy as np
from typing import Any, Tuple


def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Any, float]:
    """
    Train the supplied model and measure wall-clock training time.

    Args:
        model: An unfitted sklearn-compatible estimator.
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector.

    Returns:
        Tuple[estimator, float]:
            - The fitted estimator.
            - Training duration in seconds (rounded to 4 decimal places).

    Raises:
        ValueError: If X_train or y_train is empty.
    """
    if X_train.empty or len(y_train) == 0:
        raise ValueError("Training data must not be empty.")

    # Ensure column names are strings (required by XGBoost / CatBoost)
    X_train = X_train.copy()
    X_train.columns = [str(c) for c in X_train.columns]

    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = round(time.perf_counter() - start, 4)

    return model, elapsed
