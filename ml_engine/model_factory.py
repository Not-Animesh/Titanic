"""
model_factory.py
Returns a configured sklearn-compatible classifier based on a name string.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


SUPPORTED_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "CatBoost",
]


def get_model(model_name: str):
    """
    Instantiate and return a classifier for the given model name.

    All models use reasonable default hyperparameters and are compatible
    with the sklearn API (fit / predict / predict_proba).

    Args:
        model_name (str): One of SUPPORTED_MODELS.

    Returns:
        Estimator: A configured, unfitted sklearn-compatible classifier.

    Raises:
        ValueError: If model_name is not in SUPPORTED_MODELS.
    """
    name = model_name.strip()

    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=42,
        )

    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        )

    if name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is not installed. Run: pip install xgboost") from exc
        return XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )

    if name == "CatBoost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("catboost is not installed. Run: pip install catboost") from exc
        return CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            random_seed=42,
            verbose=0,
            allow_writing_files=False,
        )

    raise ValueError(
        f"Unknown model '{name}'. Choose from: {SUPPORTED_MODELS}"
    )
