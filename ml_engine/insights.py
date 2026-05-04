"""
insights.py
Generates 2–5 human-readable insights by comparing the current run
against the experiment history.
"""

import pandas as pd
from typing import Any, Dict, List


def generate_insights(
    current_run: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> List[str]:
    """
    Produce a list of concise insight strings based on the current experiment
    results and the accumulated history.

    Args:
        current_run (dict): Result dict from pipeline.run_pipeline().
        history (list): List of previous result dicts stored in session state.

    Returns:
        List[str]: 2–5 human-readable insight bullet strings.
    """
    insights: List[str] = []

    metrics = current_run.get("metrics", {})
    accuracy = metrics.get("accuracy", None)
    f1 = metrics.get("f1_score", None)
    precision = metrics.get("precision", None)
    recall = metrics.get("recall", None)
    model_name = current_run.get("model_name", "Unknown model")
    dropped = current_run.get("dropped_features", [])
    n_dropped = len(dropped)
    train_time = current_run.get("train_time", None)
    feature_importance: pd.DataFrame = current_run.get("feature_importance", pd.DataFrame())

    # ── Insight 1: overall accuracy snapshot ────────────────────────────────
    if accuracy is not None:
        level = "excellent" if accuracy >= 0.85 else "good" if accuracy >= 0.78 else "moderate"
        insights.append(
            f"🎯 **{model_name}** achieved {level} accuracy of "
            f"**{accuracy * 100:.1f}%** on the test set."
        )

    # ── Insight 2: impact of dropping features ───────────────────────────────
    if n_dropped == 0:
        insights.append("📋 No features were dropped — the model used all available features.")
    else:
        noun = "feature" if n_dropped == 1 else "features"
        dropped_str = ", ".join(f"`{f}`" for f in dropped[:3])
        suffix = " and more" if n_dropped > 3 else ""
        # Compare with a previous run of the same model without dropping
        baseline = _find_baseline(model_name, history)
        if baseline is not None:
            delta = accuracy - baseline["metrics"].get("accuracy", accuracy)
            direction = "increased" if delta >= 0 else "decreased"
            insights.append(
                f"📉 Dropping {n_dropped} {noun} ({dropped_str}{suffix}) "
                f"{direction} accuracy by **{abs(delta) * 100:.1f}%** "
                f"compared to the full-feature run."
            )
        else:
            insights.append(
                f"🗑️ Dropped {n_dropped} {noun}: {dropped_str}{suffix}."
            )

    # ── Insight 3: top influential feature ───────────────────────────────────
    if not feature_importance.empty:
        top_feature = feature_importance.iloc[0]["feature"]
        top_importance = feature_importance.iloc[0]["importance"]
        insights.append(
            f"⭐ Most influential feature: **`{top_feature}`** "
            f"(mean |SHAP| = {top_importance:.4f})."
        )

    # ── Insight 4: precision vs recall balance ────────────────────────────────
    if precision is not None and recall is not None:
        if abs(precision - recall) > 0.10:
            dominant = "precision" if precision > recall else "recall"
            insights.append(
                f"⚖️ The model is **{dominant}-heavy** "
                f"(precision={precision:.2f}, recall={recall:.2f}). "
                "Consider adjusting the classification threshold if balance matters."
            )
        else:
            insights.append(
                f"✅ Good precision–recall balance "
                f"(precision={precision:.2f}, recall={recall:.2f})."
            )

    # ── Insight 5: compare across history ────────────────────────────────────
    if len(history) >= 2:
        best_run = max(history, key=lambda r: r["metrics"].get("accuracy", 0))
        best_model = best_run.get("model_name", "Unknown")
        best_acc = best_run["metrics"].get("accuracy", 0)
        insights.append(
            f"🏆 Best experiment so far: **{best_model}** with "
            f"**{best_acc * 100:.1f}%** accuracy "
            f"({best_run.get('n_features', '?')} features used)."
        )
    elif train_time is not None:
        insights.append(f"⏱️ Training completed in **{train_time:.2f}s**.")

    return insights[:5]  # cap at 5


# ── Helpers ────────────────────────────────────────────────────────────────

def _find_baseline(
    model_name: str,
    history: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """
    Find the earliest history entry for the same model with 0 features dropped.

    Args:
        model_name (str): Current model name.
        history (list): Experiment history.

    Returns:
        dict | None: Baseline run or None if not found.
    """
    for run in history:
        if run.get("model_name") == model_name and run.get("dropped_features") == []:
            return run
    return None
