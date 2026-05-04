"""
feature_selector.py
Provides a safe utility to drop user-selected features from a DataFrame.
The target column is always protected from removal.
"""

import pandas as pd
from typing import List


TARGET_COLUMN = "Survived"


def drop_features(df: pd.DataFrame, drop_list: List[str], target: str = TARGET_COLUMN) -> pd.DataFrame:
    """
    Drop specified feature columns from the DataFrame.

    Rules:
        - The target column is never dropped.
        - Columns not present in the DataFrame are silently ignored.
        - An empty drop_list is valid (returns the DataFrame unchanged).

    Args:
        df (pd.DataFrame): Input DataFrame.
        drop_list (List[str]): Column names to drop.
        target (str): Target column to protect.

    Returns:
        pd.DataFrame: DataFrame with selected columns removed.
    """
    if not isinstance(drop_list, (list, tuple)):
        raise TypeError(f"drop_list must be a list, got {type(drop_list).__name__}.")

    # Protect the target column
    safe_drop = [col for col in drop_list if col != target]

    # Only drop columns that actually exist
    existing_drop = [col for col in safe_drop if col in df.columns]

    if not existing_drop:
        return df.copy()

    return df.drop(columns=existing_drop)


def get_droppable_features(df: pd.DataFrame, target: str = TARGET_COLUMN) -> List[str]:
    """
    Return all column names that are eligible to be dropped (i.e., not the target).

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Target column to exclude.

    Returns:
        List[str]: Sorted list of droppable column names.
    """
    return sorted([c for c in df.columns if c != target])
