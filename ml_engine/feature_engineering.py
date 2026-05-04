"""
feature_engineering.py
Creates derived features from the preprocessed Titanic DataFrame.
All transformations are deterministic and leak-free.
"""

import pandas as pd
import numpy as np
import re


# ---------------------------------------------------------------------------
# Title extraction helpers
# ---------------------------------------------------------------------------

_TITLE_MAP = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    # Rare / honorary titles → 'Rare'
}


def _extract_title(name: str) -> str:
    """Extract title from a passenger name string."""
    match = re.search(r",\s*([^\.]+)\.", str(name))
    if match:
        title = match.group(1).strip()
        return _TITLE_MAP.get(title, "Rare")
    return "Unknown"


# ---------------------------------------------------------------------------
# Feature creation functions
# ---------------------------------------------------------------------------

def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    FamilySize = SibSp + Parch + 1 (self included).

    Args:
        df: DataFrame containing SibSp and Parch columns.

    Returns:
        DataFrame with FamilySize column added.
    """
    df = df.copy()
    if "SibSp" in df.columns and "Parch" in df.columns:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df


def add_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    """
    IsAlone = 1 if FamilySize == 1 else 0.

    Args:
        df: DataFrame containing FamilySize column.

    Returns:
        DataFrame with IsAlone column added.
    """
    df = df.copy()
    if "FamilySize" in df.columns:
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def add_title(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract passenger title from the raw Name column and one-hot encode it.

    Args:
        df: Preprocessed DataFrame (Name already dropped).
        raw_df: Original raw DataFrame (still has Name column).

    Returns:
        DataFrame with Title_* dummy columns added.
    """
    df = df.copy()
    if "Name" not in raw_df.columns:
        return df

    titles = raw_df["Name"].apply(_extract_title).rename("Title")
    title_dummies = pd.get_dummies(titles, prefix="Title", drop_first=False)
    # Align index before concatenation
    title_dummies.index = df.index
    df = pd.concat([df, title_dummies], axis=1)
    return df


def add_age_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin Age into three categories:
        child  (< 16)
        adult  (16–60)
        senior (> 60)

    The numeric Age column is retained; bins are added as integer dummy columns.

    Args:
        df: DataFrame containing Age column.

    Returns:
        DataFrame with AgeBin_child, AgeBin_adult, AgeBin_senior columns added.
    """
    df = df.copy()
    if "Age" not in df.columns:
        return df

    bins = [0, 16, 60, 120]
    labels = ["child", "adult", "senior"]
    age_bin = pd.cut(df["Age"], bins=bins, labels=labels)
    age_dummies = pd.get_dummies(age_bin, prefix="AgeBin", drop_first=False).astype(int)
    age_dummies.index = df.index
    df = pd.concat([df, age_dummies], axis=1)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature-engineering pipeline.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        raw_df (pd.DataFrame): Original raw DataFrame (needed for Name/Title).

    Returns:
        pd.DataFrame: DataFrame enriched with engineered features.
    """
    df = add_family_size(df)
    df = add_is_alone(df)
    df = add_title(df, raw_df)
    df = add_age_bins(df)

    # Ensure all column names are strings (get_dummies can produce categories)
    df.columns = [str(c) for c in df.columns]

    return df
