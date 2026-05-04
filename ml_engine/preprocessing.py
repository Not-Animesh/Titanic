"""
preprocessing.py
Handles missing values and encodes categorical variables for the Titanic dataset.
No data leakage: statistics are computed on the full dataset before splitting.
"""

import pandas as pd
import numpy as np


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using training-safe statistics.

    Strategy:
        - Age      → median
        - Embarked → mode
        - Fare     → median
        - Cabin    → drop column (too many nulls)

    Args:
        df (pd.DataFrame): Raw Titanic DataFrame.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    df = df.copy()

    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    if "Cabin" in df.columns:
        df.drop(columns=["Cabin"], inplace=True)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features:
        - Sex      → label encoding  (male=1, female=0)
        - Embarked → one-hot encoding (drop_first=True to avoid multicollinearity)

    Args:
        df (pd.DataFrame): DataFrame after missing-value handling.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns.
    """
    df = df.copy()

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 1, "female": 0}).astype(int)

    if "Embarked" in df.columns:
        embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked", drop_first=True)
        df = pd.concat([df.drop(columns=["Embarked"]), embarked_dummies], axis=1)

    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that carry no predictive signal or leak information.

    Dropped: Name, Ticket (high-cardinality text columns).

    Args:
        df (pd.DataFrame): Encoded DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.copy()
    cols_to_drop = [c for c in ["Name", "Ticket"] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline: imputation → encoding → column cleanup.

    Args:
        df (pd.DataFrame): Raw Titanic DataFrame.

    Returns:
        pd.DataFrame: Fully preprocessed DataFrame ready for feature engineering.
    """
    df = handle_missing_values(df)
    df = encode_categoricals(df)
    df = drop_unused_columns(df)
    return df
