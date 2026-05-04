"""
data_loader.py
Loads the Titanic dataset and provides a train/test split utility.
Falls back to a procedurally generated dataset with realistic Titanic
distributions when network access is unavailable.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def _generate_titanic_dataset(n: int = 891, seed: int = 0) -> pd.DataFrame:
    """
    Generate a synthetic dataset with statistical properties matching the
    real Titanic training set (used as an offline fallback).
    """
    rng = np.random.RandomState(seed)
    pclass = rng.choice([1, 2, 3], n, p=[0.242, 0.206, 0.552])
    sex = rng.choice(["male", "female"], n, p=[0.647, 0.353])
    age_raw = rng.normal(29.7, 14.5, n).clip(0.42, 80.0)
    age = np.where(rng.rand(n) < 0.198, np.nan, age_raw)
    sibsp = rng.choice([0, 1, 2, 3, 4, 5, 8], n,
                       p=[0.682, 0.234, 0.031, 0.016, 0.012, 0.008, 0.017])
    parch = rng.choice([0, 1, 2, 3, 4, 5, 6], n,
                       p=[0.760, 0.132, 0.090, 0.007, 0.004, 0.004, 0.003])
    fare_base = np.where(pclass == 1, 84.2, np.where(pclass == 2, 20.7, 13.7))
    fare = np.abs(rng.normal(fare_base, fare_base * 0.5))
    embarked = rng.choice(["S", "C", "Q"], n, p=[0.724, 0.188, 0.086]).astype(object)
    embarked[rng.rand(n) < 0.002] = None

    surv_prob = (
        0.15
        + 0.50 * (sex == "female")
        + 0.20 * (pclass == 1)
        - 0.10 * (pclass == 3)
        + rng.normal(0, 0.05, n)
    ).clip(0.05, 0.95)
    survived = (rng.rand(n) < surv_prob).astype(int)

    titles_m = ["Mr.", "Dr.", "Rev.", "Major.", "Col."]
    titles_f = ["Mrs.", "Miss.", "Ms.", "Lady."]
    names = []
    for i in range(n):
        last = f"Passenger{i}"
        if sex[i] == "male":
            title = titles_m[rng.randint(0, len(titles_m))]
        else:
            title = titles_f[rng.randint(0, len(titles_f))]
        names.append(f"{last}, {title} Name{i}")

    tickets = [f"T{rng.randint(10000, 99999)}" for _ in range(n)]
    cabins = np.where(rng.rand(n) < 0.23,
                      [f"{rng.choice(list('ABCDE'))}{rng.randint(1, 150)}" for _ in range(n)],
                      None)

    return pd.DataFrame({
        "Survived": survived,
        "Pclass": pclass,
        "Name": names,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": tickets,
        "Fare": fare,
        "Cabin": cabins,
        "Embarked": embarked,
    })


def load_titanic() -> pd.DataFrame:
    """
    Load the Titanic dataset.

    Priority order:
        1. seaborn.load_dataset (requires internet)
        2. CSV download from GitHub (requires internet)
        3. Procedurally generated synthetic dataset (always works offline)

    Returns:
        pd.DataFrame: Titanic-style dataset with Kaggle-standard column names.
    """
    rename_map = {
        "survived": "Survived",
        "pclass": "Pclass",
        "sex": "Sex",
        "age": "Age",
        "sibsp": "SibSp",
        "parch": "Parch",
        "fare": "Fare",
        "embarked": "Embarked",
        "name": "Name",
        "ticket": "Ticket",
        "cabin": "Cabin",
    }

    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        keep = [c for c in rename_map.values() if c in df.columns]
        return df[keep].copy().reset_index(drop=True)
    except Exception:
        pass

    try:
        url = (
            "https://raw.githubusercontent.com/datasciencedojo/datasets/"
            "master/titanic.csv"
        )
        return pd.read_csv(url).reset_index(drop=True)
    except Exception:
        pass

    return _generate_titanic_dataset().reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    target: str = "Survived",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split a processed DataFrame into train and test sets.

    Args:
        df (pd.DataFrame): Processed feature + target DataFrame.
        target (str): Target column name.
        test_size (float): Proportion for the test set.
        random_state (int): Reproducibility seed.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
