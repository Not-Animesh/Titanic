import pandas as pd
def load_titanic(path: str = "titanic.csv") -> pd.DataFrame:
    try:
        return pd.read_csv("titanic.csv")
    except FileNotFoundError:
        import seaborn as sns
        df = sns.load_dataset("titanic")
        return df.rename(columns={
            "survived": "Survived",
            "pclass": "Pclass",
            "sex": "Sex",
            "age": "Age",
            "sibsp": "SibSp",
            "parch": "Parch",
            "fare": "Fare",
            "embarked": "Embarked"
        })