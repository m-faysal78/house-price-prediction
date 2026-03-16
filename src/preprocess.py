import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess(path):

    df = pd.read_csv(path)

    X = df.drop("Price", axis=1)
    y = df["Price"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
