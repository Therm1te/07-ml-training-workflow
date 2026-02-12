import pandas as pd

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()
    return df
