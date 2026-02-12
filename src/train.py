import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess
import os

def train_model():
    # Assuming run from root
    data_path = 'data/sample.csv'
    if not os.path.exists(data_path):
        # Fallback if run from src
        data_path = '../data/sample.csv'

    df = load_and_preprocess(data_path)
    X = df[['id']]
    y = df['value']
    
    # Simple training for demonstration
    model = LogisticRegression()
    model.fit(X, y)
    
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f"Model Accuracy: {accuracy}")

if __name__ == "__main__":
    train_model()
