"""
Train an anamoly detection model on transaction data and save it for later use.
    - Creates a models directory if it does not already exists
    - Reads CSV file from data/training.csv
    - Write a serialized modle to file models/anomaly_model.pkl

Parameters:
    None

Returns:
    None
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

def train_and_save_anomaly_model():
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv("data/training.csv")
    X = df[["amount"]].fillna(df["amount"].median())

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)

    joblib.dump(model, "models/anomaly_model.pkl")
    # print("Anomaly detection model saved at models/anomaly_model.pkl")

if __name__ == "__main__":
    train_and_save_anomaly_model()