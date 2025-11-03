"""
This function loads a pre-trained anamoly detection model

Parameters:
    df (pandas.DataFrame): The input DataFrame containing the transaction data.

Returns:
    df (pandas.DataFrame): Same dataframe with an additional column is_anomaly with True or False.
"""
import joblib
import pandas as pd
import os

def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    os.makedirs("output", exist_ok=True)
    model = joblib.load("models/anomaly_model.pkl")
    X = df[["amount"]].fillna(df["amount"].median())
    preds = model.predict(X)
    df['is_anomaly'] = preds == -1
    df.to_csv("output/anomaly_output.csv", index=False)
    return df
