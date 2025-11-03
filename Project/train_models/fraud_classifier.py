"""
Classify transactions as fraudulent or legitimate using a pre-trained ML model

Parameters:
    df (pandas.DataFrame): The input DataFrame containing the transaction data.

Returns:
    df (pandas.DataFrame): The same DataFrame containing 2 additional columns
        - predicted_fraud
        - fraud_type
"""
import pandas as pd
import joblib
import os

def run_fraud_classification(df: pd.DataFrame) -> pd.DataFrame:
    os.makedirs("output", exist_ok=True)
    encoder = joblib.load("models/encoder.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    model = joblib.load("models/fraud_classifier.pkl")
    x = df[["txn_type", "device_type", "status", "customer_type", "amount"]].fillna("Unknown")
    x_encoded = encoder.transform(x)
    preds = model.predict(x_encoded)
    preds_labels = label_encoder.inverse_transform(preds)
    df['predicted_fraud'] = (preds != 0).astype(int)
    df['fraud_type'] = preds_labels
    df.to_csv("output/classified_frauds.csv", index=False)
    return df
