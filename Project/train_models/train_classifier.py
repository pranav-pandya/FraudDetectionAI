"""
Train a fraud classification model using transaction data and save the model & encoders.
    - Creates a models directory if it does not already exists
    - Reads input data from data/training.csv
    - Encode features and train XGBo0st classifier
    - Save the trained model

Parameters:
    None

Returns:
    None
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os

def train_and_save_model():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv("data/training.csv").fillna("Unknown")
    features = ["txn_type", "device_type", "status", "customer_type", "amount"]
    x = df[features]
    y = df["fraud_type"]

    # Encode categorical features
    encoder = OneHotEncoder(handle_unknown="ignore")
    x_encoded = encoder.fit_transform(x)

    # Encode target string labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x_encoded, y_encoded, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
    )
    model.fit(x_train, y_train)

    joblib.dump(model, "models/fraud_classifier.pkl")
    joblib.dump(encoder, "models/encoder.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    # print("XGBoost classifier, encoder, and label encoder saved successfully.")

if __name__ == "__main__":
    train_and_save_model()