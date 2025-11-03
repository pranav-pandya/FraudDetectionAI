"""
file_loader.py: File loader for loading the transaction data

Parameters:
    path (str): Path to the CSV file

Returns:
    pd.DataFrame: DataFrame containing the data from the CSV file
"""
import pandas as pd

def load_transaction_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)