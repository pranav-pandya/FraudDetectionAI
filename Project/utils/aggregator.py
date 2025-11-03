"""
aggregator.py: Aggregation logic for grouping uding with the Pandas dataframe

Parameters:
    df (pandas.DataFrame): The input DataFrame containing the fraud data.

Returns:
    dict: A dictionary containing the fraud summary for each branch.
"""
def group_fraud_summary(df):
    if "predicted_fraud" not in df.columns:
        return {}
    return df[df["predicted_fraud"] == 1].groupby("branch_code")["predicted_fraud"].count().to_dict()