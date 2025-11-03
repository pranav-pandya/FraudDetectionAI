"""
This function creates a decidce-focussed fraud summary using Google Gemini API
It analyzes a dataframe containing the predicted fraudulent transaction and generates a
concise summary

Parameters:
    df (pandas.DataFrame): The input DataFrame containing the transaction data.
    api_key: API key for Google Gemini API

Returns:
    str: A device focussed fraud summary string
"""
import google.generativeai as genai
import pandas as pd
import os
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_device_summary(df: pd.DataFrame, api_key: str) -> str:
    """Generates a device-focused fraud summary using Gemini."""
    print("Generating device fraud summary...")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash') 

    # Filter for frauds
    fraud_df = df[df['predicted_fraud'] == 1]
    
    if fraud_df.empty:
        return "No fraudulent activities detected to analyze device patterns."

    # --- START OF FIX ---

    # Aggregate data for the prompt
    device_distribution = fraud_df['device_type'].value_counts().to_dict()
    
    # Get patterns: Top 2 transaction types per device (using 'txn_type' as 'fraud_type' is missing)
    top_fraud_types_by_device = fraud_df.groupby('device_type')['txn_type'].apply(
        lambda x: x.value_counts().nlargest(2).to_dict()
    ).to_dict()
    
    # Robustly select available columns for the example to prevent KeyError
    available_cols = fraud_df.columns
    # Define the columns we'd *like* to show Gemini, using the correct names from the CSV
    desired_cols = ['device_type', 'txn_type', 'amount', 'location', 'timestamp'] 
    
    # Find which of our desired columns are actually in the DataFrame
    example_cols = [col for col in desired_cols if col in available_cols]
    
    # Select only those available columns
    examples = fraud_df[example_cols].head(5).to_string()

    prompt = f"""
    Analyze the following device fraud data for a financial region.

    1. Total Fraud Counts by Device Type:
    {device_distribution}

    2. Top Transaction Types (txn_type) Observed per Device:
    {top_fraud_types_by_device}

    3. Example Fraudulent Transactions (from available data like 'timestamp' and 'amount'):
    {examples}

    Based on this data, generate a short, structured "Device Risk Summary" paragraph.
    Your summary must identify:
    - Which devices are most exploited (e.g., "Mobile" or "ATM").
    - Any apparent vulnerability patterns (e.g., "Web Banking is primarily used for 'Transfer'" or "ATM fraud peaks around 'timestamp' 02:00:00").
    - Succinct, actionable mitigations or policy suggestions for the IT/Security team (e.g., "Implement step-up authentication for web transfers > $1000" or "Increase physical monitoring of ATMs at specific branches").

    The tone should be advisory and technical.
    """
    
    # --- END OF FIX ---
    
    try:
        response = model.generate_content(prompt)
        summary = response.text
        
        # Save the summary as requested
        os.makedirs('output', exist_ok=True)
        with open('output/device_policy_suggestions.txt', 'w') as f:
            f.write(summary)
            
        return summary
    except Exception as e:
        logging.error(f"Error generating device summary: {e}")
        return f"Error generating summary: {e}"