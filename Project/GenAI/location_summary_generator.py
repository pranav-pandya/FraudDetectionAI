"""
Generates a fraud summary for a specific region using Google Gemni API

Parameters:
    region_name: The region name for which the summary is generated.
    df (pandas.DataFrame): The input DataFrame containing the transaction data.
    api_key: API key for Google Gemini API

Returns:
    str: Generated summary as a string.
"""
import google.generativeai as genai
import pandas as pd
import os
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_region_summary(region_name: str, df: pd.DataFrame, api_key: str) -> str:
    """Generates a fraud summary for a region using Gemini."""
    logging.info(f"Generating summary for {region_name}...")

    #get region name from file to get prompt context
    region_name = region_name.replace("_Region.csv","")
    #print(f"Region name extracted: {region_name}")
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel('gemini-2.5-flash')

    # Filter for frauds in the specified region
    fraud_df = df[(df['predicted_fraud'] == 1) & (df['region'] == region_name)]
    
    if fraud_df.empty:
        return f"No fraudulent activities detected in the {region_name} region."

    # Aggregate data for the prompt
    top_branches = fraud_df['branch_code'].value_counts().nlargest(3).to_dict()
    fraud_types = fraud_df['fraud_type'].value_counts().to_dict()
    top_devices = fraud_df['device_type'].value_counts().nlargest(3).to_dict()
    
    # Get a few concrete examples
    examples = fraud_df.head(3).to_string()

    prompt = f"""
    Generate an executive summary for fraud activity in the {region_name} region.
    Based on the following data:

    1. Top 3 Branches with Highest Fraud Counts:
    {top_branches}

    2. Prevalent Fraud Types:
    {fraud_types}

    3. Top Devices Used for Fraudulent Transactions:
    {top_devices}

    4. Concrete Examples of Fraudulent Transactions:
    {examples}

    Please provide a concise summary covering the key trends, highlight the riskiest branches,
    and suggest potential areas for investigation. The tone should be formal and advisory.
    """
    
    try:
        response = model.generate_content(prompt)
        summary = response.text
        
        # Save the summary
        os.makedirs('output', exist_ok=True)
        with open('output/region_summary.txt', 'w') as f:
            f.write(summary)
            
        return summary
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return "Error generating the summary"