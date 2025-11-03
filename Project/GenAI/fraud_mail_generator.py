"""
Generates an advisory email for a branch regarding fraud incidents, combining fraud data 
with branch policies extracted form PDF and produces results using Google Gemni API

Parameters:
    location: The branch location for which to generate the advisory email.
    df (pandas.DataFrame): The input DataFrame containing the transaction and fraud data.
    api_key: API key for Google Gemini API

Returns:
    str: Generated advisory email content as a string.
"""
import os
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.pdf_reader import extract_rules_from_pdf
from utils.send_mail import extract_branch_contact


def generate_advisory_email(location: str, df: pd.DataFrame, api_key: str) -> str:

    if not api_key or api_key.lower == "dummy-api-key":
        output = f"**Error:** Invalid or dummy API key provided. Cannot generate advisory eamil for {location}."
        pd.DataFrame([{
            "branch": location,
            "region": df['region'].iloc[0] if 'region' in df.columns else '',
            "advisory_content": output
        }]).to_csv("output/fraud_action_advice.csv", index= False)
        return output
    try:
        contactinfo = extract_branch_contact("location")
        os.makedirs("output", exist_ok=True)
        filtered = df[df["location"] == location] if "location" in df else df
        total = len(filtered)
        fraud_counts = filtered["fraud_type"].value_counts().to_dict()

        pdf_rules = extract_rules_from_pdf("data/branch_rules.pdf")
        all_regions = list(pdf_rules.keys())
        all_texts = list(pdf_rules.values())
        # Fallback in case region not found exactly
        region_rules_text = pdf_rules.get(location, "No specific rules found for this location.")

        # Use cosine similarity to find closest matching region
        vectorizer = TfidfVectorizer().fit(all_regions)
        vectors = vectorizer.transform(all_regions + [location])
        similarity_scores = cosine_similarity(vectors[-1], vectors[:-1])
        best_match_idx = similarity_scores.argmax()
        matched_region = all_regions[best_match_idx]
        matched_rules = pdf_rules[matched_region]

        # --- Generate summary via Gemini ---
        fraud_summary_prompt = f"""
        You are a compliance communication officer.
        Write a formal email advisory for the {location} branch.

        Include:
        - Total fraud count: {total}
        - Fraud types and counts: {fraud_counts}
        - Most likely fraud pattern (based on data)
        - Policy rules from the following matched region: {matched_region}
        - Actionable recommendations from: {matched_rules}
        

        Tone: business-formal, clear, and directive.
        Include:
        1. Greeting
        2. Summary of findings
        3. Recommended actions
        4. Contact details from {contactinfo.get('name', 'N/A')} ({contactinfo.get('role', 'N/A')}, {contactinfo.get('sla', 'N/A')})
        5. Signature block "Fraud Intelligence System"

        Please do not include special characters like *, # in the generated content strictly
        """
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(fraud_summary_prompt)
        email_content = response.text.strip()

        # --- Save advisory email to CSV ---
        record = pd.DataFrame({
            "branch": [location],
            "region": [matched_region],
            "advisory_content": [email_content],
        })
        record.to_csv("output/fraud_action_advice.csv", index=False)

        return email_content
    except Exception as e:
        error_msg = f"Faile to generate advisory email for {location}: {str(e)}"
        print(error_msg)
        return f"**Error generating advisory email:** {error_msg}"
