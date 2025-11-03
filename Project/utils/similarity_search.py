"""
similarity_search.py: Similarity search for finding the matched advisory section

Parameters:
    location (str): The location to search for

Returns:
    tuple: A tuple containing the matched state, section text, and escalation info
"""
def get_matched_advisory_section(location: str):
    matched_state = location
    section_text = f"Matched rules found for {location}: Review KYC processes and account linkage patterns."
    escalation_info = "Escalation contact: frauddesk@bank.com within 48 hours."
    return matched_state, section_text, escalation_info