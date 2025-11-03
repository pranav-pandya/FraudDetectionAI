"""
pdf_reader.py: PDF reader for extracting rules from PDF files

Parameters:
    pdf_path (str): Path to the PDF file

Returns:
    dict: A dictionary containing the rules for each region/state
"""
import fitz  # PyMuPDF

def extract_rules_from_pdf(pdf_path: str) -> dict:
    # Extracts text from PDF grouped by region/state name
    doc = fitz.open(pdf_path)
    rules_dict = {}
    current_region = None
    for page in doc:
        blocks = page.get_text("blocks")  # Extract text blocks
        for block in blocks:
            text = block[4].strip()
            lines = text.split('\n')
            for line in lines:
                if not line:
                    continue
                # Identify region titles roughly by line length and capitalization
                if line.isupper() or line.istitle():
                    current_region = line.strip()
                    rules_dict[current_region] = []
                elif current_region:
                    rules_dict[current_region].append(line)
    # Join lines per region for easy searching later
    for key in rules_dict:
        rules_dict[key] = "\n".join(rules_dict[key])
    return rules_dict