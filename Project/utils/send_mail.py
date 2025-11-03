"""
This automates the process of sending fraud advisory email to specific branch contacts

Required environment variables:
    - SENDER_EMAIL: The sender's Gmail address
    - SENDER_PASSWORD: The Gmail password for secure SMTP login
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import fitz  # PyMuPDF
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("SENDER_EMAIL")  # e.g. "yourmail@gmail.com"
SENDER_PASSWORD = os.getenv("SENDER_APP_PASSWORD")  # app password for Gmail

"""
Extract contact details from 'branch_rules.pdf' for the given branch.

Parameters:
    branch_name (str): The specifid branch name
    pdf_path: The PDF path to find and extract the contact details of the specified branch

Returns:
    contact_info: contact information of the specific branch
"""
def extract_branch_contact(branch_name: str, pdf_path="data/branch_rules.pdf"):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        # Simple parsing logic: find block starting with branch_name and extract lines after
        idx = text.find(branch_name)
        if idx == -1:
            return {}

        snippet = text[idx: idx+600]  # read ahead 600 chars approx
        contact_info = {}
        for line in snippet.split('\n'):
            line = line.strip()
            if any(k in line.lower() for k in ['name:', 'role:', 'sla:', '@']):
                # Parse key:value
                if 'name:' in line.lower():
                    contact_info['name'] = line.split(':',1)[1].strip()
                if 'role:' in line.lower():
                    contact_info['role'] = line.split(':',1)[1].strip()
                if 'sla:' in line.lower():
                    contact_info['sla'] = line.split(':',1)[1].strip()
                if '@' in line:
                    # try to get email address out (use simple heuristic)
                    words = line.split()
                    for w in words:
                        if '@' in w and '.' in w:
                            contact_info['email'] = w.strip()
                            break
        return contact_info
    except Exception as e:
        logging.error(f"Error reading PDF for contacts: {e}")
        return {}

"""
Sends an email advisory with content to the concerned branch email address extracted from branch_rules.pdf.

Parameters:
    branch: The specifid branch name
    content: The PDF path to find and extract the contact details of the specified branch

Returns:
    This function does not return a value
"""
def send_advisory_email(branch: str, content: str) -> None:
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        raise ValueError("Sender email or app password missing in environment variables")

    contact = extract_branch_contact(branch)
    if not contact or 'email' not in contact:
        raise ValueError(f"Could not find contact email for branch {branch} in PDF")
    
    recipient = contact['email']

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient
    msg['Subject'] = f"Fraud Advisory for Branch {branch}"

    # Create email body with salutation and advisory content
    body = f"Dear {contact.get('name', 'Branch Head')},\n\n"
    body += content + "\n\n"
    body += f"Role: {contact.get('role', 'N/A')}\n"
    body += f"SLA: {contact.get('sla', 'N/A')}\n\n"
    body += "Regards,\nFraud Intelligence Team"

    msg.attach(MIMEText(body, 'plain'))

    # Send email via SMTP
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        logging.info(f"Advisory email sent to {recipient} for branch {branch}")
    except smtplib.SMTPException as e:
        logging.error(f"SMTP error: {e}")