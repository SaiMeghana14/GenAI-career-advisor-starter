import re
import fitz  # PyMuPDF for PDF parsing

# Extract text from PDF
def read_pdf_text(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Simple skill extractor (extend with NLP if needed)
def extract_skills(text: str):
    skills = [
        "Python", "C", "C++", "Java", "JavaScript", "SQL",
        "IoT", "Machine Learning", "Deep Learning",
        "Data Science", "Cloud", "Networking", "MATLAB",
        "Communication", "Git", "Embedded Systems"
    ]
    found = []
    for skill in skills:
        if re.search(rf"\b{skill}\b", text, re.IGNORECASE):
            found.append(skill)

    found = list(set(found))  # remove duplicates
    joined = ", ".join(found) if found else ""
    return found, joined

