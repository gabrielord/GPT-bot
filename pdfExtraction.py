import pdfplumber 
import os
from pdfminer.high_level import extract_text
from io import BytesIO

def extract_text_from_pdf(file):
    file_stream = BytesIO(file.getvalue())
    return extract_text(file_stream)
