import os
import fitz  # PyMuPDF
import pandas as pd
from docx import Document as DocxReader

def extract_text_from_file(file_path: str) -> str:
    """
    Identifies file type and uses the correct strategy to pull text.
    Referred by: services/indexer.py
    """
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            with fitz.open(file_path) as doc:
                for page in doc: 
                    text += page.get_text()
        elif ext == ".docx":
            doc = DocxReader(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext in [".xlsx", ".xls"]:
            df_dict = pd.read_excel(file_path, sheet_name=None)
            for sheet in df_dict.values():
                text += sheet.to_csv(index=False, sep=' ')
        return text
    except Exception as e:
        print(f"Extraction Error: {e}")
        return ""