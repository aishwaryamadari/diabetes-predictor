"""
file_parser.py - Routes uploaded files to appropriate parser and extracts text/data.
"""
import io
import os
import logging
import pandas as pd
from werkzeug.datastructures import FileStorage

logger = logging.getLogger(__name__)

def parse_file(file: FileStorage) -> list[dict]:
    """
    Parses an uploaded file based on its extension.
    Returns a list of dictionaries (raw rows for CSV/Excel, or extracted features for Text/PDF/Images).
    """
    filename = file.filename.lower()
    ext = os.path.splitext(filename)[1]
    
    if ext == '.csv':
        return _parse_csv(file)
    elif ext in ['.xls', '.xlsx']:
        return _parse_excel(file)
    elif ext == '.pdf':
        return _parse_pdf(file)
    elif ext == '.docx':
        return _parse_docx(file)
    elif ext == '.txt':
        return _parse_txt(file)
    elif ext in ['.png', '.jpg', '.jpeg']:
        return _parse_image(file)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def _parse_csv(file: FileStorage) -> list[dict]:
    content = file.read().decode('utf-8', errors='ignore')
    df = pd.read_csv(io.StringIO(content))
    return df.to_dict('records')

def _parse_excel(file: FileStorage) -> list[dict]:
    df = pd.read_excel(file)
    return df.to_dict('records')

def _parse_txt(file: FileStorage) -> list[dict]:
    from app.services.text_extractor import extract_features_from_text
    content = file.read().decode('utf-8', errors='ignore')
    return extract_features_from_text(content)

def _parse_pdf(file: FileStorage) -> list[dict]:
    import pdfplumber
    from app.services.text_extractor import extract_features_from_text
    
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                
    if not text.strip():
        # Scanned PDF without OCR
        raise ValueError("No text found in PDF. If this is a scanned report, please upload it as an image (JPG/PNG).")
        
    return extract_features_from_text(text)

def _parse_docx(file: FileStorage) -> list[dict]:
    import docx
    from app.services.text_extractor import extract_features_from_text
    
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return extract_features_from_text(text)

def _parse_image(file: FileStorage) -> list[dict]:
    import pytesseract
    from PIL import Image
    from app.services.text_extractor import extract_features_from_text
    
    # Read image using Pillow
    img = Image.open(file)
    
    # Basic preprocessing to improve OCR accuracy
    # Convert to grayscale
    img = img.convert('L')
    # Simple thresholding
    img = img.point(lambda p: p > 128 and 255)
    
    try:
        text = pytesseract.image_to_string(img)
    except Exception as e:
        logger.error(f"OCR Error: {str(e)}")
        raise ValueError("Tesseract OCR is not installed or configured correctly on this system. "
                         "Please install Tesseract or upload digital files instead.")
        
    return extract_features_from_text(text)
