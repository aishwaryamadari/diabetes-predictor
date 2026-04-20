"""
text_extractor.py - Smart NLP/Regex engine to extract medical features from unstructured text.
"""
import re
import logging

logger = logging.getLogger(__name__)

# Required features for the ML model
REQUIRED_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

def extract_features_from_text(text: str) -> list[dict]:
    """
    Parses unstructured text and extracts instances of patient features.
    Returns a list of dictionaries (one per patient block found).
    """
    # Normalize text (lowercase, remove extra spaces)
    text_clean = re.sub(r'\s+', ' ', text.lower())
    
    # Let's find all numbers that could match our features.
    # We'll use regex to find labels and the closest number following them.
    
    patterns = {
        "Pregnancies": r"(?:pregnanc(?:y|ies)|gravida|para)[\s:]*([0-9]{1,2})",
        "Glucose": r"(?:glucose|fbs|fasting blood sugar|blood sugar)[\s:]*([0-9]{2,3}(?:\.[0-9]+)?)",
        # For BP, if we see 120/80, we want the diastolic (80)
        "BloodPressure": r"(?:blood pressure|bp|diastolic)[\s:]*(?:[0-9]{2,3}\/)?([0-9]{2,3})",
        "SkinThickness": r"(?:skin thickness|triceps skinfold|skinfold)[\s:]*([0-9]{1,3}(?:\.[0-9]+)?)",
        "Insulin": r"(?:insulin|serum insulin)[\s:]*([0-9]{1,4}(?:\.[0-9]+)?)",
        "BMI": r"(?:bmi|body mass index)[\s:]*([0-9]{2}(?:\.[0-9]+)?)",
        "DiabetesPedigreeFunction": r"(?:diabetes pedigree|pedigree function|dpf)[\s:]*([0-9](?:\.[0-9]+)?)",
        "Age": r"(?:age|patient age)[\s:]*([0-9]{1,3})"
    }
    
    # We will find all matches for all patterns with their start index.
    matches = []
    for feature, pattern in patterns.items():
        for match in re.finditer(pattern, text_clean):
            matches.append({
                "feature": feature,
                "value": float(match.group(1)),
                "start": match.start()
            })
            
    # Sort matches by where they appear in the document
    matches.sort(key=lambda x: x["start"])
    
    # Group matches into patient records.
    # A new record starts when we see a feature we already have in the current record.
    records = []
    current_record = {}
    
    for match in matches:
        feat = match["feature"]
        val = match["value"]
        
        if feat in current_record:
            # We already have this feature, so save the current record and start a new one.
            records.append(current_record)
            current_record = {feat: val}
        else:
            current_record[feat] = val
            
    if current_record:
        records.append(current_record)
        
    return records
