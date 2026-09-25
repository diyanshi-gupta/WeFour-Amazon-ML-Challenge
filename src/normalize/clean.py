import re
from typing import Any

def basic_clean(text: Any) -> str:
    """
    Cleans text by handling nulls, lowercasing, removing punctuation,
    and collapsing whitespace. Keeps only alphanumeric characters (a-z, 0-9)
    and standard whitespace.

    Args:
        text (Any): The input text to clean. Can be string, None, NaN, etc.

    Returns:
        str: The cleaned, normalized string.
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # Lowercase the string
    text = text.lower()
    
    # Convert tabs, newlines, and carriage returns into single spaces
    text = re.sub(r'[\t\n\r]', ' ', text)
    
    # Strip all punctuation/special characters, keeping only a-z, 0-9 and whitespace
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Collapse contiguous whitespace and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
