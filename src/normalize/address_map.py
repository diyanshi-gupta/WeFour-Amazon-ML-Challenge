from typing import Any

ADDRESS_ABBR_MAP: dict[str, str] = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "apt": "apartment",
    "blvd": "boulevard",
    "ave": "avenue",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "ste": "suite",
    "fl": "floor",
    "flr": "floor",
    "bldg": "building",
    "pkwy": "parkway",
    "opp": "opposite",
    "nr": "near",
    "ext": "extension",
    "ind": "industrial",
    "est": "estate"
}

def apply_address_map(text: Any) -> str:
    """
    Replaces common address token abbreviations with their canonical forms.
    Matches strictly on whole tokens to ensure word-boundary safety.
    Does not attempt any geocoding or landmark resolution.
    
    Args:
        text (Any): The input string to normalize.
        
    Returns:
        str: The normalized string with expanded address abbreviations.
    """
    # Gracefully handle None, non-strings, or empty strings
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Tokenize the text by whitespace
    tokens = text.split()
    
    # Replace matched abbreviation tokens (case-insensitive lookup), otherwise keep original
    normalized_tokens = [ADDRESS_ABBR_MAP.get(token.lower(), token) for token in tokens]
    
    # Rejoin tokens into a space-delimited string
    return " ".join(normalized_tokens)
