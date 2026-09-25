from typing import Any

LEGAL_SUFFIX_MAP: dict[str, str] = {
    "pvt": "private",
    "pvtltd": "private limited",
    "ltd": "limited",
    "corp": "corporation",
    "corporation": "corporation",
    "inc": "incorporated",
    "co": "company",
    "comp": "company",
    "llp": "limited liability partnership",
    "plc": "public limited company",
    "gmbh": "gmbh",
    "sa": "sa",
    "assoc": "associates",
    "tech": "technology",
    "solutions": "solutions",
    "services": "services",
    "&": "and"
}

def apply_suffix_map(text: Any) -> str:
    """
    Replaces common business legal suffix abbreviations with their canonical full forms.
    Matches strictly on whole tokens to ensure word-boundary safety.
    
    Args:
        text (Any): The input string to normalize.
        
    Returns:
        str: The normalized string with expanded suffixes.
    """
    # Gracefully handle None, non-strings, or empty strings
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Tokenize the text by whitespace
    tokens = text.split()
    
    # Replace matched abbreviation tokens, otherwise keep the original token
    normalized_tokens = [LEGAL_SUFFIX_MAP.get(token.lower(), token) for token in tokens]
    
    # Rejoin tokens into a space-delimited string
    return " ".join(normalized_tokens)
