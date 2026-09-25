import re
import unicodedata
import pandas as pd


def remove_accents(text: str) -> str:
    """Normalize Unicode characters, e.g. é -> e."""
    if pd.isna(text):
        return ""

    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(
        c for c in nfkd
        if not unicodedata.combining(c)
    )


def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, and standardize spaces."""
    if pd.isna(text):
        return ""

    text = remove_accents(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(name: str) -> str:
    """Clean business names and standardize common legal suffixes."""
    text = normalize_text(name)

    replacements = {
        r"\bcorp\b": "corporation",
        r"\binc\b": "incorporated",
        r"\bltd\b": "limited",
        r"\bpvt\b": "private",
        r"\bllc\b": "limited liability company",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    return text


def normalize_address(address: str) -> str:
    """Clean business addresses and standardize common abbreviations."""
    text = normalize_text(address)

    replacements = {
        r"\brd\b": "road",
        r"\bst\b": "street",
        r"\bave\b": "avenue",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    return text
