"""
Normalization package for text cleaning and harmonization.
"""

from typing import Union
import pandas as pd


def normalize_name(text: Union[str, float, None]) -> str:
    """Basic text normalization for business names."""
    if text is None or pd.isna(text):
        return ""
    text_str = str(text).lower().strip()
    cleaned = "".join(c for c in text_str if c.isalnum() or c.isspace())
    return " ".join(cleaned.split())


def normalize_address(text: Union[str, float, None]) -> str:
    """Basic text normalization for business addresses."""
    if text is None or pd.isna(text):
        return ""
    text_str = str(text).lower().strip()
    cleaned = "".join(c for c in text_str if c.isalnum() or c.isspace())
    return " ".join(cleaned.split())


def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Applies normalize_name and normalize_address to a DataFrame."""
    df_clean = df.copy()
    if "business_name" in df_clean.columns:
        df_clean["normalized_name"] = df_clean["business_name"].apply(normalize_name)
    else:
        df_clean["normalized_name"] = ""

    if "business_address" in df_clean.columns:
        df_clean["normalized_address"] = df_clean["business_address"].apply(normalize_address)
    else:
        df_clean["normalized_address"] = ""

    return df_clean


__all__ = ["normalize_name", "normalize_address", "apply_normalization"]
