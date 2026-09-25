from typing import Any, Tuple
from src.normalize.clean import basic_clean
from src.normalize.suffix_map import apply_suffix_map
from src.normalize.address_map import apply_address_map

def normalize_name(text: Any) -> str:
    """
    Applies foundational cleaning followed by legal suffix normalization
    specifically tailored for business names.
    
    Args:
        text (Any): Raw business name.
        
    Returns:
        str: Normalized business name.
    """
    cleaned = basic_clean(text)
    return apply_suffix_map(cleaned)

def normalize_address(text: Any) -> str:
    """
    Applies foundational cleaning followed by address abbreviation expansion
    specifically tailored for business addresses.
    
    Args:
        text (Any): Raw business address.
        
    Returns:
        str: Normalized business address.
    """
    cleaned = basic_clean(text)
    return apply_address_map(cleaned)

def normalize_record(name: Any, address: Any) -> Tuple[str, str]:
    """
    End-to-end normalization for a full business record.
    
    Args:
        name (Any): Raw business name.
        address (Any): Raw business address.
        
    Returns:
        Tuple[str, str]: A tuple of (normalized_name, normalized_address).
    """
    return normalize_name(name), normalize_address(address)
