import pytest
from src.normalize.suffix_map import apply_suffix_map

def test_apply_suffix_map_basic_expansions():
    assert apply_suffix_map("acme pvt ltd") == "acme private limited"
    assert apply_suffix_map("global corp inc") == "global corporation incorporated"

def test_apply_suffix_map_word_boundary_safety():
    # Verifies "co" inside "costco" is untouched
    assert apply_suffix_map("costco co") == "costco company"
    # Verifies "assoc" inside "associates" or other words doesn't get double replaced if it was a substring
    assert apply_suffix_map("tech solutions") == "technology solutions"

def test_apply_suffix_map_special_chars():
    assert apply_suffix_map("smith & sons") == "smith and sons"

def test_apply_suffix_map_edge_cases():
    assert apply_suffix_map("") == ""
    assert apply_suffix_map("   ") == ""
    assert apply_suffix_map(None) == ""
    assert apply_suffix_map(123) == ""
