import pytest
from src.normalize.address_map import apply_address_map

def test_apply_address_map_basic_expansions():
    assert apply_address_map("123 main st apt 4b") == "123 main street apartment 4b"
    assert apply_address_map("industrial est opp station rd") == "industrial estate opposite station road"

def test_apply_address_map_word_boundary_safety():
    # Verifies "st" inside "state" and "first" is untouched
    assert apply_address_map("state street first ave") == "state street first avenue"

def test_apply_address_map_no_geocoding_rule():
    # Keep landmarks like "near sbi atm" intact, but expand literal abbreviations like "rd"
    assert apply_address_map("near sbi atm rd") == "near sbi atm road"

def test_apply_address_map_edge_cases():
    assert apply_address_map("") == ""
    assert apply_address_map("   ") == ""
    assert apply_address_map(None) == ""
    assert apply_address_map(123) == ""
