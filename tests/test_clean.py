import pytest
import math
import numpy as np
from src.normalize.clean import basic_clean

def test_basic_clean_nulls():
    assert basic_clean(None) == ""
    assert basic_clean(float('nan')) == ""
    assert basic_clean(np.nan) == ""

def test_basic_clean_empty_and_whitespace():
    assert basic_clean("") == ""
    assert basic_clean("   ") == ""

def test_basic_clean_special_chars():
    assert basic_clean("!!!@#$%^&*()") == ""

def test_basic_clean_casing_and_punctuation():
    assert basic_clean("  ACME   CORP.  ") == "acme corp"
    assert basic_clean("Company\tName\n123!") == "company name 123"

def test_basic_clean_numeric_retention():
    assert basic_clean("110001 New Delhi #42") == "110001 new delhi 42"
