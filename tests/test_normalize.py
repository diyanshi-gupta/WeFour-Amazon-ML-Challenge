import pytest
import math
from src.normalize.pipeline import normalize_name, normalize_address, normalize_record

def test_full_pipeline_regression_suite():
    # 1. Business Legal Suffix Expansion
    assert normalize_name("ACME Pvt. Ltd.") == "acme private limited"
    
    # 2. Address Token Expansion
    assert normalize_address("42 Wall St. Apt #3") == "42 wall street apartment 3"
    
    # 3. Special Character & Ampersand Handling
    assert normalize_name("Johnson & Johnson, Inc.") == "johnson and johnson incorporated"
    
    # 4. Digit & PIN Code Retention
    assert normalize_address("New Delhi 110001") == "new delhi 110001"
    
    # 5. Extra Spaces, Tabs, and Newlines
    assert normalize_name("   Global   Tech\n  Services  ") == "global technology services"
    
    # 6. Landmark Preservation (No external geocoding)
    assert normalize_address("Near SBI ATM, Station Rd") == "near sbi atm station road"
    
    # 7. Word Boundary Safety
    assert normalize_name("First State Corp") == "first state corporation"
    
    # 8. Combined Name and Address Tuple Pipeline
    assert normalize_record("Costco Co.", "100 Pkwy Dr.") == ("costco company", "100 parkway drive")
    
    # 9. Missing / Null Values
    assert normalize_record(None, float('nan')) == ("", "")
    
    # 10. Empty String inputs
    assert normalize_record("", "") == ("", "")
