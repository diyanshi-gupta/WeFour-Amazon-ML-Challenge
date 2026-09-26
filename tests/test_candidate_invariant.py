import pandas as pd
import pytest

def check_candidate_invariant(matching_results: pd.DataFrame, candidate_pairs: pd.DataFrame):
    """
    Asserts that every (s1_entity_id, matched_entity_id) in matching_results 
    also appears as a candidate pair in candidate_pairs.
    
    matching_results: DataFrame with ['s1_entity_id', 'matched_entity_id']
    candidate_pairs: DataFrame with ['s1_entity_id', 's2_s3_entity_id']
    """
    # Create a set of tuples for fast lookup
    candidates_set = set(zip(candidate_pairs['s1_entity_id'], candidate_pairs['s2_s3_entity_id']))
    
    missing_matches = []
    for _, row in matching_results.iterrows():
        s1_id = row['s1_entity_id']
        match_id = row['matched_entity_id']
        
        # Split matched_entity_id if it's a comma-separated list of IDs
        if pd.notna(match_id):
            matches = [m.strip() for m in str(match_id).split(',')]
            for m in matches:
                if (s1_id, m) not in candidates_set:
                    missing_matches.append((s1_id, m))
            
    assert len(missing_matches) == 0, f"Invariant violated! Found {len(missing_matches)} matches not in candidates. Example: {missing_matches[:5]}"

def test_candidate_invariant_valid():
    # Mock data
    candidate_pairs = pd.DataFrame({
        's1_entity_id': ['S1-1', 'S1-1', 'S1-2'],
        's2_s3_entity_id': ['S2-A', 'S3-B', 'S2-C']
    })
    
    # Valid matching results (subset of candidates)
    valid_matches = pd.DataFrame({
        's1_entity_id': ['S1-1', 'S1-2'],
        'matched_entity_id': ['S2-A, S3-B', 'S2-C']
    })
    
    check_candidate_invariant(valid_matches, candidate_pairs)

def test_candidate_invariant_invalid():
    candidate_pairs = pd.DataFrame({
        's1_entity_id': ['S1-1'],
        's2_s3_entity_id': ['S2-A']
    })
    
    # Invalid matching results (violates invariant)
    invalid_matches = pd.DataFrame({
        's1_entity_id': ['S1-1'],
        'matched_entity_id': ['S2-X']
    })
    
    with pytest.raises(AssertionError):
        check_candidate_invariant(invalid_matches, candidate_pairs)
