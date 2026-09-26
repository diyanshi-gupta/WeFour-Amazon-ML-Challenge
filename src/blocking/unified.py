import pandas as pd
import re
from typing import Dict, Set, List

def get_address_tokens(address: str) -> set:
    if pd.isna(address) or not address:
        return set()
    address = str(address).lower()
    return set(re.findall(r'\b\w+\b', address))

def get_numeric_address_tokens(address: str) -> set:
    if pd.isna(address) or not address:
        return set()
    # Looking for PIN/ZIP codes or numeric tokens of length 3-6
    return set(re.findall(r'\b\d{3,6}\b', str(address)))

def address_jaccard_similarity(tokens1: set, tokens2: set) -> float:
    if not tokens1 and not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union) if len(union) > 0 else 0.0

def unify_and_corroborate_candidates(
    candidate_dicts: List[Dict[str, Set[str]]],
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    jaccard_threshold: float = 0.2
) -> pd.DataFrame:
    """
    Takes multiple candidate mappings (e.g. from token blocking, rule-based, TF-IDF), 
    unions them per S1 entity, and produces a consolidated candidate pairs DataFrame.
    
    IMPORTANT: Adds a `has_address_corroboration` boolean flag. This flag is True if:
      - The candidate pair shares at least one PIN/numeric address token
      - OR has address Jaccard similarity >= a threshold (default 0.2)
    This flags 'name-only' matches so downstream logic can treat them with caution.
    """
    # 1. Union candidates
    all_s1_ids = set()
    for d in candidate_dicts:
        all_s1_ids.update(d.keys())
        
    union_dict = {s1_id: set() for s1_id in all_s1_ids}
    for d in candidate_dicts:
        for s1_id, cands in d.items():
            union_dict[s1_id].update(cands)
            
    # Prepare DataFrames for quick lookup
    s2_s3_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    def get_addr(row):
        return str(row.get("clean_address", row.get("business_address", row.get("address", ""))))
        
    s1_address_map = {}
    for _, row in s1_df.iterrows():
        addr = get_addr(row)
        s1_address_map[row["entity_id"]] = {
            "tokens": get_address_tokens(addr),
            "num_tokens": get_numeric_address_tokens(addr)
        }
        
    s2_s3_address_map = {}
    for _, row in s2_s3_df.iterrows():
        addr = get_addr(row)
        s2_s3_address_map[row["entity_id"]] = {
            "tokens": get_address_tokens(addr),
            "num_tokens": get_numeric_address_tokens(addr)
        }

    # Build the final list of pairs
    records = []
    for s1_id, s2_s3_ids in union_dict.items():
        s1_info = s1_address_map.get(s1_id, {"tokens": set(), "num_tokens": set()})
        for match_id in s2_s3_ids:
            s2_s3_info = s2_s3_address_map.get(match_id, {"tokens": set(), "num_tokens": set()})
            
            # Check numeric match
            shared_nums = s1_info["num_tokens"].intersection(s2_s3_info["num_tokens"])
            has_num_match = len(shared_nums) > 0
            
            # Check Jaccard
            jaccard = address_jaccard_similarity(s1_info["tokens"], s2_s3_info["tokens"])
            has_jaccard_match = jaccard >= jaccard_threshold
            
            corroborated = has_num_match or has_jaccard_match
            
            records.append({
                "s1_entity_id": s1_id,
                "s2_s3_entity_id": match_id,
                "has_address_corroboration": corroborated
            })
            
    return pd.DataFrame(records)
