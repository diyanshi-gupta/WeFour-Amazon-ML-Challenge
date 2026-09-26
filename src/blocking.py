# src/blocking.py
import pandas as pd
import re
from typing import Dict, Set

def generate_exact_name_candidates(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Groups S2 and S3 records matching S1 entities on clean_name + country."""
    s2_s3 = pd.concat([s2_df, s3_df], ignore_index=True)
    lookup = {}
    for _, row in s2_s3.iterrows():
        key = (row["clean_name"], row["country"])
        lookup.setdefault(key, set()).add(row["entity_id"])

    candidates = {}
    for _, row in s1_df.iterrows():
        key = (row["clean_name"], row["country"])
        candidates[row["entity_id"]] = lookup.get(key, set())

    return candidates

def calculate_blocking_metrics(gt_dict: Dict[str, Set[str]], candidate_dict: Dict[str, Set[str]]):
    """Calculates blocking recall and average candidate count per S1 entity."""
    total_true_matches = 0
    retained_true_matches = 0
    total_candidates = 0

    for s1_id, true_matches in gt_dict.items():
        cands = candidate_dict.get(s1_id, set())
        total_candidates += len(cands)
        total_true_matches += len(true_matches)
        retained_true_matches += len(true_matches & cands)

    recall = (retained_true_matches / total_true_matches) if total_true_matches > 0 else 0.0
    avg_candidates = total_candidates / len(gt_dict) if gt_dict else 0.0

    print(f"Blocking Recall: {recall * 100:.2f}%")
    print(f"Average Candidates per S1: {avg_candidates:.2f}")
    return recall, avg_candidates

def generate_numeric_token_candidates(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Groups S2 and S3 records matching S1 entities on shared numeric tokens (len>=4) + country from normalized address."""
    s2_s3 = pd.concat([s2_df, s3_df], ignore_index=True)
    
    def extract_tokens(row):
        addr = str(row.get("normalized_address", row.get("clean_address", row.get("address", ""))))
        return set(re.findall(r'\b\d{4,}\b', addr))
        
    lookup = {}
    for _, row in s2_s3.iterrows():
        country = row.get("country", "")
        tokens = extract_tokens(row)
        for token in tokens:
            key = (token, country)
            lookup.setdefault(key, set()).add(row["entity_id"])

    candidates = {}
    for _, row in s1_df.iterrows():
        s1_id = row["entity_id"]
        country = row.get("country", "")
        tokens = extract_tokens(row)
        
        cands = set()
        for token in tokens:
            key = (token, country)
            if key in lookup:
                cands.update(lookup[key])
        candidates[s1_id] = cands

    return candidates

def generate_rare_token_candidates(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Groups S2 and S3 records matching S1 entities on shared rare tokens in the name."""
    s2_s3 = pd.concat([s2_df, s3_df], ignore_index=True)
    
    def get_tokens(name):
        name = str(name) if pd.notna(name) else ""
        return set(name.lower().split())
        
    token_freq = {}
    s2_s3_tokens = []
    
    for _, row in s2_s3.iterrows():
        name = row.get("normalized_name", row.get("clean_name", row.get("name", "")))
        tokens = get_tokens(name)
        s2_s3_tokens.append((row["entity_id"], tokens))
        for token in tokens:
            token_freq[token] = token_freq.get(token, 0) + 1
            
    candidates = {}
    if not token_freq:
        for _, row in s1_df.iterrows():
            candidates[row["entity_id"]] = set()
        return candidates

    threshold = pd.Series(list(token_freq.values())).quantile(0.25)
    rare_tokens = {token for token, freq in token_freq.items() if freq <= threshold}
    
    lookup = {}
    for entity_id, tokens in s2_s3_tokens:
        for token in tokens:
            if token in rare_tokens:
                lookup.setdefault(token, set()).add(entity_id)
                
    for _, row in s1_df.iterrows():
        s1_id = row["entity_id"]
        name = row.get("normalized_name", row.get("clean_name", row.get("name", "")))
        tokens = get_tokens(name)
        
        cands = set()
        for token in tokens:
            if token in rare_tokens and token in lookup:
                cands.update(lookup[token])
                
        candidates[s1_id] = cands

    return candidates
