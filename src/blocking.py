# src/blocking.py
import pandas as pd
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
