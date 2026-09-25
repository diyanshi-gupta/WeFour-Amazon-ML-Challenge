import pandas as pd
from typing import Dict, Set

def parse_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """Reads train_ground_truth.tsv into a dictionary mapping S1 ID to set of True Matches."""
    df = pd.read_csv(gt_path, sep="\t")
    gt_dict = {}
    for _, row in df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        if matched_str.strip() == "":
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set([m.strip() for m in matched_str.split(",") if m.strip()])
    return gt_dict

def calculate_macro_f05(gt_dict: Dict[str, Set[str]], pred_dict: Dict[str, Set[str]]) -> float:
    """Computes Macro-Averaged F0.5 score including singletons."""
    scores = []
    for s1_id, true_matches in gt_dict.items():
        pred_matches = pred_dict.get(s1_id, set())
        
        if not true_matches and not pred_matches:
            scores.append(1.0)
            continue
        if not pred_matches:
            scores.append(0.0)
            continue
            
        tp = len(pred_matches & true_matches)
        fp = len(pred_matches - true_matches)
        fn = len(true_matches - pred_matches)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        denom = (0.25 * precision + recall)
        if denom == 0:
            scores.append(0.0)
        else:
            f05 = (1.25 * precision * recall) / denom
            scores.append(f05)
            
    return sum(scores) / len(scores) if scores else 0.0