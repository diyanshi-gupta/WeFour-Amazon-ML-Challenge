# src/tuning.py
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from src.evaluation import calculate_macro_f05

DEFAULT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]


def generate_predictions_at_threshold(
    pred_df: pd.DataFrame,
    threshold: float,
    all_s1_ids: List[str],
) -> Dict[str, Set[str]]:
    """Generates a mapping of S1 ID -> Set of Matched Entity IDs at a given probability threshold.

    Guarantees every S1 entity is present (singletons map to empty set()).
    """
    pred_dict: Dict[str, Set[str]] = {s1_id: set() for s1_id in all_s1_ids}

    # Filter candidate pairs that meet or exceed the cutoff
    filtered = pred_df[pred_df["prob_match"] >= threshold]

    for _, row in filtered.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        cand_id = str(row["candidate_entity_id"]).strip()
        if s1_id in pred_dict:
            pred_dict[s1_id].add(cand_id)

    return pred_dict


def evaluate_thresholds(
    pred_df: pd.DataFrame,
    gt_dict: Dict[str, Set[str]],
    all_s1_ids: List[str],
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, pd.DataFrame]:
    """Sweeps probability thresholds to evaluate Macro F0.5, total predicted matches,

    and false positive behavior to identify the optimal cutoff.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    records = []
    best_threshold = 0.5
    best_f05 = -1.0

    for t in thresholds:
        pred_dict = generate_predictions_at_threshold(pred_df, t, all_s1_ids)
        f05 = calculate_macro_f05(gt_dict, pred_dict)

        total_predicted = sum(len(cands) for cands in pred_dict.values())
        singletons = sum(1 for cands in pred_dict.values() if len(cands) == 0)

        # Calculate Global TP, FP, FN
        tp, fp, fn = 0, 0, 0
        for s1_id, true_set in gt_dict.items():
            pred_set = pred_dict.get(s1_id, set())
            tp += len(pred_set & true_set)
            fp += len(pred_set - true_set)
            fn += len(true_set - pred_set)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        records.append({
            "Threshold": t,
            "Macro F0.5": round(f05, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "Matches Pred": total_predicted,
            "False Positives": fp,
            "Singletons": singletons,
        })

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = t

    results_df = pd.DataFrame(records)

    print("\n" + "=" * 80)
    print("Probability Threshold Tuning Report (Macro F0.5 Optimization)")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("-" * 80)
    print(f"Optimal Threshold: {best_threshold:.2f} (Macro F0.5: {best_f05:.4f})")
    print("=" * 80)

    return best_threshold, results_df


def export_matching_results_tsv(
    pred_dict: Dict[str, Set[str]],
    all_s1_ids: List[str],
    output_path: str = "output/matching_results.tsv",
) -> Path:
    """Exports final predictions matching official challenge requirements:

    Columns: source1_entity_id \t matched_entity_ids
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id in all_s1_ids:
        matches = sorted(list(pred_dict.get(s1_id, set())))
        matched_str = ",".join(matches) if matches else ""
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": matched_str,
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)
    print(f"Saved matching results to: {path} ({len(df)} rows)")
    return path


def export_candidate_pairs_tsv(
    candidate_dict: Dict[str, Set[str]],
    all_s1_ids: List[str],
    output_path: str = "output/candidate_pairs.tsv",
) -> Path:
    """Exports blocking candidate pairs:

    Columns: source1_entity_id \t candidate_entity_ids
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id in all_s1_ids:
        cands = sorted(list(candidate_dict.get(s1_id, set())))
        cands_str = ",".join(cands) if cands else ""
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": cands_str,
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)
    print(f"Saved candidate pairs to: {path} ({len(df)} rows)")
    return path
