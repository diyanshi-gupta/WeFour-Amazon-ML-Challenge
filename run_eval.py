#!/usr/bin/env python3
"""
Evaluation CLI wrapper and metrics breakdown tool for Business Entity Resolution.
Computes overall, singleton-only, multi-match, and per-country macro F_0.5 scores.
"""

import argparse
import sys
from typing import Dict, Set, Union, Optional, Any
import pandas as pd

from src.eval.f_beta import f_beta_score


def parse_matches_to_dict(
    source: Union[str, pd.DataFrame, Dict[str, Set[str]]]
) -> Dict[str, Set[str]]:
    """
    Parses a TSV filepath, pandas DataFrame, or dictionary into Dict[str, Set[str]].
    """
    if isinstance(source, dict):
        return source

    if isinstance(source, str):
        df = pd.read_csv(source, sep="\t", dtype=str)
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        raise ValueError(f"Unsupported source type for parsing: {type(source)}")

    # Determine column names
    s1_col = "source1_entity_id" if "source1_entity_id" in df.columns else "entity_id"
    if s1_col not in df.columns:
        raise KeyError(f"Could not find entity ID column ('source1_entity_id' or 'entity_id') in {df.columns.tolist()}")

    match_col = None
    for candidate in ["matched_entity_ids", "candidate_entity_ids", "matched_entity_id"]:
        if candidate in df.columns:
            match_col = candidate
            break

    if match_col is None:
        raise KeyError(f"Could not find matched entity IDs column in {df.columns.tolist()}")

    result_dict: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        s1_id = str(row[s1_col]).strip() if pd.notna(row[s1_col]) else ""
        if not s1_id:
            continue

        raw_matches = row[match_col]
        if pd.isna(raw_matches) or raw_matches is None:
            result_dict[s1_id] = set()
        else:
            match_str = str(raw_matches).strip()
            if not match_str:
                result_dict[s1_id] = set()
            else:
                matches = set(m.strip() for m in match_str.split(",") if m.strip())
                result_dict[s1_id] = matches

    return result_dict


def parse_s1_country_map(
    s1_df_or_path: Union[str, pd.DataFrame, Dict[str, str]]
) -> Dict[str, str]:
    """
    Parses Source1 metadata file or DataFrame into a dictionary mapping entity_id -> country.
    """
    if isinstance(s1_df_or_path, dict):
        return s1_df_or_path

    if isinstance(s1_df_or_path, str):
        df = pd.read_csv(s1_df_or_path, sep="\t", dtype=str, usecols=lambda c: c in ["source1_entity_id", "entity_id", "country"])
    elif isinstance(s1_df_or_path, pd.DataFrame):
        df = s1_df_or_path
    else:
        return {}

    id_col = "source1_entity_id" if "source1_entity_id" in df.columns else "entity_id"
    if id_col not in df.columns or "country" not in df.columns:
        return {}

    country_map = {}
    for _, row in df.iterrows():
        s1_id = str(row[id_col]).strip() if pd.notna(row[id_col]) else ""
        country = str(row["country"]).strip() if pd.notna(row["country"]) else "UNKNOWN"
        if s1_id:
            country_map[s1_id] = country

    return country_map


def evaluate_breakdown(
    val_gt: Union[Dict[str, Set[str]], str, pd.DataFrame],
    val_pred: Union[Dict[str, Set[str]], str, pd.DataFrame],
    s1_df: Optional[Union[str, pd.DataFrame, Dict[str, str]]] = None,
    beta: float = 0.5
) -> Dict[str, Any]:
    """
    Computes macro F_beta score overall, for singletons, for multi-matches, and per country.

    Returns dict with keys:
        'overall_f05', 'total_entities',
        'singleton_f05', 'singleton_count',
        'multimatch_f05', 'multimatch_count',
        'by_country': { country: {'f05': float, 'count': int, ...} }
    """
    gt_dict = parse_matches_to_dict(val_gt)
    pred_dict = parse_matches_to_dict(val_pred)

    # 1. Overall Score
    overall_f05 = f_beta_score(gt_dict, pred_dict, beta=beta)

    # 2. Singleton vs Multi-Match subsets
    singleton_gt = {k: v for k, v in gt_dict.items() if len(v) == 0}
    singleton_pred = {k: pred_dict.get(k, set()) for k in singleton_gt}
    singleton_f05 = f_beta_score(singleton_gt, singleton_pred, beta=beta) if singleton_gt else 0.0

    multimatch_gt = {k: v for k, v in gt_dict.items() if len(v) > 0}
    multimatch_pred = {k: pred_dict.get(k, set()) for k in multimatch_gt}
    multimatch_f05 = f_beta_score(multimatch_gt, multimatch_pred, beta=beta) if multimatch_gt else 0.0

    # 3. Country Breakdown
    country_breakdown = {}
    if s1_df is not None:
        country_map = parse_s1_country_map(s1_df)
        if country_map:
            # Group entity IDs by country
            country_to_ids: Dict[str, Set[str]] = {}
            for s1_id in gt_dict:
                country = country_map.get(s1_id, "UNKNOWN")
                country_to_ids.setdefault(country, set()).add(s1_id)

            for country, ids in country_to_ids.items():
                c_gt = {k: gt_dict[k] for k in ids}
                c_pred = {k: pred_dict.get(k, set()) for k in ids}
                c_f05 = f_beta_score(c_gt, c_pred, beta=beta)

                c_sing_gt = {k: v for k, v in c_gt.items() if len(v) == 0}
                c_sing_pred = {k: c_pred.get(k, set()) for k in c_sing_gt}
                c_sing_f05 = f_beta_score(c_sing_gt, c_sing_pred, beta=beta) if c_sing_gt else 0.0

                c_multi_gt = {k: v for k, v in c_gt.items() if len(v) > 0}
                c_multi_pred = {k: c_pred.get(k, set()) for k in c_multi_gt}
                c_multi_f05 = f_beta_score(c_multi_gt, c_multi_pred, beta=beta) if c_multi_gt else 0.0

                country_breakdown[country] = {
                    "f05": c_f05,
                    "count": len(ids),
                    "singleton_f05": c_sing_f05,
                    "singleton_count": len(c_sing_gt),
                    "multimatch_f05": c_multi_f05,
                    "multimatch_count": len(c_multi_gt),
                }

    return {
        "overall_f05": overall_f05,
        "total_entities": len(gt_dict),
        "singleton_f05": singleton_f05,
        "singleton_count": len(singleton_gt),
        "multimatch_f05": multimatch_f05,
        "multimatch_count": len(multimatch_gt),
        "by_country": country_breakdown,
    }


def print_summary_report(results: Dict[str, Any], pred_path: str = "", gt_path: str = ""):
    """Formats and prints a clean ASCII evaluation report."""
    print("=" * 72)
    print("                WEFOUR EVALUATION REPORT (F_0.5 METRIC)")
    print("=" * 72)
    if pred_path:
        print(f" Predictions TSV : {pred_path}")
    if gt_path:
        print(f" Ground Truth TSV: {gt_path}")
    print("-" * 72)
    print(f"{'Metric Segment':<35} {'Entities':<15} {'F_0.5 Score':<15}")
    print("-" * 72)
    print(f"{'Overall Macro F_0.5':<35} {results['total_entities']:<15,d} {results['overall_f05']:<15.4f}")
    print(f"{'Singleton-only (Ground Truth=0)':<35} {results['singleton_count']:<15,d} {results['singleton_f05']:<15.4f}")
    print(f"{'Multi-match (Ground Truth>=1)':<35} {results['multimatch_count']:<15,d} {results['multimatch_f05']:<15.4f}")

    if results.get("by_country"):
        print("-" * 72)
        print("PER-COUNTRY BREAKDOWN:")
        print(f"{'Country':<20} {'Entities':<15} {'Overall F_0.5':<15} {'Singleton F_0.5':<15}")
        print("-" * 72)
        for country, data in sorted(results["by_country"].items()):
            print(f"{country:<20} {data['count']:<15,d} {data['f05']:<15.4f} {data['singleton_f05']:<15.4f}")

    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation and compute granular macro F_0.5 breakdown."
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="Path to prediction matching results TSV (columns: source1_entity_id, matched_entity_ids)"
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="Path to ground truth TSV (columns: source1_entity_id, matched_entity_ids)"
    )
    parser.add_argument(
        "--s1-df",
        default=None,
        help="Optional path to Source1 metadata TSV for country breakdown"
    )

    args = parser.parse_args()

    results = evaluate_breakdown(val_gt=args.gt, val_pred=args.pred, s1_df=args.s1_df)
    print_summary_report(results, pred_path=args.pred, gt_path=args.gt)


if __name__ == "__main__":
    main()
