"""
Blocking module for candidate pair generation.
"""

from typing import Dict, Set
import pandas as pd


def generate_exact_name_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame
) -> Dict[str, Set[str]]:
    """
    Generates exact match candidate pairs between Source1 and Source2/Source3.
    Pairs are created when normalized_name is non-empty AND exact match AND country matches.

    Args:
        s1_df: Source1 DataFrame (normalized)
        s2_df: Source2 DataFrame (normalized)
        s3_df: Source3 DataFrame (normalized)

    Returns:
        Dict[str, Set[str]]: Mapping from source1_entity_id to set of candidate S2/S3 IDs.
    """
    s1_id_col = "source1_entity_id" if "source1_entity_id" in s1_df.columns else "entity_id"
    s2_id_col = "source2_entity_id" if "source2_entity_id" in s2_df.columns else "entity_id"
    s3_id_col = "source3_entity_id" if "source3_entity_id" in s3_df.columns else "entity_id"

    # Combine S2 and S3 into one target DataFrame
    s2_subset = s2_df[[s2_id_col, "normalized_name", "country"]].rename(columns={s2_id_col: "target_id"})
    s3_subset = s3_df[[s3_id_col, "normalized_name", "country"]].rename(columns={s3_id_col: "target_id"})
    target_df = pd.concat([s2_subset, s3_subset], ignore_index=True)

    # Filter out empty names
    target_clean = target_df[target_df["normalized_name"].str.strip() != ""].copy()

    # Create lookup map: (normalized_name, country) -> set of target_ids
    lookup: Dict[tuple, Set[str]] = {}
    for _, row in target_clean.iterrows():
        key = (str(row["normalized_name"]).strip(), str(row["country"]).strip())
        target_id = str(row["target_id"]).strip()
        lookup.setdefault(key, set()).add(target_id)

    # Match Source1 records against lookup
    candidates: Dict[str, Set[str]] = {}
    for _, row in s1_df.iterrows():
        s1_id = str(row[s1_id_col]).strip()
        name = str(row["normalized_name"]).strip()
        country = str(row["country"]).strip()
        key = (name, country)

        if name != "" and key in lookup:
            candidates[s1_id] = set(lookup[key])
        else:
            candidates[s1_id] = set()

    return candidates


__all__ = ["generate_exact_name_candidates"]
