# src/features.py
import re
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates character n-grams from text."""
    text = f"  {text}  "
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
    """Computes character n-gram Dice similarity coefficient."""
    if not text1 or not text2:
        return 0.0
    if text1 == text2:
        return 1.0
    ngrams1 = get_char_ngrams(text1, n)
    ngrams2 = get_char_ngrams(text2, n)
    intersection = len(ngrams1 & ngrams2)
    total = len(ngrams1) + len(ngrams2)
    return (2.0 * intersection) / total if total > 0 else 0.0


def token_jaccard(text1: str, text2: str) -> float:
    """Computes word token Jaccard similarity."""
    tokens1 = set(re.findall(r"\w+", text1.lower()))
    tokens2 = set(re.findall(r"\w+", text2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def extract_numbers(text: str) -> Set[str]:
    """Extracts numeric strings (e.g. street numbers, PIN/ZIP codes)."""
    if pd.isna(text) or not str(text).strip():
        return set()
    return set(re.findall(r"\b\d+\b", str(text)))


def compute_pair_features(
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    cand_name: str,
    cand_addr: str,
    cand_country: str,
) -> Dict[str, float]:
    """Computes simple, highly discriminative pairwise features between S1 and candidate.

    Features computed:
    - Name similarities (exact, token Jaccard, 3-gram Dice, prefix match, length difference)
    - Address similarities (exact, token Jaccard, missing flag)
    - Country match
    - Numeric & PIN overlap
    """
    s1_name_str = str(s1_name or "").strip()
    cand_name_str = str(cand_name or "").strip()

    s1_addr_str = str(s1_addr or "").strip()
    cand_addr_str = str(cand_addr or "").strip()

    s1_country_str = str(s1_country or "").strip().upper()
    cand_country_str = str(cand_country or "").strip().upper()

    # 1. Name Features
    exact_name = 1.0 if (s1_name_str and s1_name_str == cand_name_str) else 0.0
    name_jaccard = token_jaccard(s1_name_str, cand_name_str)
    name_3gram = ngram_similarity(s1_name_str, cand_name_str, n=3)

    prefix_match = 0.0
    if s1_name_str and cand_name_str:
        if s1_name_str.startswith(cand_name_str) or cand_name_str.startswith(s1_name_str):
            prefix_match = 1.0

    max_len = max(len(s1_name_str), len(cand_name_str), 1)
    name_len_diff = abs(len(s1_name_str) - len(cand_name_str)) / max_len

    # 2. Address Features
    exact_addr = 1.0 if (s1_addr_str and s1_addr_str == cand_addr_str) else 0.0
    addr_jaccard = token_jaccard(s1_addr_str, cand_addr_str)
    addr_missing = 1.0 if (not s1_addr_str or not cand_addr_str) else 0.0

    # 3. Country Match Feature
    country_match = 1.0 if (s1_country_str and s1_country_str == cand_country_str) else 0.0

    # 4. Numeric & PIN Overlap Features
    s1_nums = extract_numbers(s1_addr_str)
    cand_nums = extract_numbers(cand_addr_str)
    shared_nums = s1_nums & cand_nums
    numeric_overlap_count = float(len(shared_nums))

    # Shared PIN / ZIP code (typically 5 or 6 digits)
    has_shared_pin = 1.0 if any(len(n) in (5, 6) for n in shared_nums) else 0.0

    return {
        "exact_name_match": exact_name,
        "name_token_jaccard": round(name_jaccard, 4),
        "name_3gram_similarity": round(name_3gram, 4),
        "name_prefix_match": prefix_match,
        "name_len_diff_ratio": round(name_len_diff, 4),
        "exact_address_match": exact_addr,
        "address_token_jaccard": round(addr_jaccard, 4),
        "address_missing": addr_missing,
        "country_match": country_match,
        "numeric_overlap_count": numeric_overlap_count,
        "has_shared_pin": has_shared_pin,
    }


def build_feature_dataframe(
    candidate_dict: Dict[str, Set[str]],
    s1_df: pd.DataFrame,
    s2_s3_df: pd.DataFrame,
    gt_dict: Optional[Dict[str, Set[str]]] = None,
) -> pd.DataFrame:
    """Builds a structured DataFrame of pairwise features for all candidate pairs.

    Optionally joins ground truth labels (1 = True Match, 0 = Non-match).
    """
    s1_indexed = s1_df.set_index("entity_id")
    cand_indexed = s2_s3_df.set_index("entity_id")

    rows = []
    for s1_id, cand_ids in candidate_dict.items():
        if s1_id not in s1_indexed.index:
            continue
        s1_row = s1_indexed.loc[s1_id]
        s1_name = s1_row.get("clean_name", s1_row.get("business_name", ""))
        s1_addr = s1_row.get("clean_address", s1_row.get("business_address", ""))
        s1_country = s1_row.get("country", "")

        true_matches = gt_dict.get(s1_id, set()) if gt_dict else set()

        for cand_id in cand_ids:
            if cand_id not in cand_indexed.index:
                continue
            cand_row = cand_indexed.loc[cand_id]
            cand_name = cand_row.get("clean_name", cand_row.get("business_name", ""))
            cand_addr = cand_row.get("clean_address", cand_row.get("business_address", ""))
            cand_country = cand_row.get("country", "")

            feats = compute_pair_features(
                s1_name=s1_name,
                s1_addr=s1_addr,
                s1_country=s1_country,
                cand_name=cand_name,
                cand_addr=cand_addr,
                cand_country=cand_country,
            )

            feats["source1_entity_id"] = s1_id
            feats["candidate_entity_id"] = cand_id
            feats["s1_name"] = str(s1_name)
            feats["cand_name"] = str(cand_name)

            if gt_dict is not None:
                feats["label"] = 1 if cand_id in true_matches else 0

            rows.append(feats)

    df = pd.DataFrame(rows)
    return df


def inspect_feature_pairs(feature_df: pd.DataFrame, n_samples: int = 3) -> None:
    """Inspects positive and negative candidate pairs manually to verify feature behavior."""
    print("=" * 70)
    print("Manual Inspection of Candidate Pairs and Features")
    print("=" * 70)

    if "label" in feature_df.columns:
        pos_df = feature_df[feature_df["label"] == 1]
        neg_df = feature_df[feature_df["label"] == 0]

        print(f"\n[+] POSITIVE PAIRS (True Matches) - {len(pos_df)} total:")
        sample_pos = pos_df.head(n_samples)
        for _, row in sample_pos.iterrows():
            print(f"  * S1: '{row['s1_name']}' <-> Cand: '{row['cand_name']}'")
            print(
                f"    - name_jaccard: {row['name_token_jaccard']:.2f}, "
                f"name_3gram: {row['name_3gram_similarity']:.2f}, "
                f"addr_jaccard: {row['address_token_jaccard']:.2f}, "
                f"pin_match: {row['has_shared_pin']:.0f}"
            )

        print(f"\n[-] NEGATIVE PAIRS (Non-matches) - {len(neg_df)} total:")
        sample_neg = neg_df.head(n_samples)
        for _, row in sample_neg.iterrows():
            print(f"  * S1: '{row['s1_name']}' <-> Cand: '{row['cand_name']}'")
            print(
                f"    - name_jaccard: {row['name_token_jaccard']:.2f}, "
                f"name_3gram: {row['name_3gram_similarity']:.2f}, "
                f"addr_jaccard: {row['address_token_jaccard']:.2f}, "
                f"pin_match: {row['has_shared_pin']:.0f}"
            )
    else:
        print(f"Total candidate pairs: {len(feature_df)}")
        for _, row in feature_df.head(n_samples).iterrows():
            print(f"  * S1: '{row['s1_name']}' <-> Cand: '{row['cand_name']}'")
            print(
                f"    - name_jaccard: {row['name_token_jaccard']:.2f}, "
                f"name_3gram: {row['name_3gram_similarity']:.2f}"
            )
    print("=" * 70)
