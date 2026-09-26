# src/blocking.py
import re
from typing import Callable, Dict, List, Optional, Set, Tuple
import pandas as pd

# Generic or high-frequency tokens to exclude from name-token blocking
COMMON_NAME_TOKENS: Set[str] = {
    "corporation",
    "incorporated",
    "limited",
    "private",
    "company",
    "llc",
    "corp",
    "inc",
    "ltd",
    "pvt",
    "the",
    "and",
    "for",
    "with",
    "solutions",
    "services",
    "enterprises",
    "group",
    "holdings",
    "international",
    "global",
    "industries",
    "agency",
    "associates",
    "store",
    "shop",
    "center",
    "centre",
}


def extract_informative_tokens(text: str, min_len: int = 3) -> Set[str]:
    """Extracts distinctive, informative tokens from normalized text.
    
    Filters out common legal suffixes, stopwords, single characters, and pure numbers.
    """
    if pd.isna(text) or not str(text).strip():
        return set()

    tokens = set()
    for word in re.findall(r"\b[a-z]{" + str(min_len) + r",}\b", str(text).lower()):
        if word not in COMMON_NAME_TOKENS:
            tokens.add(word)
    return tokens


def extract_address_numeric_tokens(address: str) -> Set[str]:
    """Extracts numeric patterns from business addresses, such as PIN/ZIP codes

    and street/building numbers (typically 3 to 6 digits).
    """
    if pd.isna(address) or not str(address).strip():
        return set()

    # Match 3 to 6 digit numbers (US 5-digit zip, Indian 6-digit PIN, building numbers)
    return set(re.findall(r"\b\d{3,6}\b", str(address)))


# -------------------------------------------------------------------------
# Strategy 1: Exact Normalized-Name + Country Blocker
# -------------------------------------------------------------------------
def generate_exact_name_candidates(
    s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame
) -> Dict[str, Set[str]]:
    """Groups S2 and S3 records matching S1 entities on exact clean_name + country."""
    s2_s3 = pd.concat([s2_df, s3_df], ignore_index=True)
    lookup: Dict[Tuple[str, str], Set[str]] = {}

    for _, row in s2_s3.iterrows():
        name = str(row.get("clean_name", "")).strip()
        country = str(row.get("country", "")).strip()
        if name:
            key = (name, country)
            lookup.setdefault(key, set()).add(row["entity_id"])

    candidates: Dict[str, Set[str]] = {}
    for _, row in s1_df.iterrows():
        name = str(row.get("clean_name", "")).strip()
        country = str(row.get("country", "")).strip()
        key = (name, country)
        candidates[row["entity_id"]] = set(lookup.get(key, set()))

    return candidates


# -------------------------------------------------------------------------
# Strategy 2: Informative Name Token Blocker
# -------------------------------------------------------------------------
def generate_name_token_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    max_block_size: int = 200,
    min_token_len: int = 3,
) -> Dict[str, Set[str]]:
    """Blocks candidate pairs sharing at least one rare/informative name token within the same country.

    Filters tokens appearing in more than max_block_size records to avoid combinatorial explosion.
    """
    s2_s3 = pd.concat([s2_df, s3_df], ignore_index=True)
    lookup: Dict[Tuple[str, str], Set[str]] = {}

    for _, row in s2_s3.iterrows():
        country = str(row.get("country", "")).strip()
        tokens = extract_informative_tokens(row.get("clean_name", ""), min_len=min_token_len)
        for token in tokens:
            key = (token, country)
            lookup.setdefault(key, set()).add(row["entity_id"])

    # Prune high-frequency blocks to keep candidate sizes reasonable and fast
    filtered_lookup = {k: v for k, v in lookup.items() if len(v) <= max_block_size}

    candidates: Dict[str, Set[str]] = {}
    for _, row in s1_df.iterrows():
        s1_id = row["entity_id"]
        country = str(row.get("country", "")).strip()
        tokens = extract_informative_tokens(row.get("clean_name", ""), min_len=min_token_len)

        matched: Set[str] = set()
        for token in tokens:
            key = (token, country)
            if key in filtered_lookup:
                matched.update(filtered_lookup[key])
        candidates[s1_id] = matched

    return candidates


# -------------------------------------------------------------------------
# Strategy 3: Numeric / PIN / Address-Token Blocker
# -------------------------------------------------------------------------
def generate_numeric_address_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    max_block_size: int = 150,
) -> Dict[str, Set[str]]:
    """Blocks candidate pairs sharing a PIN/ZIP code or street number in the same country

    paired with the first 2 letters of the business name for high-precision local matching.
    """
    s2_s3 = pd.concat([s2_df, s3_df], ignore_index=True)
    lookup: Dict[Tuple[str, str, str], Set[str]] = {}

    for _, row in s2_s3.iterrows():
        country = str(row.get("country", "")).strip()
        name_prefix = str(row.get("clean_name", ""))[:2].strip()
        numbers = extract_address_numeric_tokens(row.get("clean_address", row.get("business_address", "")))

        for num in numbers:
            key = (num, country, name_prefix)
            lookup.setdefault(key, set()).add(row["entity_id"])

    filtered_lookup = {k: v for k, v in lookup.items() if len(v) <= max_block_size}

    candidates: Dict[str, Set[str]] = {}
    for _, row in s1_df.iterrows():
        s1_id = row["entity_id"]
        country = str(row.get("country", "")).strip()
        name_prefix = str(row.get("clean_name", ""))[:2].strip()
        numbers = extract_address_numeric_tokens(row.get("clean_address", row.get("business_address", "")))

        matched: Set[str] = set()
        for num in numbers:
            key = (num, country, name_prefix)
            if key in filtered_lookup:
                matched.update(filtered_lookup[key])
        candidates[s1_id] = matched

    return candidates


# -------------------------------------------------------------------------
# Union Strategy: Combine Candidates
# -------------------------------------------------------------------------
def union_candidate_dicts(*candidate_dicts: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Takes the UNION of candidates from multiple blocking strategies.
    
    Rule: If any reasonable blocker says a pair is plausible, keep it.
    """
    all_s1_ids: Set[str] = set()
    for d in candidate_dicts:
        all_s1_ids.update(d.keys())

    union_dict: Dict[str, Set[str]] = {}
    for s1_id in all_s1_ids:
        merged: Set[str] = set()
        for d in candidate_dicts:
            if s1_id in d:
                merged.update(d[s1_id])
        union_dict[s1_id] = merged

    return union_dict


# -------------------------------------------------------------------------
# Evaluation & Metrics
# -------------------------------------------------------------------------
def calculate_blocking_metrics(
    gt_dict: Dict[str, Set[str]],
    candidate_dict: Dict[str, Set[str]],
    strategy_name: str = "Blocking Strategy",
) -> Tuple[float, float]:
    """Calculates blocking recall and average candidate count per S1 entity.
    
    Prints a clear evaluation summary for the strategy version.
    """
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

    print(f"\n[{strategy_name}]")
    print(f"  * Blocking Recall:           {recall * 100:.2f}% ({retained_true_matches}/{total_true_matches})")
    print(f"  * Average Candidates per S1: {avg_candidates:.2f}")
    print(f"  * Total Candidate Pairs:     {total_candidates:,}")

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

# -------------------------------------------------------------------------
# High-Level Pipeline: Build Blocking v1
# -------------------------------------------------------------------------
def run_blocking_v1_pipeline(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    gt_dict: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Set[str]]:
    """Builds and evaluates Blocking v1 progressively across all versions:
    
    1. Version 1: Exact normalized-name + country
    2. Version 2: Union of (Exact + Informative Name Tokens)
    3. Version 3: Full Union of (Exact + Name Tokens + Numeric/PIN Address Tokens)
    """
    print("=" * 60)
    print("Executing Blocking v1 Pipeline")
    print("=" * 60)

    # 1. Exact Name Blocker
    exact_candidates = generate_exact_name_candidates(s1_df, s2_df, s3_df)
    if gt_dict:
        calculate_blocking_metrics(gt_dict, exact_candidates, strategy_name="v1.0: Exact Name + Country")

    # 2. Informative Name Token Blocker
    token_candidates = generate_name_token_candidates(s1_df, s2_df, s3_df)
    v2_union = union_candidate_dicts(exact_candidates, token_candidates)
    if gt_dict:
        calculate_blocking_metrics(gt_dict, v2_union, strategy_name="v1.1: Exact + Name Tokens (Union)")

    # 3. Numeric / Address-Token Blocker
    num_candidates = generate_numeric_address_candidates(s1_df, s2_df, s3_df)
    v3_full_union = union_candidate_dicts(exact_candidates, token_candidates, num_candidates)
    if gt_dict:
        calculate_blocking_metrics(
            gt_dict, v3_full_union, strategy_name="v1.2: Exact + Name Tokens + Numeric/PIN (Full Union)"
        )

    print("=" * 60)
    return v3_full_union
