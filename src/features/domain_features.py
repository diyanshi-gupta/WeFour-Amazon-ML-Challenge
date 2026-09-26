import re
import pandas as pd

def _safe_str(text) -> str:
    if pd.isna(text) or not str(text).strip():
        return ""
    return str(text).strip().lower()

def _get_numeric_tokens(text: str) -> set:
    return set(re.findall(r'\b\d{4,}\b', _safe_str(text)))

def compute_domain_features(
    candidate_pairs: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_s3_df: pd.DataFrame,
    address_jaccard_series: pd.Series = None,
    unified_corroboration_series: pd.Series = None
) -> pd.DataFrame:
    """
    Computes domain-specific flags for candidate pairs.
    Returns a DataFrame keyed by (source1_entity_id, candidate_entity_id).
    """
    # Build fast lookups
    def build_lookup(df, col):
        if col in df.columns:
            return df.set_index("entity_id")[col].to_dict()
        return {}

    s1_country = build_lookup(s1_df, "country")
    cand_country = build_lookup(s2_s3_df, "country")
    
    # Resolve name columns
    def get_name_col(df):
        for c in ["normalized_name", "clean_name", "business_name"]:
            if c in df.columns: return c
        return None
        
    s1_name_col = get_name_col(s1_df)
    cand_name_col = get_name_col(s2_s3_df)
    s1_name = build_lookup(s1_df, s1_name_col) if s1_name_col else {}
    cand_name = build_lookup(s2_s3_df, cand_name_col) if cand_name_col else {}
    
    # Resolve address columns
    def get_addr_col(df):
        for c in ["normalized_address", "clean_address", "business_address"]:
            if c in df.columns: return c
        return None
        
    s1_addr_col = get_addr_col(s1_df)
    cand_addr_col = get_addr_col(s2_s3_df)
    s1_addr = build_lookup(s1_df, s1_addr_col) if s1_addr_col else {}
    cand_addr = build_lookup(s2_s3_df, cand_addr_col) if cand_addr_col else {}

    rows = []
    
    # Handle optional aligned series
    if address_jaccard_series is not None:
        jaccard_vals = address_jaccard_series.values
    else:
        jaccard_vals = [0.0] * len(candidate_pairs)
        
    if unified_corroboration_series is not None:
        unified_corr_vals = unified_corroboration_series.values
    else:
        unified_corr_vals = [False] * len(candidate_pairs)

    cands_dict = candidate_pairs.to_dict(orient="records")
    
    for i, row in enumerate(cands_dict):
        s1_id = str(row["source1_entity_id"])
        cand_id = str(row["candidate_entity_id"])
        
        # 1. Exact country match
        c1 = _safe_str(s1_country.get(s1_id, ""))
        c2 = _safe_str(cand_country.get(cand_id, ""))
        exact_country = (c1 == c2) if (c1 and c2) else False
        
        # 2. Exact name match
        n1 = _safe_str(s1_name.get(s1_id, ""))
        n2 = _safe_str(cand_name.get(cand_id, ""))
        exact_name = (n1 == n2) if (n1 and n2) else False
        
        # 3. PIN / numeric overlap
        a1 = s1_addr.get(s1_id, "")
        a2 = cand_addr.get(cand_id, "")
        nums1 = _get_numeric_tokens(a1)
        nums2 = _get_numeric_tokens(a2)
        overlap = len(nums1 & nums2)
        
        # 4. Corroboration & Risk Flags
        jaccard = float(jaccard_vals[i])
        unified_corr = bool(unified_corr_vals[i])
        
        has_corroboration = (overlap > 0) or (jaccard > 0.2) or unified_corr
        name_only_high_risk = exact_name and not has_corroboration
        
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "exact_country_match": exact_country,
            "pin_numeric_overlap": overlap,
            "exact_name_match": exact_name,
            "has_corroboration": has_corroboration,
            "name_only_high_risk": name_only_high_risk
        })
        
    df = pd.DataFrame(rows)
    if not df.empty:
        df.set_index(["source1_entity_id", "candidate_entity_id"], inplace=True)
    return df
