import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.blocking.blocking import run_blocking_v1_pipeline if os.path.exists("src/blocking/blocking.py") else None
try:
    from src.blocking import run_blocking_v1_pipeline
except ImportError:
    pass

from code.business_entity_resolution.src.blocking.tfidf_channel import tfidf_faiss_blocking
from code.business_entity_resolution.src.blocking.embedding_channel import embedding_faiss_blocking
from src.blocking.unified import unify_and_corroborate_candidates
from src.features.domain_features import compute_domain_features

def df_to_dict(df):
    d = {}
    if df.empty: return d
    for s1, cand in zip(df["source1_entity_id"], df["candidate_entity_id"]):
        d.setdefault(str(s1), set()).add(str(cand))
    return d

def compute_recall_and_reduction(cand_dict, gt_df, s1_len, s2_s3_len):
    total_true_matches = 0
    retained_true_matches = 0
    
    gt_map = {}
    for _, row in gt_df.iterrows():
        s1 = str(row["source1_entity_id"])
        matches = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if pd.notna(row["matched_entity_ids"])]
        if matches:
            gt_map[s1] = set(matches)
            total_true_matches += len(matches)
            
    for s1, true_matches in gt_map.items():
        cands = cand_dict.get(s1, set())
        retained_true_matches += len(true_matches.intersection(cands))
        
    recall = retained_true_matches / total_true_matches if total_true_matches > 0 else 0.0
    
    total_pairs = sum(len(c) for c in cand_dict.values())
    max_pairs = s1_len * s2_s3_len
    reduction_ratio = 1.0 - (total_pairs / max_pairs) if max_pairs > 0 else 0.0
    
    return recall, reduction_ratio, total_pairs

def get_false_candidate_rate(unified_df, s1_df, s2_s3_df):
    """
    Compute false-candidate-rate proxy:
    fraction of candidates where S1 and candidate share an exact/near-exact name 
    but have has_address_corroboration == False.
    """
    if "has_address_corroboration" not in unified_df.columns:
        return 0.0
        
    domain_features = compute_domain_features(
        unified_df, 
        s1_df, 
        s2_s3_df, 
        address_jaccard_series=None, 
        unified_corroboration_series=unified_df["has_address_corroboration"]
    ).reset_index()
    
    merged = pd.merge(
        unified_df, 
        domain_features[["source1_entity_id", "candidate_entity_id", "exact_name_match"]],
        on=["source1_entity_id", "candidate_entity_id"],
        how="left"
    )
    
    total_candidates = len(merged)
    if total_candidates == 0: return 0.0
    
    name_only_exact = merged[(merged["exact_name_match"] == True) & (merged["has_address_corroboration"] == False)]
    return len(name_only_exact) / total_candidates

def main():
    print("Loading datasets...")
    # Load subset of data for tuning speed
    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", nrows=2000)
    s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", nrows=10000)
    s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", nrows=10000)
    gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")
    
    s1_len = len(s1)
    s2_s3_len = len(s2) + len(s3)
    gt = gt[gt["source1_entity_id"].isin(s1["entity_id"])]
    
    for df in [s1, s2, s3]:
        if "business_name" in df.columns:
            df["clean_name"] = df["business_name"]
            df["normalized_name"] = df["business_name"]
        if "business_address" in df.columns:
            df["clean_address"] = df["business_address"]
            df["normalized_address"] = df["business_address"]

    print("Running Rule-Based Blocker v1...")
    # If run_blocking_v1_pipeline is not available, mock an empty dict
    try:
        from src.blocking import run_blocking_v1_pipeline
        rule_dict = run_blocking_v1_pipeline(s1, s2, s3, gt_dict=None)
    except:
        try:
            from src.blocking.blocking import run_blocking_v1_pipeline
            rule_dict = run_blocking_v1_pipeline(s1, s2, s3, gt_dict=None)
        except:
            print("Could not import run_blocking_v1_pipeline. Continuing without it.")
            rule_dict = {}

    rule_rec, rule_red, rule_count = compute_recall_and_reduction(rule_dict, gt, s1_len, s2_s3_len)
    
    best_k = 20
    for k in [10, 20, 30, 40]:
        print(f"\n--- Testing K={k} ---")
        
        print("Running TF-IDF Channel...")
        tfidf_df = tfidf_faiss_blocking(s1, s2, s3, k=k)
        tfidf_dict = df_to_dict(tfidf_df)
        tfidf_rec, tfidf_red, tfidf_count = compute_recall_and_reduction(tfidf_dict, gt, s1_len, s2_s3_len)
        
        print("Running Embedding Channel...")
        emb_df = embedding_faiss_blocking(s1, s2, s3, k=k, batch_size=256)
        emb_dict = df_to_dict(emb_df)
        emb_rec, emb_red, emb_count = compute_recall_and_reduction(emb_dict, gt, s1_len, s2_s3_len)
        
        print("Computing Union...")
        unified_df = unify_and_corroborate_candidates([rule_dict, tfidf_dict, emb_dict], s1, s2, s3)
        union_dict = df_to_dict(unified_df)
        union_rec, union_red, union_count = compute_recall_and_reduction(union_dict, gt, s1_len, s2_s3_len)
        
        fcr_proxy = get_false_candidate_rate(unified_df, s1, pd.concat([s2, s3]))
        
        print(f"Union Recall: {union_rec*100:.2f}% | FCR Proxy: {fcr_proxy*100:.2f}%")
        
        best_k = k
        if union_rec >= 0.97:
            print("Found K hitting 97%+ recall!")
            break

    print(f"\nGenerating Report with K={best_k}...")
    
    report = f"# Blocking Recall & False-Candidate Proxy Report\n\n"
    report += f"**Tuned Top-K**: {best_k}\n\n"
    report += "## Individual Channels\n\n"
    
    report += "### 1. Rule-Based Pipeline (v1)\n"
    report += f"- **Recall**: {rule_rec*100:.2f}%\n"
    report += f"- **Reduction Ratio**: {rule_red*100:.6f}%\n"
    report += f"- **Total Candidates**: {rule_count}\n\n"
    
    report += f"### 2. TF-IDF Channel (Top-{best_k})\n"
    report += f"- **Recall**: {tfidf_rec*100:.2f}%\n"
    report += f"- **Reduction Ratio**: {tfidf_red*100:.6f}%\n"
    report += f"- **Total Candidates**: {tfidf_count}\n\n"
    
    report += f"### 3. Embedding Channel (Top-{best_k})\n"
    report += f"- **Recall**: {emb_rec*100:.2f}%\n"
    report += f"- **Reduction Ratio**: {emb_red*100:.6f}%\n"
    report += f"- **Total Candidates**: {emb_count}\n\n"
    
    report += "## Union Pipeline\n\n"
    report += f"- **Recall**: {union_rec*100:.2f}%\n"
    report += f"- **Reduction Ratio**: {union_red*100:.6f}%\n"
    report += f"- **Total Candidates**: {union_count}\n\n"
    
    report += "## Precision Risk (False-Candidate-Rate Proxy)\n\n"
    report += (
        f"**FCR Proxy**: {fcr_proxy*100:.2f}%\n\n"
        "*(Fraction of candidates where S1 and candidate share an exact/near-exact name but have `has_address_corroboration == False`)*\n\n"
        "This metric tracks whether adding more blocking channels (like TF-IDF and embeddings) "
        "is exacerbating the singleton false-merge problem diagnosed on Day 1. "
        "If this number increases drastically, it confirms that the address-corroboration gate "
        "is necessary to prevent a precision drop in the downstream classifier.\n"
    )
    
    os.makedirs("findings", exist_ok=True)
    with open("findings/blocking_recall.md", "w") as f:
        f.write(report)
        
    print("Report written to findings/blocking_recall.md")

if __name__ == "__main__":
    main()
