import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.features.name_similarity import compute_name_similarity_features
from src.features.address_similarity import compute_address_similarity_features
from src.features.embedding_similarity import compute_embedding_similarity
from src.features.domain_features import compute_domain_features

def build_feature_matrix(
    candidate_pairs: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_s3_df: pd.DataFrame,
    gt_df: pd.DataFrame = None,
    save_path: str = "feature_matrix.parquet"
) -> pd.DataFrame:
    """
    Computes all feature blocks, merges them, attaches ground truth labels, 
    and saves the consolidated feature matrix to parquet.
    """
    print(f"Building feature matrix for {len(candidate_pairs)} candidate pairs...")
    
    # Flatten candidate_pairs if it is multi-indexed
    cands_flat = candidate_pairs.reset_index() if isinstance(candidate_pairs.index, pd.MultiIndex) else candidate_pairs
    
    # 1. Name Similarity
    print("Computing name similarity...")
    name_sim_df = compute_name_similarity_features(cands_flat, s1_df, s2_s3_df)
    if not isinstance(name_sim_df.index, pd.MultiIndex):
        name_sim_df.set_index(["source1_entity_id", "candidate_entity_id"], inplace=True)
        
    # 2. Address Similarity
    print("Computing address similarity...")
    addr_sim_df = compute_address_similarity_features(cands_flat, s1_df, s2_s3_df)
    if not isinstance(addr_sim_df.index, pd.MultiIndex):
        addr_sim_df.set_index(["source1_entity_id", "candidate_entity_id"], inplace=True)
        
    # 3. Embedding Similarity
    print("Computing embedding similarity...")
    emb_sim_df = compute_embedding_similarity(cands_flat, s1_df, s2_s3_df)
    if not isinstance(emb_sim_df.index, pd.MultiIndex):
        emb_sim_df.set_index(["source1_entity_id", "candidate_entity_id"], inplace=True)
        
    # 4. Domain Features & Risk Flags
    print("Computing domain features...")
    addr_sim_reset = addr_sim_df.reset_index()
    jaccard_series = addr_sim_reset["address_token_jaccard"] if "address_token_jaccard" in addr_sim_reset.columns else None
    unified_corr_series = cands_flat["has_address_corroboration"] if "has_address_corroboration" in cands_flat.columns else None

    domain_df = compute_domain_features(
        cands_flat, 
        s1_df, 
        s2_s3_df, 
        address_jaccard_series=jaccard_series,
        unified_corroboration_series=unified_corr_series
    )
    if not isinstance(domain_df.index, pd.MultiIndex):
        domain_df.set_index(["source1_entity_id", "candidate_entity_id"], inplace=True)

    # 5. Merge all DataFrames
    print("Merging feature DataFrames...")
    merged = pd.concat([name_sim_df, addr_sim_df, emb_sim_df, domain_df], axis=1)
    feature_matrix = merged.reset_index()
    
    # Keep the source column if it existed
    if "source" in cands_flat.columns and "source" not in feature_matrix.columns:
        feature_matrix = feature_matrix.merge(
            cands_flat[["source1_entity_id", "candidate_entity_id", "source"]], 
            on=["source1_entity_id", "candidate_entity_id"], 
            how="left"
        )
    
    # 6. Attach Ground Truth Labels
    if gt_df is not None and not gt_df.empty:
        print("Attaching ground truth labels...")
        gt_map = {}
        for _, row in gt_df.iterrows():
            s1_id = str(row["source1_entity_id"])
            if pd.notna(row["matched_entity_ids"]):
                matches = [m.strip() for m in str(row["matched_entity_ids"]).split(",")]
                gt_map[s1_id] = set(matches)
            else:
                gt_map[s1_id] = set()
                
        labels = []
        for _, row in feature_matrix.iterrows():
            s1 = str(row["source1_entity_id"])
            cand = str(row["candidate_entity_id"])
            if s1 in gt_map and cand in gt_map[s1]:
                labels.append(1)
            else:
                labels.append(0)
        feature_matrix["label"] = labels
        if len(labels) > 0:
            print(f"Added labels. Match ratio: {sum(labels)/len(labels):.4f}")
        
    if save_path:
        print(f"Saving feature matrix to {save_path}...")
        feature_matrix.to_parquet(save_path, index=False)
        print("Done!")
        
    return feature_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", type=str, default="dataset/train/train_source1.tsv")
    parser.add_argument("--s2", type=str, default="dataset/train/train_source2.tsv")
    parser.add_argument("--s3", type=str, default="dataset/train/train_source3.tsv")
    parser.add_argument("--gt", type=str, default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--candidates", type=str, default="output/unified_candidates.parquet")
    parser.add_argument("--out", type=str, default="feature_matrix.parquet")
    args = parser.parse_args()
    
    if os.path.exists(args.s1) and os.path.exists(args.candidates):
        s1 = pd.read_csv(args.s1, sep='\t')
        s2 = pd.read_csv(args.s2, sep='\t') if os.path.exists(args.s2) else pd.DataFrame()
        s3 = pd.read_csv(args.s3, sep='\t') if os.path.exists(args.s3) else pd.DataFrame()
        gt = pd.read_csv(args.gt, sep='\t') if os.path.exists(args.gt) else None
        
        s2_s3 = pd.concat([s2, s3], ignore_index=True)
        if args.candidates.endswith('.parquet'):
            cands = pd.read_parquet(args.candidates)
        else:
            cands = pd.read_csv(args.candidates, sep='\t')
        
        build_feature_matrix(cands, s1, s2_s3, gt, args.out)
    else:
        print("Required datasets or candidates file not found.")
