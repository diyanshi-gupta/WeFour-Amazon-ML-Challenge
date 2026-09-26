"""blocking/tfidf_channel.py

Re-exports and entrypoint for the TF-IDF character 3-gram FAISS blocking channel.
"""
import sys
from pathlib import Path

# Add project root to sys.path if not present
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from code.business_entity_resolution.src.blocking.tfidf_channel import (
    TfidfFaissBlocker,
    tfidf_faiss_blocking,
    prepare_text_series,
    to_candidate_dict,
    evaluate_blocking_recall,
    infer_entity_source,
)

__all__ = [
    "TfidfFaissBlocker",
    "tfidf_faiss_blocking",
    "prepare_text_series",
    "to_candidate_dict",
    "evaluate_blocking_recall",
    "infer_entity_source",
]

if __name__ == "__main__":
    from code.business_entity_resolution.src.blocking.tfidf_channel import _find_dataset_train_dir, _create_benchmark_data
    import pandas as pd

    train_dir = _find_dataset_train_dir()
    if train_dir:
        print(f"Found dataset directory at: {train_dir}")
        print("Loading sample of 5,000 records per source for benchmark...")
        s1 = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", nrows=5000)
        s2 = pd.read_csv(train_dir / "train_source2.tsv", sep="\t", nrows=5000)
        s3 = pd.read_csv(train_dir / "train_source3.tsv", sep="\t", nrows=5000)

        from src.normalize import normalize_name, normalize_address  # type: ignore

        for df in [s1, s2, s3]:
            df["normalized_name"] = df["business_name"].apply(normalize_name)
            df["normalized_address"] = df["business_address"].apply(normalize_address)

        from code.business_entity_resolution.src.blocking.tfidf_channel import load_ground_truth_sample

        s1_ids_set = set(s1["entity_id"].astype(str))
        gt_dict = load_ground_truth_sample(train_dir / "train_ground_truth.tsv", s1_ids_set)
    else:
        print("No local dataset files detected; executing on benchmark dataset...")
        s1, s2_s3, gt_dict = _create_benchmark_data()
        s2 = s2_s3[s2_s3["source"] == "S2"]
        s3 = s2_s3[s2_s3["source"] == "S3"]

    print("\nRunning Character 3-Gram TF-IDF FAISS Blocking (K=20)...")
    candidates = tfidf_faiss_blocking(
        s1_df=s1,
        s2_df=s2,
        s3_df=s3,
        k=20,
        min_similarity=0.0,
        max_features=25000,
        include_similarity=True,
    )

    print(f"\nGenerated {len(candidates)} candidate pairs.")
    print("\nSample Candidate Pairs with Cosine Similarity:")
    print(candidates.head(10).to_string(index=False))

    evaluate_blocking_recall(
        candidate_pairs_df=candidates,
        gt_dict=gt_dict,
        total_s1_count=len(s1),
    )
