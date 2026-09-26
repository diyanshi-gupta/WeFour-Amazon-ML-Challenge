# code/business_entity_resolution/src/blocking/tfidf_channel.py
"""TF-IDF + FAISS Dense Character n-gram Blocking Channel.

Implements character 3-gram TF-IDF vectorization across normalized business
name and address fields, constructs a FAISS IndexFlatIP (cosine similarity)
over candidate records (Source 2 + Source 3), and retrieves top-K nearest
neighbors for each Source 1 entity.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

try:
    from code.business_entity_resolution.src.schemas import (
        CANDIDATE_PAIR_COLUMNS,
        CandidatePair,
    )
except ImportError:
    try:
        from src.schemas import (  # type: ignore
            CANDIDATE_PAIR_COLUMNS,
            CandidatePair,
        )
    except ImportError:
        CANDIDATE_PAIR_COLUMNS = [
            "source1_entity_id",
            "candidate_entity_id",
            "source",
        ]


def prepare_text_series(df: pd.DataFrame) -> pd.Series:
    """Concatenates normalized_name and normalized_address into a single text series.

    Falls back cleanly to clean_name / business_name and clean_address / business_address
    if normalized columns are not present. Missing values are filled with empty strings.
    """
    # Coalesce name columns with row-level fallback
    names = pd.Series([""] * len(df), index=df.index, dtype=str)
    for col in ["business_name", "clean_name", "normalized_name"]:
        if col in df.columns:
            val = df[col].fillna("").astype(str).str.strip().str.lower()
            names = names.where(names != "", val)

    # Coalesce address columns with row-level fallback
    addrs = pd.Series([""] * len(df), index=df.index, dtype=str)
    for col in ["business_address", "clean_address", "normalized_address"]:
        if col in df.columns:
            val = df[col].fillna("").astype(str).str.strip().str.lower()
            addrs = addrs.where(addrs != "", val)

    combined = (names + " " + addrs).str.strip().str.replace(r"\s+", " ", regex=True)
    return combined


def infer_entity_source(entity_id: str, default_source: str = "unknown") -> str:
    """Infers whether an entity originates from Source 2 or Source 3 from its prefix."""
    eid = str(entity_id).strip().upper()
    if eid.startswith("S2") or eid.startswith("SOURCE2"):
        return "S2"
    if eid.startswith("S3") or eid.startswith("SOURCE3"):
        return "S3"
    return default_source


class TfidfFaissBlocker:
    """Character n-gram TF-IDF Vectorizer with FAISS Cosine Indexing for Candidate Blocking.

    Parameters
    ----------
    ngram_range : Tuple[int, int], default=(3, 3)
        Character n-gram boundaries. Default is character 3-grams (n=3).
    max_features : int or None, default=50000
        Maximum vocabulary size for character n-grams to balance recall and RAM usage.
    min_df : int or float, default=1
        Minimum document frequency for n-gram inclusion.
    sublinear_tf : bool, default=True
        Apply sublinear scaling (1 + log(tf)) to prevent dominant frequencies.
    k : int, default=20
        Default number of nearest neighbors to retrieve per Source 1 entity.
    min_similarity : float, default=0.0
        Minimum cosine similarity threshold to retain a candidate pair.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 3),
        max_features: Optional[int] = 50000,
        min_df: Union[int, float] = 1,
        sublinear_tf: bool = True,
        k: int = 20,
        min_similarity: float = 0.0,
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.k = k
        self.min_similarity = min_similarity

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.candidate_ids: List[str] = []
        self.candidate_sources: List[str] = []
        self.dim: int = 0

    def fit_vectorizer(
        self,
        all_texts: Union[pd.Series, List[str]],
    ) -> "TfidfFaissBlocker":
        """Fits character 3-gram TF-IDF vectorizer across combined corpus text."""
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            norm="l2",
        )
        self.vectorizer.fit(all_texts)
        self.dim = len(self.vectorizer.vocabulary_)
        return self

    def build_candidate_index(
        self,
        s2_s3_df: pd.DataFrame,
        batch_size: int = 10000,
    ) -> "TfidfFaissBlocker":
        """Transforms Source 2 + Source 3 texts and builds a FAISS IndexFlatIP (cosine)."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted before building candidate index. Call fit() first.")

        # Store candidate entity IDs and sources
        self.candidate_ids = s2_s3_df["entity_id"].astype(str).tolist()

        if "source" in s2_s3_df.columns:
            self.candidate_sources = s2_s3_df["source"].astype(str).tolist()
        else:
            self.candidate_sources = [
                infer_entity_source(eid) for eid in self.candidate_ids
            ]

        # Extract concatenated name + address texts
        texts = prepare_text_series(s2_s3_df)

        # Initialize FAISS IndexFlatIP
        self.faiss_index = faiss.IndexFlatIP(self.dim)

        # Add vectors in batches to conserve memory
        n_records = len(s2_s3_df)
        for start_idx in range(0, n_records, batch_size):
            end_idx = min(start_idx + batch_size, n_records)
            chunk_texts = texts.iloc[start_idx:end_idx]

            tfidf_chunk = self.vectorizer.transform(chunk_texts)
            dense_chunk = np.ascontiguousarray(tfidf_chunk.toarray(), dtype=np.float32)
            faiss.normalize_L2(dense_chunk)
            self.faiss_index.add(dense_chunk)

        return self

    def fit_and_index(
        self,
        s1_df: pd.DataFrame,
        s2_s3_df: pd.DataFrame,
        batch_size: int = 10000,
    ) -> "TfidfFaissBlocker":
        """Fits TF-IDF across all S1, S2, S3 records and builds FAISS index over S2+S3."""
        s1_texts = prepare_text_series(s1_df)
        cand_texts = prepare_text_series(s2_s3_df)
        all_texts = pd.concat([s1_texts, cand_texts], ignore_index=True)

        self.fit_vectorizer(all_texts)
        self.build_candidate_index(s2_s3_df, batch_size=batch_size)
        return self

    def retrieve_candidates(
        self,
        s1_df: pd.DataFrame,
        k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        batch_size: int = 2048,
        include_similarity: bool = False,
    ) -> pd.DataFrame:
        """Retrieves top-K nearest neighbors from S2+S3 for each Source 1 entity."""
        if self.faiss_index is None or self.vectorizer is None:
            raise ValueError("Index is not built. Call fit_and_index() before querying.")

        top_k = k if k is not None else self.k
        sim_thresh = min_similarity if min_similarity is not None else self.min_similarity

        if self.faiss_index.ntotal == 0:
            cols = list(CANDIDATE_PAIR_COLUMNS)
            if include_similarity:
                cols.append("similarity_score")
            return pd.DataFrame(columns=cols)

        # Search up to available records in index
        k_search = min(top_k, self.faiss_index.ntotal)

        s1_ids = s1_df["entity_id"].astype(str).tolist()
        s1_texts = prepare_text_series(s1_df)

        pairs_s1: List[str] = []
        pairs_cand: List[str] = []
        pairs_source: List[str] = []
        pairs_sim: List[float] = []

        n_s1 = len(s1_df)
        for start_idx in range(0, n_s1, batch_size):
            end_idx = min(start_idx + batch_size, n_s1)
            chunk_texts = s1_texts.iloc[start_idx:end_idx]
            chunk_ids = s1_ids[start_idx:end_idx]

            tfidf_chunk = self.vectorizer.transform(chunk_texts)
            query_vectors = np.ascontiguousarray(tfidf_chunk.toarray(), dtype=np.float32)
            faiss.normalize_L2(query_vectors)

            distances, indices = self.faiss_index.search(query_vectors, k_search)

            for i, s1_id in enumerate(chunk_ids):
                for score, cand_idx in zip(distances[i], indices[i]):
                    if cand_idx < 0:
                        continue
                    if score < sim_thresh:
                        continue

                    cand_id = self.candidate_ids[cand_idx]
                    cand_src = self.candidate_sources[cand_idx]

                    pairs_s1.append(s1_id)
                    pairs_cand.append(cand_id)
                    pairs_source.append(cand_src)
                    if include_similarity:
                        pairs_sim.append(float(score))

        data = {
            "source1_entity_id": pairs_s1,
            "candidate_entity_id": pairs_cand,
            "source": pairs_source,
        }
        if include_similarity:
            data["similarity_score"] = pairs_sim

        df_candidates = pd.DataFrame(data).drop_duplicates(
            subset=["source1_entity_id", "candidate_entity_id"]
        ).reset_index(drop=True)

        return df_candidates


def tfidf_faiss_blocking(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: Optional[pd.DataFrame] = None,
    k: int = 20,
    min_similarity: float = 0.0,
    max_features: Optional[int] = 50000,
    batch_size: int = 2048,
    include_similarity: bool = False,
) -> pd.DataFrame:
    """Generates candidate pairs using a character 3-gram TF-IDF FAISS blocker.

    Fits character 3-gram TF-IDF vectorizer across S1, S2, S3 records, builds
    a FAISS cosine index over S2+S3, and retrieves top-K nearest neighbors.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Source 1 records with entity_id, and normalized_name/normalized_address.
    s2_df : pd.DataFrame
        Source 2 records (or combined S2+S3 if s3_df is None).
    s3_df : pd.DataFrame or None, optional
        Source 3 records if provided separately.
    k : int, default=20
        Number of nearest neighbors to retrieve per Source 1 record.
    min_similarity : float, default=0.0
        Minimum cosine similarity threshold to retain a candidate pair.
    max_features : int or None, default=50000
        Maximum vocabulary size for TF-IDF character n-grams.
    batch_size : int, default=2048
        Chunk size for FAISS batch operations to avoid excessive memory use.
    include_similarity : bool, default=False
        If True, appends a 'similarity_score' column with cosine similarity.

    Returns
    -------
    pd.DataFrame
        DataFrame of candidate pairs adhering to CandidatePair schema:
        ['source1_entity_id', 'candidate_entity_id', 'source']
    """
    if s3_df is not None:
        s2_s3 = pd.concat([s2_df, s3_df], ignore_index=True)
    else:
        s2_s3 = s2_df.copy()

    blocker = TfidfFaissBlocker(
        ngram_range=(3, 3),
        max_features=max_features,
        k=k,
        min_similarity=min_similarity,
    )
    blocker.fit_and_index(s1_df, s2_s3, batch_size=batch_size)

    candidates_df = blocker.retrieve_candidates(
        s1_df=s1_df,
        k=k,
        min_similarity=min_similarity,
        batch_size=batch_size,
        include_similarity=include_similarity,
    )

    return candidates_df


def to_candidate_dict(candidate_pairs_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Converts a CandidatePair DataFrame into a mapping of s1_id -> set of candidate IDs."""
    cand_dict: Dict[str, Set[str]] = {}
    for _, row in candidate_pairs_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        cand = str(row["candidate_entity_id"]).strip()
        cand_dict.setdefault(s1, set()).add(cand)
    return cand_dict


def evaluate_blocking_recall(
    candidate_pairs_df: pd.DataFrame,
    gt_dict: Dict[str, Set[str]],
    total_s1_count: Optional[int] = None,
) -> Tuple[float, float, int]:
    """Calculates recall and candidate pool statistics against ground truth."""
    cand_map = to_candidate_dict(candidate_pairs_df)

    total_true_matches = 0
    retained_true_matches = 0
    total_candidates = len(candidate_pairs_df)

    for s1_id, true_matches in gt_dict.items():
        total_true_matches += len(true_matches)
        cands = cand_map.get(s1_id, set())
        retained_true_matches += len(true_matches & cands)

    num_s1 = total_s1_count if total_s1_count is not None else len(gt_dict)
    recall = (retained_true_matches / total_true_matches) if total_true_matches > 0 else 0.0
    avg_cands = (total_candidates / num_s1) if num_s1 > 0 else 0.0

    print("=" * 65)
    print("TF-IDF Character 3-Gram FAISS Blocking Evaluation")
    print("=" * 65)
    print(f"Total Candidate Pairs Generated: {total_candidates:,}")
    print(f"Total Evaluated S1 Entities:     {num_s1:,}")
    print(f"Average Candidates per S1:       {avg_cands:.2f}")
    print(f"Total True Matches in GT:        {total_true_matches:,}")
    print(f"Retained True Matches (Hits):    {retained_true_matches:,}")
    print(f"Estimated Blocking Recall:       {recall * 100:.2f}%")
    print("=" * 65)

    return recall, avg_cands, total_candidates


# =============================================================================
# Benchmark & Standalone Execution Block
# =============================================================================
def _create_benchmark_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    """Generates benchmark sample normalized records for verification."""
    s1_data = [
        {
            "entity_id": "s1_001",
            "business_name": "Walmart Inc.",
            "business_address": "702 SW 8th St, Bentonville, AR 72716",
            "country": "US",
            "normalized_name": "walmart incorporated",
            "normalized_address": "702 southwest 8th street bentonville ar 72716",
        },
        {
            "entity_id": "s1_002",
            "business_name": "Tata Consultancy Services Ltd.",
            "business_address": "TCS House, Fort, Mumbai 400001",
            "country": "India",
            "normalized_name": "tata consultancy services limited",
            "normalized_address": "tcs house fort mumbai 400001",
        },
        {
            "entity_id": "s1_003",
            "business_name": "McDonald's Fast Food",
            "business_address": "110 N Carpenter St, Chicago, IL 60607",
            "country": "US",
            "normalized_name": "mcdonalds fast food",
            "normalized_address": "110 north carpenter street chicago il 60607",
        },
        {
            "entity_id": "s1_004",
            "business_name": "Singleton Bakery Ltd",
            "business_address": "12 High Rd",
            "country": "UK",
            "normalized_name": "singleton bakery limited",
            "normalized_address": "12 high road",
        },
    ]

    s2_s3_data = [
        {
            "entity_id": "s2_001",
            "business_name": "Walmart Inc.",
            "business_address": "702 SW 8th Street, 72716",
            "country": "US",
            "normalized_name": "walmart incorporated",
            "normalized_address": "702 southwest 8th street 72716",
            "source": "S2",
        },
        {
            "entity_id": "s2_002",
            "business_name": "Tata Consultancy Services Ltd",
            "business_address": "Mumbai 400001",
            "country": "India",
            "normalized_name": "tata consultancy services limited",
            "normalized_address": "mumbai 400001",
            "source": "S2",
        },
        {
            "entity_id": "s3_001",
            "business_name": "McDonald's Restaurant",
            "business_address": "110 Carpenter Ave",
            "country": "US",
            "normalized_name": "mcdonalds restaurant",
            "normalized_address": "110 carpenter avenue",
            "source": "S3",
        },
        {
            "entity_id": "s3_002",
            "business_name": "Unrelated Shop",
            "business_address": "50 Main Rd",
            "country": "India",
            "normalized_name": "unrelated shop",
            "normalized_address": "50 main road",
            "source": "S3",
        },
    ]

    gt = {
        "s1_001": {"s2_001"},
        "s1_002": {"s2_002"},
        "s1_003": {"s3_001"},
        "s1_004": set(),
    }

    return pd.DataFrame(s1_data), pd.DataFrame(s2_s3_data), gt


def _find_dataset_train_dir() -> Optional[Path]:
    """Finds training dataset directory across standard paths."""
    candidates = [
        Path("student_resource/dataset/train"),
        Path("dataset/train"),
        Path("../dataset/train"),
        Path("../../dataset/train"),
    ]
    for p in candidates:
        if p.exists() and (p / "train_source1.tsv").exists():
            return p.resolve()
    return None


def load_ground_truth_sample(
    gt_path: Union[str, Path],
    s1_ids: Set[str],
) -> Dict[str, Set[str]]:
    """Fast ground-truth parser filtering strictly for evaluated S1 entities."""
    gt_df = pd.read_csv(gt_path, sep="\t")
    filtered = gt_df[gt_df["source1_entity_id"].astype(str).isin(s1_ids)]
    gt_dict: Dict[str, Set[str]] = {s1: set() for s1 in s1_ids}
    for s1_id, m_str in zip(filtered["source1_entity_id"], filtered["matched_entity_ids"]):
        s1_id_str = str(s1_id).strip()
        if pd.notna(m_str) and str(m_str).strip():
            gt_dict[s1_id_str] = {m.strip() for m in str(m_str).split(",") if m.strip()}
    return gt_dict


if __name__ == "__main__":
    train_dir = _find_dataset_train_dir()
    if train_dir:
        print(f"Found dataset directory at: {train_dir}")
        print("Loading sample of 5,000 records per source for benchmark...")
        s1 = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", nrows=5000)
        s2 = pd.read_csv(train_dir / "train_source2.tsv", sep="\t", nrows=5000)
        s3 = pd.read_csv(train_dir / "train_source3.tsv", sep="\t", nrows=5000)

        # Standardize normalization columns if not already computed
        from src.normalize import normalize_name, normalize_address  # type: ignore

        for df in [s1, s2, s3]:
            df["normalized_name"] = df["business_name"].apply(normalize_name)
            df["normalized_address"] = df["business_address"].apply(normalize_address)

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
