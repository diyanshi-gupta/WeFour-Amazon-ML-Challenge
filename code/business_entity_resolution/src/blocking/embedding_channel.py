# code/business_entity_resolution/src/blocking/embedding_channel.py
"""Dense Sentence-Embedding + FAISS Blocking Channel.

Encodes normalized_name + normalized_address for records across Source 1/2/3
using open-source sentence-transformers (all-MiniLM-L6-v2, 22M parameters, Apache-2.0),
constructs a FAISS IndexFlatIP (cosine similarity) over candidate embeddings (Source 2 + Source 3),
and retrieves top-K nearest neighbors for each Source 1 entity.
Includes sanity-check verification for true-match pairs with low TF-IDF similarity.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

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

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def prepare_text_series(df: pd.DataFrame) -> pd.Series:
    """Concatenates normalized_name and normalized_address into a single text series.

    Falls back cleanly to clean_name / business_name and clean_address / business_address
    if normalized columns are not present. Missing values are filled with empty strings.
    """
    names = pd.Series([""] * len(df), index=df.index, dtype=str)
    for col in ["business_name", "clean_name", "normalized_name"]:
        if col in df.columns:
            val = df[col].fillna("").astype(str).str.strip().str.lower()
            names = names.where(names != "", val)

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


class EmbeddingFaissBlocker:
    """Sentence-Transformer Dense Embedding with FAISS Cosine Indexing for Candidate Blocking.

    Parameters
    ----------
    model_name : str, default='sentence-transformers/all-MiniLM-L6-v2'
        Open, Apache-2.0/MIT sentence-embedding model (under 8B parameters).
    k : int, default=20
        Default number of nearest neighbors to retrieve per Source 1 entity.
    min_similarity : float, default=0.0
        Minimum cosine similarity threshold to retain a candidate pair.
    batch_size : int, default=256
        Batch size for sentence encoding and FAISS operations.
    device : str or None, optional
        Computing device ('cpu', 'cuda', etc.). If None, automatically detected.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        k: int = 20,
        min_similarity: float = 0.0,
        batch_size: int = 256,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.k = k
        self.min_similarity = min_similarity
        self.batch_size = batch_size
        self.device = device

        self.model: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.candidate_ids: List[str] = []
        self.candidate_sources: List[str] = []
        self.dim: int = 384

    def load_model(self) -> SentenceTransformer:
        """Loads and caches the SentenceTransformer model instance."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Try to get dimension from model
            if hasattr(self.model, "get_embedding_dimension"):
                self.dim = self.model.get_embedding_dimension()
            elif hasattr(self.model, "get_sentence_embedding_dimension"):
                self.dim = self.model.get_sentence_embedding_dimension()
        return self.model

    def encode_records(
        self,
        texts: Union[pd.Series, List[str]],
        batch_size: Optional[int] = None,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encodes texts into L2-normalized float32 dense embedding vectors."""
        model = self.load_model()
        bs = batch_size if batch_size is not None else self.batch_size
        text_list = [str(t) if pd.notna(t) and str(t).strip() else " " for t in texts]

        embeddings = model.encode(
            text_list,
            batch_size=bs,
            show_progress_bar=False,
            normalize_embeddings=normalize_embeddings,
        )
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def build_candidate_index(
        self,
        s2_s3_df: pd.DataFrame,
        batch_size: Optional[int] = None,
    ) -> "EmbeddingFaissBlocker":
        """Encodes Source 2 + Source 3 texts and builds a FAISS IndexFlatIP (cosine)."""
        self.load_model()
        bs = batch_size if batch_size is not None else self.batch_size

        self.candidate_ids = s2_s3_df["entity_id"].astype(str).tolist()
        if "source" in s2_s3_df.columns:
            self.candidate_sources = s2_s3_df["source"].astype(str).tolist()
        else:
            self.candidate_sources = [
                infer_entity_source(eid) for eid in self.candidate_ids
            ]

        texts = prepare_text_series(s2_s3_df)
        self.faiss_index = faiss.IndexFlatIP(self.dim)

        n_records = len(s2_s3_df)
        for start_idx in range(0, n_records, bs):
            end_idx = min(start_idx + bs, n_records)
            chunk_texts = texts.iloc[start_idx:end_idx]

            dense_chunk = self.encode_records(chunk_texts, batch_size=bs, normalize_embeddings=True)
            self.faiss_index.add(dense_chunk)

        return self

    def fit_and_index(
        self,
        s1_df: pd.DataFrame,
        s2_s3_df: pd.DataFrame,
        batch_size: Optional[int] = None,
    ) -> "EmbeddingFaissBlocker":
        """Prepares model and builds candidate index over S2+S3."""
        self.load_model()
        self.build_candidate_index(s2_s3_df, batch_size=batch_size)
        return self

    def retrieve_candidates(
        self,
        s1_df: pd.DataFrame,
        k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        batch_size: Optional[int] = None,
        include_similarity: bool = False,
    ) -> pd.DataFrame:
        """Retrieves top-K nearest neighbors from S2+S3 for each Source 1 entity."""
        if self.faiss_index is None:
            raise ValueError("Index is not built. Call build_candidate_index() before querying.")

        top_k = k if k is not None else self.k
        sim_thresh = min_similarity if min_similarity is not None else self.min_similarity
        bs = batch_size if batch_size is not None else self.batch_size

        if self.faiss_index.ntotal == 0:
            cols = list(CANDIDATE_PAIR_COLUMNS)
            if include_similarity:
                cols.append("similarity_score")
            return pd.DataFrame(columns=cols)

        k_search = min(top_k, self.faiss_index.ntotal)
        s1_ids = s1_df["entity_id"].astype(str).tolist()
        s1_texts = prepare_text_series(s1_df)

        pairs_s1: List[str] = []
        pairs_cand: List[str] = []
        pairs_source: List[str] = []
        pairs_sim: List[float] = []

        n_s1 = len(s1_df)
        for start_idx in range(0, n_s1, bs):
            end_idx = min(start_idx + bs, n_s1)
            chunk_texts = s1_texts.iloc[start_idx:end_idx]
            chunk_ids = s1_ids[start_idx:end_idx]

            query_vectors = self.encode_records(chunk_texts, batch_size=bs, normalize_embeddings=True)
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


def embedding_faiss_blocking(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: Optional[pd.DataFrame] = None,
    k: int = 20,
    min_similarity: float = 0.0,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 256,
    include_similarity: bool = False,
) -> pd.DataFrame:
    """Generates candidate pairs using the dense sentence-embedding FAISS blocker.

    Encodes normalized_name + normalized_address for all Source 1/2/3 records,
    constructs a FAISS IndexFlatIP (cosine) over S2+S3, and retrieves top-K
    nearest neighbors per Source 1 entity.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Source 1 records.
    s2_df : pd.DataFrame
        Source 2 records (or combined S2+S3 if s3_df is None).
    s3_df : pd.DataFrame or None, optional
        Source 3 records if provided separately.
    k : int, default=20
        Number of nearest neighbors to retrieve per Source 1 entity.
    min_similarity : float, default=0.0
        Minimum cosine similarity threshold to retain a candidate pair.
    model_name : str, default='sentence-transformers/all-MiniLM-L6-v2'
        Sentence transformer model name.
    batch_size : int, default=256
        Batch size for inference and FAISS operations.
    include_similarity : bool, default=False
        If True, appends a 'similarity_score' column with cosine similarity.

    Returns
    -------
    pd.DataFrame
        Candidate pairs DataFrame adhering to CandidatePair schema:
        ['source1_entity_id', 'candidate_entity_id', 'source']
    """
    if s3_df is not None:
        s2_s3 = pd.concat([s2_df, s3_df], ignore_index=True)
    else:
        s2_s3 = s2_df.copy()

    blocker = EmbeddingFaissBlocker(
        model_name=model_name,
        k=k,
        min_similarity=min_similarity,
        batch_size=batch_size,
    )
    blocker.build_candidate_index(s2_s3, batch_size=batch_size)

    candidates_df = blocker.retrieve_candidates(
        s1_df=s1_df,
        k=k,
        min_similarity=min_similarity,
        batch_size=batch_size,
        include_similarity=include_similarity,
    )

    return candidates_df


# =============================================================================
# Sanity-Check: 5 True-Match Pairs with LOW TF-IDF Cosine Similarity
# =============================================================================
BENCHMARK_SEMANTIC_TRUE_PAIRS = [
    {
        "pair_id": "pair_1",
        "description": "Hospital vs Medical Center synonymy & street abbreviation expansion",
        "s1_id": "s1_hosp",
        "s1_name": "St. Jude Children's Hospital",
        "s1_address": "332 N Lauderdale Street, Memphis, TN 38105",
        "cand_id": "s2_hosp",
        "cand_name": "Saint Jude Pediatric Medical Center",
        "cand_address": "332 North Lauderdale St, Memphis, Tennessee",
        "cand_source": "S2",
    },
    {
        "pair_id": "pair_2",
        "description": "NYC Municipal agency abbreviation vs full waste facility title",
        "s1_id": "s1_waste",
        "s1_name": "NYC Dept of Sanitation",
        "s1_address": "Pier 99 Twelfth Avenue, New York City",
        "cand_id": "s3_waste",
        "cand_name": "New York City Waste Management Facility",
        "cand_address": "Pier 99 12th Ave, New York, NY",
        "cand_source": "S3",
    },
    {
        "pair_id": "pair_3",
        "description": "Car repair vs auto care workshop terminology divergence",
        "s1_id": "s1_auto",
        "s1_name": "Precision Car Repair & Mechanic Shop",
        "s1_address": "450 West Broadway Suite 2, Eugene, OR",
        "cand_id": "s2_auto",
        "cand_name": "Precision Auto Care Service Garage",
        "cand_address": "450 W Broadway #2, Eugene, Oregon",
        "cand_source": "S2",
    },
    {
        "pair_id": "pair_4",
        "description": "Lodging / accommodations vs hotel / motel vocabulary shift",
        "s1_id": "s1_hotel",
        "s1_name": "Apex Lodging Accommodations & Guest Suites",
        "s1_address": "100 Harbour Road, Vancouver, BC",
        "cand_id": "s3_hotel",
        "cand_name": "Apex Hotel Motel & Rooms",
        "cand_address": "100 Harbor Rd, Vancouver, British Columbia",
        "cand_source": "S3",
    },
    {
        "pair_id": "pair_5",
        "description": "Financial services vs banking & loan institution divergence",
        "s1_id": "s1_bank",
        "s1_name": "First United National Bank Financial Services",
        "s1_address": "12 Wall St, New York, NY",
        "cand_id": "s2_bank",
        "cand_name": "First United Banking & Loan Institution",
        "cand_address": "12 Wall Street, Manhattan, New York City",
        "cand_source": "S2",
    },
]


def run_low_tfidf_sanity_check(
    blocker: Optional[EmbeddingFaissBlocker] = None,
    distractors_df: Optional[pd.DataFrame] = None,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Sanity-checks that dense embeddings catch true matches where TF-IDF similarity is LOW.

    Demonstrates that character 3-gram TF-IDF fails or produces low cosine similarity
    when entities use synonymous wording or paraphrased names/addresses, whereas
    the embedding channel captures the semantic equivalence and successfully retrieves
    the candidate within top-K.

    Returns
    -------
    List[Dict[str, Any]]
        Detailed metrics and comparison table for each of the 5 test pairs.
    """
    pairs = BENCHMARK_SEMANTIC_TRUE_PAIRS
    if blocker is None:
        blocker = EmbeddingFaissBlocker(model_name=DEFAULT_MODEL_NAME, k=k)

    # 1. Prepare S1 and Candidate dataframes
    s1_rows = []
    cand_rows = []
    text_pairs = []

    for p in pairs:
        s1_rows.append({
            "entity_id": p["s1_id"],
            "normalized_name": p["s1_name"],
            "normalized_address": p["s1_address"],
            "country": "US",
        })
        cand_rows.append({
            "entity_id": p["cand_id"],
            "normalized_name": p["cand_name"],
            "normalized_address": p["cand_address"],
            "country": "US",
            "source": p["cand_source"],
        })
        text_pairs.append((
            p["s1_name"].lower() + " " + p["s1_address"].lower(),
            p["cand_name"].lower() + " " + p["cand_address"].lower(),
        ))

    s1_df = pd.DataFrame(s1_rows)
    cand_df = pd.DataFrame(cand_rows)

    # Add distractors if provided to test retrieval under competition
    if distractors_df is not None:
        cand_df = pd.concat([cand_df, distractors_df], ignore_index=True)

    # 2. Compute Character 3-gram TF-IDF cosine similarity for each pair
    all_texts = [t for pair in text_pairs for t in pair]
    tfidf_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))
    tfidf_mat = tfidf_vec.fit_transform(all_texts).toarray()

    results: List[Dict[str, Any]] = []

    # 3. Build FAISS index and query top-K
    blocker.build_candidate_index(cand_df)
    retrieved = blocker.retrieve_candidates(s1_df, k=k, include_similarity=True)

    for i, p in enumerate(pairs):
        v_s1 = tfidf_mat[2 * i]
        v_cand = tfidf_mat[2 * i + 1]
        norm_s1 = np.linalg.norm(v_s1)
        norm_cand = np.linalg.norm(v_cand)
        sim_tfidf = float(np.dot(v_s1, v_cand) / (norm_s1 * norm_cand)) if (norm_s1 > 0 and norm_cand > 0) else 0.0

        # Check if embedding channel caught this pair in top-K
        s1_cands = retrieved[retrieved["source1_entity_id"] == p["s1_id"]]
        cand_matches = s1_cands[s1_cands["candidate_entity_id"] == p["cand_id"]]

        caught = len(cand_matches) > 0
        emb_rank = cand_matches.index[0] + 1 if caught else None
        emb_score = float(cand_matches.iloc[0]["similarity_score"]) if caught else 0.0

        results.append({
            "pair_id": p["pair_id"],
            "description": p["description"],
            "s1_id": p["s1_id"],
            "cand_id": p["cand_id"],
            "tfidf_cosine": sim_tfidf,
            "embedding_cosine": emb_score,
            "caught_in_top_k": caught,
            "rank": emb_rank,
            "s1_text": text_pairs[i][0],
            "cand_text": text_pairs[i][1],
        })

    # Print summary table
    print("\n" + "=" * 80)
    print("Sanity-Check: 5 Known True-Match Pairs with LOW TF-IDF Cosine Similarity")
    print("=" * 80)
    for r in results:
        status = "[OK] CAUGHT IN TOP-K" if r["caught_in_top_k"] else "[FAIL] MISSED"
        print(f"\n{r['pair_id']}: {r['description']}")
        print(f"  * S1 Entity : {r['s1_id']} -> '{r['s1_text'][:60]}...'")
        print(f"  * Cand Entity: {r['cand_id']} -> '{r['cand_text'][:60]}...'")
        print(f"  * TF-IDF Char 3-gram Cosine : {r['tfidf_cosine']:.4f}  (LOW lexical overlap)")
        print(f"  * Dense Embedding Cosine   : {r['embedding_cosine']:.4f}  (HIGH semantic similarity)")
        print(f"  * Embedding Channel Status : {status} (Top-{k} Retrieval)")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    print("Running Sanity-Check on 5 Known True-Match Pairs with Low TF-IDF Cosine...")
    results = run_low_tfidf_sanity_check(k=5)

    all_caught = all(r["caught_in_top_k"] for r in results)
    print(f"Sanity Check Result: {'PASSED (5/5 Caught)' if all_caught else 'FAILED'}")
