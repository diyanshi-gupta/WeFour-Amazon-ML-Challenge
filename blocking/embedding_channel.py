"""blocking/embedding_channel.py

Re-exports and entrypoint for the dense sentence-embedding FAISS blocking channel.
"""
import sys
from pathlib import Path

# Add project root to sys.path if not present
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from code.business_entity_resolution.src.blocking.embedding_channel import (
    EmbeddingFaissBlocker,
    embedding_faiss_blocking,
    prepare_text_series,
    infer_entity_source,
    run_low_tfidf_sanity_check,
    BENCHMARK_SEMANTIC_TRUE_PAIRS,
    DEFAULT_MODEL_NAME,
)

__all__ = [
    "EmbeddingFaissBlocker",
    "embedding_faiss_blocking",
    "prepare_text_series",
    "infer_entity_source",
    "run_low_tfidf_sanity_check",
    "BENCHMARK_SEMANTIC_TRUE_PAIRS",
    "DEFAULT_MODEL_NAME",
]

if __name__ == "__main__":
    print("Running Sanity-Check on 5 Known True-Match Pairs with Low TF-IDF Cosine...")
    results = run_low_tfidf_sanity_check(k=5)

    all_caught = all(r["caught_in_top_k"] for r in results)
    print(f"Sanity Check Result: {'PASSED (5/5 Caught)' if all_caught else 'FAILED'}")
