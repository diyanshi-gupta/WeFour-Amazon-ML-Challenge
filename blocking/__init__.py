"""Blocking package root."""
from code.business_entity_resolution.src.blocking.tfidf_channel import (
    TfidfFaissBlocker,
    tfidf_faiss_blocking,
    prepare_text_series,
    to_candidate_dict,
    evaluate_blocking_recall,
)
from code.business_entity_resolution.src.blocking.embedding_channel import (
    EmbeddingFaissBlocker,
    embedding_faiss_blocking,
    run_low_tfidf_sanity_check,
)

__all__ = [
    "TfidfFaissBlocker",
    "tfidf_faiss_blocking",
    "prepare_text_series",
    "to_candidate_dict",
    "evaluate_blocking_recall",
    "EmbeddingFaissBlocker",
    "embedding_faiss_blocking",
    "run_low_tfidf_sanity_check",
]

