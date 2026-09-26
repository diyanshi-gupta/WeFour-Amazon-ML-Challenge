"""Blocking package root."""
from code.business_entity_resolution.src.blocking.tfidf_channel import (
    TfidfFaissBlocker,
    tfidf_faiss_blocking,
    prepare_text_series,
    to_candidate_dict,
    evaluate_blocking_recall,
)

__all__ = [
    "TfidfFaissBlocker",
    "tfidf_faiss_blocking",
    "prepare_text_series",
    "to_candidate_dict",
    "evaluate_blocking_recall",
]
