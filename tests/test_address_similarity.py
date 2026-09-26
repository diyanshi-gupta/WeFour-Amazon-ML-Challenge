"""
tests/test_address_similarity.py

Pytest suite for src/features/address_similarity.py (Commit 2.2).
Covers:
  - Low-level Jaccard token overlap & TF-IDF cosine similarity helpers
  - Missing and sparse address edge case handling (sentinel -1.0)
  - Boolean has_address flag behavior
  - Independent address vectorizer creation and fitting
  - DataFrame generation via compute_address_similarity_features()
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.features.address_similarity import (
    address_tfidf_cosine,
    address_token_jaccard,
    compute_address_similarity_features,
    create_address_vectorizer,
    fit_address_vectorizer,
    _is_empty_or_sparse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_s1_df():
    return pd.DataFrame(
        {
            "entity_id": ["S1-001", "S1-002", "S1-003", "S1-004", "S1-005"],
            "normalized_address": [
                "702 southwest 8th street bentonville 72716",   # standard address
                "1000 nicollet mall minneapolis 55403",         # standard address
                "",                                             # empty string
                None,                                           # None value
                "---",                                          # sparse / symbols only
            ],
        }
    )


@pytest.fixture
def sample_cand_df():
    return pd.DataFrame(
        {
            "entity_id": ["S2-001", "S2-002", "S2-003", "S2-004", "S2-005"],
            "normalized_address": [
                "702 southwest 8th street bentonville 72716",   # exact match
                "702 southwest 8th st suite 100 bentonville",   # noisy variant
                "unrelated address completely disjoint",        # non-matching address
                "",                                             # empty string
                float("nan"),                                   # NaN value
            ],
        }
    )


@pytest.fixture
def sample_pairs():
    return [
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-001"},  # Exact match
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-002"},  # High similarity
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-003"},  # Valid but disjoint
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-004"},  # Candidate empty
        {"source1_entity_id": "S1-003", "candidate_entity_id": "S2-001"},  # S1 empty
        {"source1_entity_id": "S1-004", "candidate_entity_id": "S2-005"},  # S1 None, Cand NaN
        {"source1_entity_id": "S1-005", "candidate_entity_id": "S2-001"},  # S1 sparse symbol
    ]


# ---------------------------------------------------------------------------
# Unit Tests: Low-level Helpers & Sentinel Handling
# ---------------------------------------------------------------------------


class TestAddressHelpers:
    def test_is_empty_or_sparse(self):
        assert _is_empty_or_sparse(None) is True
        assert _is_empty_or_sparse("") is True
        assert _is_empty_or_sparse("   ") is True
        assert _is_empty_or_sparse(float("nan")) is True
        assert _is_empty_or_sparse(np.nan) is True
        assert _is_empty_or_sparse("nan") is True
        assert _is_empty_or_sparse("---") is True
        assert _is_empty_or_sparse("!!@#") is True
        assert _is_empty_or_sparse("702 sw 8th st") is False


class TestAddressTokenJaccard:
    def test_exact_match(self):
        score = address_token_jaccard("702 sw 8th st", "702 sw 8th st")
        assert score == 1.0

    def test_partial_match(self):
        score = address_token_jaccard("702 sw 8th st bentonville", "702 sw 8th st suite 100")
        assert 0.0 < score < 1.0

    def test_disjoint_match_returns_zero(self):
        # When both addresses exist but share zero tokens, return 0.0 (not sentinel -1.0)
        score = address_token_jaccard("apple street", "banana avenue")
        assert score == 0.0

    def test_empty_first_address_sentinel(self):
        assert address_token_jaccard("", "702 sw 8th st") == -1.0
        assert address_token_jaccard(None, "702 sw 8th st") == -1.0

    def test_empty_second_address_sentinel(self):
        assert address_token_jaccard("702 sw 8th st", "") == -1.0
        assert address_token_jaccard("702 sw 8th st", float("nan")) == -1.0

    def test_both_empty_sentinel(self):
        assert address_token_jaccard("", "") == -1.0
        assert address_token_jaccard(None, None) == -1.0

    def test_sparse_symbol_sentinel(self):
        assert address_token_jaccard("---", "702 sw 8th st") == -1.0


class TestAddressTfidfCosine:
    def test_exact_match(self):
        score = address_tfidf_cosine("702 sw 8th st", "702 sw 8th st")
        assert score == 1.0

    def test_partial_match(self):
        score = address_tfidf_cosine(
            "702 southwest 8th street bentonville",
            "702 southwest 8th street suite 100 bentonville"
        )
        assert 0.4 < score < 1.0

    def test_disjoint_match_returns_zero(self):
        score = address_tfidf_cosine("100 first ave", "900 second boulevard")
        assert score == 0.0

    def test_empty_first_sentinel(self):
        assert address_tfidf_cosine("", "702 sw 8th st") == -1.0
        assert address_tfidf_cosine(None, "702 sw 8th st") == -1.0

    def test_empty_second_sentinel(self):
        assert address_tfidf_cosine("702 sw 8th st", "") == -1.0
        assert address_tfidf_cosine("702 sw 8th st", float("nan")) == -1.0

    def test_both_empty_sentinel(self):
        assert address_tfidf_cosine("", "") == -1.0
        assert address_tfidf_cosine(None, None) == -1.0

    def test_with_custom_fitted_vectorizer(self):
        corpus = [
            "702 southwest 8th street bentonville",
            "1000 nicollet mall minneapolis",
            "7601 penn ave south richfield",
        ]
        vec = fit_address_vectorizer(corpus)
        score = address_tfidf_cosine(
            "702 southwest 8th street bentonville",
            "702 southwest 8th street",
            vectorizer=vec
        )
        assert 0.5 < score <= 1.0


# ---------------------------------------------------------------------------
# Integration Tests: compute_address_similarity_features()
# ---------------------------------------------------------------------------


class TestComputeAddressSimilarityFeatures:
    def test_output_shape_and_columns(self, sample_pairs, sample_s1_df, sample_cand_df):
        df = compute_address_similarity_features(sample_pairs, sample_s1_df, sample_cand_df)
        assert len(df) == len(sample_pairs)
        expected_cols = [
            "source1_entity_id",
            "candidate_entity_id",
            "address_token_jaccard",
            "address_tfidf_cosine",
            "has_address",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_exact_match_row(self, sample_pairs, sample_s1_df, sample_cand_df):
        df = compute_address_similarity_features(sample_pairs, sample_s1_df, sample_cand_df)
        exact_row = df[
            (df["source1_entity_id"] == "S1-001") & (df["candidate_entity_id"] == "S2-001")
        ].iloc[0]

        assert bool(exact_row["has_address"]) is True
        assert exact_row["address_token_jaccard"] == 1.0
        assert exact_row["address_tfidf_cosine"] == 1.0

    def test_noisy_variant_row(self, sample_pairs, sample_s1_df, sample_cand_df):
        df = compute_address_similarity_features(sample_pairs, sample_s1_df, sample_cand_df)
        noisy_row = df[
            (df["source1_entity_id"] == "S1-001") & (df["candidate_entity_id"] == "S2-002")
        ].iloc[0]

        assert bool(noisy_row["has_address"]) is True
        assert noisy_row["address_token_jaccard"] > 0.3
        assert noisy_row["address_tfidf_cosine"] > 0.3

    def test_disjoint_row(self, sample_pairs, sample_s1_df, sample_cand_df):
        df = compute_address_similarity_features(sample_pairs, sample_s1_df, sample_cand_df)
        disjoint_row = df[
            (df["source1_entity_id"] == "S1-001") & (df["candidate_entity_id"] == "S2-003")
        ].iloc[0]

        assert bool(disjoint_row["has_address"]) is True
        assert disjoint_row["address_token_jaccard"] == 0.0
        assert disjoint_row["address_tfidf_cosine"] == 0.0

    def test_candidate_missing_address_sentinel(self, sample_pairs, sample_s1_df, sample_cand_df):
        df = compute_address_similarity_features(sample_pairs, sample_s1_df, sample_cand_df)
        missing_row = df[
            (df["source1_entity_id"] == "S1-001") & (df["candidate_entity_id"] == "S2-004")
        ].iloc[0]

        assert bool(missing_row["has_address"]) is False
        assert missing_row["address_token_jaccard"] == -1.0
        assert missing_row["address_tfidf_cosine"] == -1.0

    def test_s1_missing_address_sentinel(self, sample_pairs, sample_s1_df, sample_cand_df):
        df = compute_address_similarity_features(sample_pairs, sample_s1_df, sample_cand_df)
        missing_row = df[
            (df["source1_entity_id"] == "S1-003") & (df["candidate_entity_id"] == "S2-001")
        ].iloc[0]

        assert bool(missing_row["has_address"]) is False
        assert missing_row["address_token_jaccard"] == -1.0
        assert missing_row["address_tfidf_cosine"] == -1.0

    def test_both_missing_sentinel(self, sample_pairs, sample_s1_df, sample_cand_df):
        df = compute_address_similarity_features(sample_pairs, sample_s1_df, sample_cand_df)
        both_missing_row = df[
            (df["source1_entity_id"] == "S1-004") & (df["candidate_entity_id"] == "S2-005")
        ].iloc[0]

        assert bool(both_missing_row["has_address"]) is False
        assert both_missing_row["address_token_jaccard"] == -1.0
        assert both_missing_row["address_tfidf_cosine"] == -1.0

    def test_sparse_symbols_sentinel(self, sample_pairs, sample_s1_df, sample_cand_df):
        df = compute_address_similarity_features(sample_pairs, sample_s1_df, sample_cand_df)
        sparse_row = df[
            (df["source1_entity_id"] == "S1-005") & (df["candidate_entity_id"] == "S2-001")
        ].iloc[0]

        assert bool(sparse_row["has_address"]) is False
        assert sparse_row["address_token_jaccard"] == -1.0
        assert sparse_row["address_tfidf_cosine"] == -1.0

    def test_candidate_pairs_as_dataframe(self, sample_pairs, sample_s1_df, sample_cand_df):
        pairs_df = pd.DataFrame(sample_pairs)
        df = compute_address_similarity_features(pairs_df, sample_s1_df, sample_cand_df)
        assert len(df) == len(sample_pairs)
        assert "address_token_jaccard" in df.columns

    def test_empty_candidate_pairs(self, sample_s1_df, sample_cand_df):
        df = compute_address_similarity_features([], sample_s1_df, sample_cand_df)
        assert len(df) == 0
        assert list(df.columns) == [
            "source1_entity_id",
            "candidate_entity_id",
            "address_token_jaccard",
            "address_tfidf_cosine",
            "has_address",
        ]
