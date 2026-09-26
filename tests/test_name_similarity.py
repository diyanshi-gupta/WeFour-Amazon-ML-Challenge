"""
tests/test_name_similarity.py

Pytest suite for src/features/name_similarity.py (Commit 2.1).
Covers:
  - Low-level metric helpers (boundary + accuracy checks)
  - compute_name_similarity_features() DataFrame output
  - Edge cases: null / empty names, exact matches, completely different names
"""

import math

import pandas as pd
import pytest

from src.features.name_similarity import (
    compute_name_similarity_features,
    jaro_winkler,
    monge_elkan,
    normalized_levenshtein,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_s1_df():
    return pd.DataFrame(
        {
            "entity_id": ["S1-001", "S1-002", "S1-003"],
            "normalized_name": ["acme private limited", "global tech services", ""],
        }
    )


@pytest.fixture
def simple_cand_df():
    return pd.DataFrame(
        {
            "entity_id": ["S2-001", "S2-002", "S2-003", "S2-004"],
            "normalized_name": [
                "acme private limited",   # exact match
                "acme pvt ltd",           # noisy variant
                "global technology services",  # expanded variant
                "",                       # missing name
            ],
        }
    )


@pytest.fixture
def simple_pairs():
    return [
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-001"},
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-002"},
        {"source1_entity_id": "S1-002", "candidate_entity_id": "S2-003"},
        {"source1_entity_id": "S1-003", "candidate_entity_id": "S2-004"},
    ]


# ---------------------------------------------------------------------------
# Low-level helper tests
# ---------------------------------------------------------------------------


class TestJaroWinkler:
    def test_exact_match_returns_one(self):
        assert jaro_winkler("acme corp", "acme corp") == 1.0

    def test_empty_a_returns_zero(self):
        assert jaro_winkler("", "acme corp") == 0.0

    def test_empty_b_returns_zero(self):
        assert jaro_winkler("acme corp", "") == 0.0

    def test_both_empty_returns_zero(self):
        assert jaro_winkler("", "") == 0.0

    def test_none_treated_as_empty(self):
        assert jaro_winkler(None, "acme") == 0.0

    def test_similar_strings_high_score(self):
        score = jaro_winkler("acme private limited", "acme pvt ltd")
        assert score > 0.7, f"Expected > 0.7, got {score}"

    def test_completely_different_strings_low_score(self):
        score = jaro_winkler("apple", "xqzjk")
        assert score < 0.6, f"Expected < 0.6, got {score}"

    def test_range_is_zero_to_one(self):
        score = jaro_winkler("hello world inc", "world hello incorporated")
        assert 0.0 <= score <= 1.0


class TestNormalizedLevenshtein:
    def test_exact_match_returns_one(self):
        assert normalized_levenshtein("acme corp", "acme corp") == 1.0

    def test_both_empty_returns_sentinel(self):
        assert normalized_levenshtein("", "") == -1.0

    def test_one_empty_low_score(self):
        score = normalized_levenshtein("", "acme corp")
        assert score == 0.0

    def test_one_char_diff(self):
        # "acme" vs "acmf" -> edit=1, max=4 -> 1 - 0.25 = 0.75
        score = normalized_levenshtein("acme", "acmf")
        assert abs(score - 0.75) < 1e-4

    def test_range_zero_to_one(self):
        score = normalized_levenshtein("hello", "world")
        assert 0.0 <= score <= 1.0


class TestMongeElkan:
    def test_exact_match_returns_one(self):
        assert monge_elkan("acme corp", "acme corp") == 1.0

    def test_empty_a_returns_zero(self):
        assert monge_elkan("", "acme corp") == 0.0

    def test_empty_b_returns_zero(self):
        assert monge_elkan("acme corp", "") == 0.0

    def test_both_empty_returns_zero(self):
        assert monge_elkan("", "") == 0.0

    def test_token_reorder_high_score(self):
        # Token sets are the same, just reordered — should be very high
        score = monge_elkan("global tech services", "services tech global")
        assert score > 0.9, f"Expected > 0.9, got {score}"

    def test_noisy_variant_moderate_score(self):
        score = monge_elkan("acme private limited", "acme pvt ltd")
        assert score > 0.6, f"Expected > 0.6, got {score}"

    def test_range_zero_to_one(self):
        score = monge_elkan("alpha beta", "gamma delta")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# DataFrame-level tests
# ---------------------------------------------------------------------------


class TestComputeNameSimilarityFeatures:
    def test_output_shape(self, simple_pairs, simple_s1_df, simple_cand_df):
        df = compute_name_similarity_features(simple_pairs, simple_s1_df, simple_cand_df)
        assert len(df) == len(simple_pairs)

    def test_required_columns_present(self, simple_pairs, simple_s1_df, simple_cand_df):
        df = compute_name_similarity_features(simple_pairs, simple_s1_df, simple_cand_df)
        for col in [
            "source1_entity_id",
            "candidate_entity_id",
            "jaro_winkler_similarity",
            "normalized_levenshtein",
            "monge_elkan_similarity",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    def test_exact_match_pair_scores_are_one(self, simple_pairs, simple_s1_df, simple_cand_df):
        df = compute_name_similarity_features(simple_pairs, simple_s1_df, simple_cand_df)
        exact = df[
            (df["source1_entity_id"] == "S1-001") &
            (df["candidate_entity_id"] == "S2-001")
        ]
        assert len(exact) == 1
        assert exact["jaro_winkler_similarity"].iloc[0] == 1.0
        assert exact["normalized_levenshtein"].iloc[0] == 1.0
        assert exact["monge_elkan_similarity"].iloc[0] == 1.0

    def test_both_empty_names_sentinel(self, simple_pairs, simple_s1_df, simple_cand_df):
        df = compute_name_similarity_features(simple_pairs, simple_s1_df, simple_cand_df)
        both_empty = df[
            (df["source1_entity_id"] == "S1-003") &
            (df["candidate_entity_id"] == "S2-004")
        ]
        assert len(both_empty) == 1
        assert both_empty["normalized_levenshtein"].iloc[0] == -1.0
        assert both_empty["jaro_winkler_similarity"].iloc[0] == 0.0
        assert both_empty["monge_elkan_similarity"].iloc[0] == 0.0

    def test_noisy_variant_scores_are_high(self, simple_pairs, simple_s1_df, simple_cand_df):
        df = compute_name_similarity_features(simple_pairs, simple_s1_df, simple_cand_df)
        noisy = df[
            (df["source1_entity_id"] == "S1-001") &
            (df["candidate_entity_id"] == "S2-002")
        ]
        assert noisy["jaro_winkler_similarity"].iloc[0] > 0.7
        assert noisy["monge_elkan_similarity"].iloc[0] > 0.6

    def test_empty_pairs_returns_empty_dataframe(self, simple_s1_df, simple_cand_df):
        df = compute_name_similarity_features([], simple_s1_df, simple_cand_df)
        assert len(df) == 0
        assert list(df.columns) == [
            "source1_entity_id",
            "candidate_entity_id",
            "jaro_winkler_similarity",
            "normalized_levenshtein",
            "monge_elkan_similarity",
        ]
