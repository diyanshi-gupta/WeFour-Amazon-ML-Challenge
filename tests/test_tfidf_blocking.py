# tests/test_tfidf_blocking.py
"""Unit tests for the TF-IDF character 3-gram FAISS blocking channel."""

import unittest
import pandas as pd
from code.business_entity_resolution.src.blocking.tfidf_channel import (
    TfidfFaissBlocker,
    tfidf_faiss_blocking,
    prepare_text_series,
    to_candidate_dict,
    evaluate_blocking_recall,
)
from code.business_entity_resolution.src.schemas import CANDIDATE_PAIR_COLUMNS


class TestTfidfFaissBlocking(unittest.TestCase):
    def setUp(self):
        self.s1_df = pd.DataFrame([
            {
                "entity_id": "s1_001",
                "normalized_name": "walmart incorporated",
                "normalized_address": "702 southwest 8th street bentonville ar 72716",
                "country": "US",
            },
            {
                "entity_id": "s1_002",
                "normalized_name": "tata consultancy services limited",
                "normalized_address": "tcs house fort mumbai 400001",
                "country": "India",
            },
            {
                "entity_id": "s1_003",
                "normalized_name": "singleton bakery limited",
                "normalized_address": "12 high road london",
                "country": "UK",
            },
        ])

        self.s2_df = pd.DataFrame([
            {
                "entity_id": "s2_001",
                "normalized_name": "walmart supercenter",
                "normalized_address": "702 sw 8th st bentonville",
                "country": "US",
                "source": "S2",
            },
            {
                "entity_id": "s2_002",
                "normalized_name": "tata consultancy services",
                "normalized_address": "raveline street mumbai 400001",
                "country": "India",
                "source": "S2",
            },
        ])

        self.s3_df = pd.DataFrame([
            {
                "entity_id": "s3_001",
                "normalized_name": "unrelated auto repair llc",
                "normalized_address": "500 industrial highway",
                "country": "US",
                "source": "S3",
            },
        ])

        self.gt_dict = {
            "s1_001": {"s2_001"},
            "s1_002": {"s2_002"},
            "s1_003": set(),  # singleton
        }

    def test_prepare_text_series(self):
        df = pd.DataFrame([
            {"normalized_name": "Test Name", "normalized_address": "123 Main St"},
            {"normalized_name": None, "normalized_address": "456 Oak Rd"},
            {"business_name": "Fallback Co", "business_address": "789 Pine Ave"},
        ])
        series = prepare_text_series(df)
        self.assertEqual(len(series), 3)
        self.assertEqual(series.iloc[0], "test name 123 main st")
        self.assertEqual(series.iloc[1], "456 oak rd")
        self.assertEqual(series.iloc[2], "fallback co 789 pine ave")

    def test_blocker_fit_and_query(self):
        s2_s3 = pd.concat([self.s2_df, self.s3_df], ignore_index=True)
        blocker = TfidfFaissBlocker(ngram_range=(3, 3), max_features=5000, k=2)
        blocker.fit_and_index(self.s1_df, s2_s3)

        self.assertGreater(blocker.dim, 0)
        self.assertEqual(blocker.faiss_index.ntotal, 3)

        candidates = blocker.retrieve_candidates(self.s1_df, k=2, include_similarity=True)
        self.assertIn("similarity_score", candidates.columns)
        for col in CANDIDATE_PAIR_COLUMNS:
            self.assertIn(col, candidates.columns)

        # Walmart S1 should retrieve Walmart S2 as its top candidate
        s1_001_cands = candidates[candidates["source1_entity_id"] == "s1_001"]
        self.assertGreater(len(s1_001_cands), 0)
        self.assertEqual(s1_001_cands.iloc[0]["candidate_entity_id"], "s2_001")

        # TCS S1 should retrieve TCS S2 as its top candidate
        s1_002_cands = candidates[candidates["source1_entity_id"] == "s1_002"]
        self.assertGreater(len(s1_002_cands), 0)
        self.assertEqual(s1_002_cands.iloc[0]["candidate_entity_id"], "s2_002")

    def test_functional_pipeline(self):
        cands_df = tfidf_faiss_blocking(
            s1_df=self.s1_df,
            s2_df=self.s2_df,
            s3_df=self.s3_df,
            k=2,
            min_similarity=0.0,
            max_features=5000,
        )

        for col in CANDIDATE_PAIR_COLUMNS:
            self.assertIn(col, cands_df.columns)

        cand_dict = to_candidate_dict(cands_df)
        self.assertIn("s2_001", cand_dict["s1_001"])
        self.assertIn("s2_002", cand_dict["s1_002"])

        recall, avg_cands, total_pairs = evaluate_blocking_recall(
            candidate_pairs_df=cands_df,
            gt_dict=self.gt_dict,
            total_s1_count=len(self.s1_df),
        )

        self.assertEqual(recall, 1.0)
        self.assertGreater(total_pairs, 0)


if __name__ == "__main__":
    unittest.main()
