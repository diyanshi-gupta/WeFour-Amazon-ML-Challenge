# tests/test_embedding_blocking.py
"""Unit tests for the dense sentence-embedding FAISS blocking channel."""

import unittest
import pandas as pd
from code.business_entity_resolution.src.blocking.embedding_channel import (
    EmbeddingFaissBlocker,
    embedding_faiss_blocking,
    run_low_tfidf_sanity_check,
    prepare_text_series,
    BENCHMARK_SEMANTIC_TRUE_PAIRS,
)
from code.business_entity_resolution.src.schemas import CANDIDATE_PAIR_COLUMNS


class TestEmbeddingFaissBlocking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize blocker once for unit tests
        cls.blocker = EmbeddingFaissBlocker(k=5)
        cls.blocker.load_model()

    def test_embedding_dimension_and_normalization(self):
        texts = ["Walmart Supercenter", "Tata Consultancy Services"]
        vecs = self.blocker.encode_records(texts, normalize_embeddings=True)
        self.assertEqual(vecs.shape[0], 2)
        self.assertEqual(vecs.shape[1], 384)
        # Check L2 norms are ~1.0
        import numpy as np
        norms = np.linalg.norm(vecs, axis=1)
        for n in norms:
            self.assertAlmostEqual(n, 1.0, places=4)

    def test_blocker_fit_and_query(self):
        s1_df = pd.DataFrame([
            {
                "entity_id": "s1_001",
                "normalized_name": "st judes childrens hospital",
                "normalized_address": "332 n lauderdale st memphis tn",
            },
            {
                "entity_id": "s1_002",
                "normalized_name": "precision car repair shop",
                "normalized_address": "450 w broadway eugene or",
            },
        ])

        s2_s3_df = pd.DataFrame([
            {
                "entity_id": "s2_001",
                "normalized_name": "saint jude pediatric medical center",
                "normalized_address": "332 north lauderdale street memphis tennessee",
                "source": "S2",
            },
            {
                "entity_id": "s3_001",
                "normalized_name": "precision auto care service garage",
                "normalized_address": "450 west broadway suite 2 eugene oregon",
                "source": "S3",
            },
            {
                "entity_id": "s2_002",
                "normalized_name": "unrelated retail bakery",
                "normalized_address": "12 high road london",
                "source": "S2",
            },
        ])

        self.blocker.build_candidate_index(s2_s3_df)
        self.assertEqual(self.blocker.faiss_index.ntotal, 3)

        candidates = self.blocker.retrieve_candidates(s1_df, k=2, include_similarity=True)

        for col in CANDIDATE_PAIR_COLUMNS:
            self.assertIn(col, candidates.columns)
        self.assertIn("similarity_score", candidates.columns)

        # S1_001 (St. Jude Children's Hospital) should match S2_001 (Saint Jude Pediatric Medical Center)
        s1_001_cands = candidates[candidates["source1_entity_id"] == "s1_001"]
        self.assertGreater(len(s1_001_cands), 0)
        self.assertEqual(s1_001_cands.iloc[0]["candidate_entity_id"], "s2_001")
        self.assertGreater(float(s1_001_cands.iloc[0]["similarity_score"]), 0.70)

        # S1_002 (Precision Car Repair) should match S3_001 (Precision Auto Care)
        s1_002_cands = candidates[candidates["source1_entity_id"] == "s1_002"]
        self.assertGreater(len(s1_002_cands), 0)
        self.assertEqual(s1_002_cands.iloc[0]["candidate_entity_id"], "s3_001")
        self.assertGreater(float(s1_002_cands.iloc[0]["similarity_score"]), 0.70)

    def test_low_tfidf_sanity_check_catches_all_five(self):
        """Sanity check: confirms embedding channel catches all 5 true-match pairs with low TF-IDF similarity."""
        results = run_low_tfidf_sanity_check(blocker=self.blocker, k=5)
        self.assertEqual(len(results), 5)

        for r in results:
            # Confirm embedding caught the true match in top-K
            self.assertTrue(r["caught_in_top_k"], f"Failed to catch pair {r['pair_id']}: {r['description']}")
            # Confirm dense embedding similarity is significantly higher than TF-IDF cosine
            self.assertGreater(
                r["embedding_cosine"],
                r["tfidf_cosine"],
                f"Embedding similarity ({r['embedding_cosine']:.3f}) should exceed TF-IDF ({r['tfidf_cosine']:.3f}) for {r['pair_id']}",
            )
            # Confirm high semantic similarity for the true match
            self.assertGreater(r["embedding_cosine"], 0.70)


if __name__ == "__main__":
    unittest.main()
