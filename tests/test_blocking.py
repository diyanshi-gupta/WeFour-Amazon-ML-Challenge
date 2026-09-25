# tests/test_blocking.py
import unittest
import pandas as pd
from src.blocking import (
    extract_informative_tokens,
    extract_address_numeric_tokens,
    generate_exact_name_candidates,
    generate_name_token_candidates,
    generate_numeric_address_candidates,
    union_candidate_dicts,
    calculate_blocking_metrics,
    run_blocking_v1_pipeline,
)


class TestBlockingV1(unittest.TestCase):
    def setUp(self):
        self.s1_df = pd.DataFrame([
            {"entity_id": "s1_1", "clean_name": "walmart inc", "clean_address": "702 road 72716", "country": "US"},
            {"entity_id": "s1_2", "clean_name": "tcs global", "clean_address": "mumbai 400001", "country": "India"},
            {"entity_id": "s1_3", "clean_name": "singleton bakery", "clean_address": "main street", "country": "US"},
        ])

        self.s2_df = pd.DataFrame([
            {"entity_id": "s2_1", "clean_name": "walmart inc", "clean_address": "702 road 72716", "country": "US"},
        ])

        self.s3_df = pd.DataFrame([
            {"entity_id": "s3_1", "clean_name": "tcs consultancy", "clean_address": "mumbai 400001", "country": "India"},
        ])

        self.gt_dict = {
            "s1_1": {"s2_1"},
            "s1_2": {"s3_1"},
            "s1_3": set(),
        }

    def test_token_extractions(self):
        tokens = extract_informative_tokens("walmart corporation retail")
        self.assertIn("walmart", tokens)
        self.assertIn("retail", tokens)
        self.assertNotIn("corporation", tokens)

        nums = extract_address_numeric_tokens("702 sw 8th st 72716")
        self.assertIn("702", nums)
        self.assertIn("72716", nums)

    def test_exact_name_blocking(self):
        cands = generate_exact_name_candidates(self.s1_df, self.s2_df, self.s3_df)
        self.assertIn("s2_1", cands["s1_1"])
        self.assertNotIn("s3_1", cands["s1_2"])  # Exact name differs

    def test_token_blocking(self):
        cands = generate_name_token_candidates(self.s1_df, self.s2_df, self.s3_df)
        self.assertIn("s3_1", cands["s1_2"])  # Shared informative token 'tcs'

    def test_numeric_address_blocking(self):
        cands = generate_numeric_address_candidates(self.s1_df, self.s2_df, self.s3_df)
        self.assertIn("s2_1", cands["s1_1"])

    def test_union_blocking(self):
        cands1 = {"s1_1": {"s2_1"}}
        cands2 = {"s1_2": {"s3_1"}}
        union = union_candidate_dicts(cands1, cands2)
        self.assertEqual(union["s1_1"], {"s2_1"})
        self.assertEqual(union["s1_2"], {"s3_1"})

    def test_pipeline_execution(self):
        full_union = run_blocking_v1_pipeline(self.s1_df, self.s2_df, self.s3_df, gt_dict=self.gt_dict)
        recall, avg_cands = calculate_blocking_metrics(self.gt_dict, full_union)
        self.assertEqual(recall, 1.0)


if __name__ == "__main__":
    unittest.main()
