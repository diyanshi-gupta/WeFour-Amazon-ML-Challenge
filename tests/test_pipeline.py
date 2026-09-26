# tests/test_pipeline.py
import unittest
import pandas as pd
from src.features import compute_pair_features, build_feature_dataframe
from src.model import train_matcher_model, predict_match_probabilities
from src.tuning import evaluate_thresholds, generate_predictions_at_threshold, export_matching_results_tsv


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.s1_df = pd.DataFrame([
            {"entity_id": "s1_1", "clean_name": "target stores inc", "clean_address": "1000 nicollet mall 55403", "country": "US"},
            {"entity_id": "s1_2", "clean_name": "best buy company", "clean_address": "7601 penn ave s 55423", "country": "US"},
        ])
        self.s2_s3_df = pd.DataFrame([
            {"entity_id": "s2_1", "clean_name": "target stores", "clean_address": "1000 nicollet mall 55403", "country": "US"},
            {"entity_id": "s3_1", "clean_name": "unrelated shop", "clean_address": "random street", "country": "US"},
        ])
        self.candidates = {
            "s1_1": {"s2_1", "s3_1"},
            "s1_2": set(),
        }
        self.gt_dict = {
            "s1_1": {"s2_1"},
            "s1_2": set(),
        }

    def test_pairwise_features(self):
        feat = compute_pair_features(
            s1_name="target stores inc",
            s1_addr="1000 nicollet mall 55403",
            s1_country="US",
            cand_name="target stores",
            cand_addr="1000 nicollet mall 55403",
            cand_country="US",
        )
        self.assertGreater(feat["name_token_jaccard"], 0.5)
        self.assertEqual(feat["exact_address_match"], 1.0)
        self.assertEqual(feat["country_match"], 1.0)
        self.assertEqual(feat["has_shared_pin"], 1.0)

    def test_feature_dataframe_and_model(self):
        feat_df = build_feature_dataframe(self.candidates, self.s1_df, self.s2_s3_df, gt_dict=self.gt_dict)
        self.assertEqual(len(feat_df), 2)
        self.assertIn("label", feat_df.columns)

        model, cols = train_matcher_model(feat_df, use_lightgbm=False)
        pred_df = predict_match_probabilities(model, feat_df, cols)
        self.assertIn("prob_match", pred_df.columns)

        best_t, table = evaluate_thresholds(pred_df, self.gt_dict, ["s1_1", "s1_2"], thresholds=[0.5])
        self.assertIsNotNone(best_t)

        preds = generate_predictions_at_threshold(pred_df, best_t, ["s1_1", "s1_2"])
        self.assertIn("s1_1", preds)
        self.assertIn("s1_2", preds)


if __name__ == "__main__":
    unittest.main()
