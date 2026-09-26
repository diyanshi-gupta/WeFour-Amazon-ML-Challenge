# tests/test_schemas.py
import unittest
from code.business_entity_resolution.src.schemas import (
    NormalizedRecord,
    CandidatePair,
    FeatureRow,
    MatchingResultRow,
    CandidatePairsRow,
    NORMALIZED_RECORD_COLUMNS,
    FEATURE_COLUMNS,
    MATCHING_RESULTS_HEADER,
    CANDIDATE_PAIRS_HEADER,
)
from src.schemas import NormalizedRecord as SrcNormalizedRecord


class TestSchemas(unittest.TestCase):
    def test_normalized_record(self):
        rec = NormalizedRecord(
            entity_id="s1_1",
            business_name="Acme Corp",
            business_address="123 Main St",
            country="US",
            normalized_name="acme corporation",
            normalized_address="123 main street",
        )
        d = rec.to_dict()
        self.assertEqual(d["entity_id"], "s1_1")
        self.assertEqual(d["normalized_name"], "acme corporation")
        self.assertEqual(set(d.keys()), set(NORMALIZED_RECORD_COLUMNS))

    def test_candidate_pair(self):
        pair = CandidatePair(
            source1_entity_id="s1_1",
            candidate_entity_id="s2_2",
            source="S2",
        )
        d = pair.to_dict()
        self.assertEqual(d["source1_entity_id"], "s1_1")
        self.assertEqual(d["source"], "S2")

    def test_feature_row(self):
        feat = FeatureRow(
            source1_entity_id="s1_1",
            candidate_entity_id="s2_2",
            exact_name_match=1.0,
            country_match=1.0,
            has_shared_pin=1.0,
        )
        d = feat.to_dict()
        self.assertEqual(d["exact_name_match"], 1.0)
        self.assertEqual(d["name_token_jaccard"], 0.0)  # default placeholder float
        for col in FEATURE_COLUMNS:
            self.assertIn(col, d)

    def test_output_rows(self):
        m = MatchingResultRow(source1_entity_id="s1_1", matched_entity_ids="s2_2,s3_3")
        self.assertEqual(m.to_dict()["matched_entity_ids"], "s2_2,s3_3")
        self.assertEqual(MATCHING_RESULTS_HEADER, ["source1_entity_id", "matched_entity_ids"])

        c = CandidatePairsRow(source1_entity_id="s1_1", candidate_entity_ids="s2_2")
        self.assertEqual(c.to_dict()["candidate_entity_ids"], "s2_2")
        self.assertEqual(CANDIDATE_PAIRS_HEADER, ["source1_entity_id", "candidate_entity_ids"])

    def test_src_schemas_alias(self):
        self.assertIs(NormalizedRecord, SrcNormalizedRecord)


if __name__ == "__main__":
    unittest.main()
