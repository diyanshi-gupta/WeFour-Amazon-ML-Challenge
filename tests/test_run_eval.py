import pytest
import pandas as pd
from run_eval import evaluate_breakdown, parse_matches_to_dict


def test_parse_matches_to_dict_and_breakdown():
    val_gt = pd.DataFrame({
        "source1_entity_id": ["S1_1", "S1_2", "S1_3", "S1_4"],
        "matched_entity_ids": ["S2_10,S3_20", "", "", "S2_30"]
    })
    
    val_pred = pd.DataFrame({
        "source1_entity_id": ["S1_1", "S1_2", "S1_3", "S1_4"],
        "matched_entity_ids": ["S2_10,S3_20,S2_99", "", "S3_50", ""]
    })

    s1_df = pd.DataFrame({
        "entity_id": ["S1_1", "S1_2", "S1_3", "S1_4"],
        "country": ["US", "US", "India", "India"]
    })

    results = evaluate_breakdown(val_gt=val_gt, val_pred=val_pred, s1_df=s1_df)

    assert results["total_entities"] == 4
    assert results["singleton_count"] == 2
    assert results["multimatch_count"] == 2

    # S1_1: F0.5 ~ 0.714
    # S1_2: F0.5 = 1.0 (singleton correct)
    # S1_3: F0.5 = 0.0 (singleton false merge)
    # S1_4: F0.5 = 0.0 (multimatch miss)
    # Overall = (5/7 + 1.0 + 0 + 0) / 4 = (12/7) / 4 = 3/7 ~ 0.42857
    assert pytest.approx(results["overall_f05"], abs=1e-3) == 0.4286

    assert "US" in results["by_country"]
    assert "India" in results["by_country"]
    assert results["by_country"]["US"]["count"] == 2
    assert results["by_country"]["India"]["count"] == 2
