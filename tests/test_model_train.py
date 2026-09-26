import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from model.train import (
    split_candidates_by_entity_id,
    train_lightgbm_matcher,
    load_val_source1_ids
)


def test_split_candidates_by_entity_id():
    df = pd.DataFrame({
        "source1_entity_id": ["S1_1", "S1_1", "S1_2", "S1_3", "S1_4", "S1_5"],
        "candidate_entity_id": ["S2_1", "S3_1", "S2_2", "S3_3", "S2_4", "S3_5"],
        "exact_name_match": [1, 0, 1, 0, 1, 0],
        "label": [1, 0, 1, 0, 0, 0]
    })

    val_ids = {"S1_2", "S1_5"}
    train_df, val_df = split_candidates_by_entity_id(df, val_ids, id_col="source1_entity_id")

    assert len(train_df) == 4
    assert len(val_df) == 2
    assert set(train_df["source1_entity_id"]).isdisjoint(val_ids)
    assert set(val_df["source1_entity_id"]).issubset(val_ids)


def test_train_lightgbm_matcher(tmp_path):
    # Setup dummy candidate data with imbalance
    np.random.seed(42)
    n_pos = 10
    n_neg = 90
    
    pos_data = pd.DataFrame({
        "source1_entity_id": [f"S1_pos_{i}" for i in range(n_pos)],
        "candidate_entity_id": [f"S2_pos_{i}" for i in range(n_pos)],
        "exact_name_match": [1] * n_pos,
        "name_token_jaccard": np.random.uniform(0.7, 1.0, n_pos),
        "label": [1] * n_pos
    })

    neg_data = pd.DataFrame({
        "source1_entity_id": [f"S1_neg_{i}" for i in range(n_neg)],
        "candidate_entity_id": [f"S2_neg_{i}" for i in range(n_neg)],
        "exact_name_match": [0] * n_neg,
        "name_token_jaccard": np.random.uniform(0.0, 0.3, n_neg),
        "label": [0] * n_neg
    })

    full_df = pd.concat([pos_data, neg_data], ignore_index=True)
    val_ids = {f"S1_pos_{i}" for i in range(2)} | {f"S1_neg_{i}" for i in range(18)}

    train_df, val_df = split_candidates_by_entity_id(full_df, val_ids, id_col="source1_entity_id")

    results = train_lightgbm_matcher(
        train_df=train_df,
        val_df=val_df,
        feature_cols=["exact_name_match", "name_token_jaccard"],
        target_col="label",
        model_output_dir=tmp_path,
        seed=42
    )

    model_path = Path(results["model_path"])
    imp_path = Path(results["imp_path"])

    assert model_path.exists()
    assert imp_path.exists()
    assert results["scale_pos_weight"] == pytest.approx(72 / 8, rel=1e-2)

    # Verify model artifact loading
    loaded_artifact = joblib.load(model_path)
    assert "model" in loaded_artifact
    assert "feature_cols" in loaded_artifact
    assert loaded_artifact["feature_cols"] == ["exact_name_match", "name_token_jaccard"]

    # Verify inference
    model = loaded_artifact["model"]
    probs = model.predict_proba(val_df[["exact_name_match", "name_token_jaccard"]])[:, 1]
    assert len(probs) == len(val_df)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
