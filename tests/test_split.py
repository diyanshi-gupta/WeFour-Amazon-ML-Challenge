import pytest
from pathlib import Path
import pandas as pd
from src.eval.split import create_stratified_split


def test_create_stratified_split(tmp_path):
    # Setup dummy dataset in tmp_path
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    gt_df = pd.DataFrame({
        "source1_entity_id": [f"S1_{i}" for i in range(100)],
        "matched_entity_ids": ["S2_1" if i % 2 == 0 else "" for i in range(100)]
    })
    gt_df.to_csv(train_dir / "train_ground_truth.tsv", sep="\t", index=False)

    s1_df = pd.DataFrame({
        "entity_id": [f"S1_{i}" for i in range(100)],
        "country": ["US" if i < 60 else "India" for i in range(100)]
    })
    s1_df.to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)

    val_ids = create_stratified_split(train_dir=train_dir, val_ratio=0.20, seed=42)

    assert len(val_ids) == 20
    assert (Path("dataset/train/val_source1_ids.txt")).exists()

    with open("dataset/train/val_source1_ids.txt", "r", encoding="utf-8") as f:
        file_ids = set(line.strip() for line in f if line.strip())

    assert file_ids == val_ids
