"""
Stratified entity-level dataset splitter for Business Entity Resolution.
Splits Source1 entities by compound key (country, is_singleton) without row-level data leakage.
"""

from pathlib import Path
from typing import Set, Union
import pandas as pd
from sklearn.model_selection import train_test_split


def find_train_dir(provided_dir: Union[Path, str]) -> Path:
    """Finds valid train dataset directory containing train_source1.tsv and train_ground_truth.tsv."""
    candidates = [
        Path("C:/Users/yashi/Downloads/student_resource/dataset/train"),
        Path(provided_dir),
        Path("dataset/train"),
        Path("../dataset/train"),
    ]
    for c in candidates:
        if c.exists() and (c / "train_ground_truth.tsv").exists() and (c / "train_source1.tsv").exists():
            return c.resolve()
    return Path(provided_dir)


def create_stratified_split(
    train_dir: Union[Path, str] = "dataset/train",
    val_ratio: float = 0.20,
    seed: int = 42
) -> Set[str]:
    """
    Creates entity-level train/validation split stratified by (country, is_singleton).

    Args:
        train_dir: Directory containing train_source1.tsv and train_ground_truth.tsv.
        val_ratio: Fraction of Source1 entities to allocate to validation set (default: 0.20).
        seed: Random state seed for reproducibility (default: 42).

    Returns:
        Set[str]: Set of validation source1_entity_ids.
    """
    resolved_dir = find_train_dir(train_dir)
    gt_path = resolved_dir / "train_ground_truth.tsv"
    s1_path = resolved_dir / "train_source1.tsv"

    if not gt_path.exists() or not s1_path.exists():
        raise FileNotFoundError(
            f"Dataset files not found in {resolved_dir}. Expected train_ground_truth.tsv and train_source1.tsv."
        )

    # Load Ground Truth
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
    gt_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else "entity_id"
    gt_df = gt_df.rename(columns={gt_col: "source1_entity_id"})

    # Determine is_singleton (True if matched_entity_ids is empty/NaN)
    def check_singleton(val):
        if pd.isna(val) or val is None:
            return True
        return len(str(val).strip()) == 0

    gt_df["is_singleton"] = gt_df["matched_entity_ids"].apply(check_singleton)

    # Load Source1 Metadata for Country
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, usecols=lambda c: c in ["source1_entity_id", "entity_id", "country"])
    s1_id_col = "source1_entity_id" if "source1_entity_id" in s1_df.columns else "entity_id"
    s1_df = s1_df.rename(columns={s1_id_col: "source1_entity_id"})

    if "country" not in s1_df.columns:
        s1_df["country"] = "UNKNOWN"
    else:
        s1_df["country"] = s1_df["country"].fillna("UNKNOWN").astype(str).str.strip()

    # Merge entity metadata
    merged = pd.merge(
        gt_df[["source1_entity_id", "is_singleton"]],
        s1_df[["source1_entity_id", "country"]],
        on="source1_entity_id",
        how="inner"
    )

    # Compound Stratification Key: (country, is_singleton)
    merged["strat_key"] = merged["country"] + "_" + merged["is_singleton"].astype(str)

    # Handle rare strata (fewer than 2 instances) by mapping them to fallback
    counts = merged["strat_key"].value_counts()
    rare_keys = counts[counts < 2].index
    if len(rare_keys) > 0:
        merged.loc[merged["strat_key"].isin(rare_keys), "strat_key"] = "OTHER_STRATA"

    # Entity-level Stratified Split
    train_df, val_df = train_test_split(
        merged,
        test_size=val_ratio,
        stratify=merged["strat_key"],
        random_state=seed
    )

    val_ids = set(val_df["source1_entity_id"])

    # Output path setup
    out_dir = Path("dataset/train")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "val_source1_ids.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        for s1_id in sorted(val_ids):
            f.write(f"{s1_id}\n")

    print(f"[OK] Stratified split complete.")
    print(f"   Total Entities   : {len(merged):,}")
    print(f"   Train Entities   : {len(train_df):,} ({1 - val_ratio:.0%})")
    print(f"   Validation Entities: {len(val_df):,} ({val_ratio:.0%})")
    print(f"   Saved validation IDs to: {out_file.resolve()}")

    return val_ids


def main():
    val_ids = create_stratified_split(train_dir="dataset/train", val_ratio=0.20, seed=42)

    # Load validation split details for reporting statistics
    val_file = Path("dataset/train/val_source1_ids.txt")
    if val_file.exists():
        print(f"\nValidation Split Summary ({len(val_ids):,} entities):")
        print(f"Sample IDs: {list(val_ids)[:5]}")


if __name__ == "__main__":
    main()
