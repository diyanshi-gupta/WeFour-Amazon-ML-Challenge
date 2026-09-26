"""
Model training pipeline for Business Entity Resolution (LightGBM binary classifier).
Loads feature_matrix.parquet, splits candidate pairs by Source1 entity ID using val_source1_ids.txt,
trains LightGBM with class balancing (scale_pos_weight), saves feature importances, and outputs model artifact.
"""

import os
import sys
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb


DEFAULT_FEATURE_COLS = [
    "exact_name_match",
    "name_token_jaccard",
    "name_3gram_similarity",
    "name_prefix_match",
    "name_len_diff_ratio",
    "exact_address_match",
    "address_token_jaccard",
    "address_missing",
    "country_match",
    "numeric_overlap_count",
    "has_shared_pin",
]


def find_feature_matrix_path(provided_path: Optional[Union[Path, str]] = None) -> Optional[Path]:
    """Finds existing feature_matrix.parquet path across candidate locations."""
    if provided_path:
        p = Path(provided_path)
        if p.exists():
            return p.resolve()

    candidates = [
        Path("output/feature_matrix.parquet"),
        Path("dataset/feature_matrix.parquet"),
        Path("feature_matrix.parquet"),
        Path("data/feature_matrix.parquet"),
        Path("../output/feature_matrix.parquet"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def find_val_ids_path(provided_path: Optional[Union[Path, str]] = None) -> Optional[Path]:
    """Finds existing val_source1_ids.txt path across candidate locations."""
    if provided_path:
        p = Path(provided_path)
        if p.exists():
            return p.resolve()

    candidates = [
        Path("dataset/train/val_source1_ids.txt"),
        Path("C:/Users/yashi/Downloads/student_resource/dataset/train/val_source1_ids.txt"),
        Path("output/val_source1_ids.txt"),
        Path("val_source1_ids.txt"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def load_val_source1_ids(val_path: Optional[Union[Path, str]] = None) -> Set[str]:
    """Loads set of validation Source1 entity IDs from text file."""
    resolved_path = find_val_ids_path(val_path)
    if not resolved_path or not resolved_path.exists():
        raise FileNotFoundError(f"Validation split file not found. Expected val_source1_ids.txt.")

    with open(resolved_path, "r", encoding="utf-8") as f:
        val_ids = set(line.strip() for line in f if line.strip())

    print(f"[OK] Loaded {len(val_ids):,} validation entity IDs from {resolved_path}")
    return val_ids


def split_candidates_by_entity_id(
    df: pd.DataFrame,
    val_ids: Set[str],
    id_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits candidate pair DataFrame into train and validation sets by source1_entity_id.

    Args:
        df: Candidate pairs DataFrame (containing features, source1_entity_id, label).
        val_ids: Set of validation Source1 entity IDs.
        id_col: Column name for Source1 entity ID (inferred if None).

    Returns:
        Tuple[train_df, val_df]
    """
    if id_col is None:
        for col in ["source1_entity_id", "s1_id", "entity_id", "source1_id"]:
            if col in df.columns:
                id_col = col
                break

    if not id_col or id_col not in df.columns:
        raise KeyError(f"Could not find Source1 entity ID column in DataFrame columns: {df.columns.tolist()}")

    is_val = df[id_col].astype(str).isin(val_ids)
    val_df = df[is_val].copy()
    train_df = df[~is_val].copy()

    print(f"\n[SPLIT] Entity-level split completed by '{id_col}':")
    print(f"   Train candidates     : {len(train_df):,} rows ({len(train_df[id_col].unique()):,} unique Source1 IDs)")
    print(f"   Validation candidates: {len(val_df):,} rows ({len(val_df[id_col].unique()):,} unique Source1 IDs)")

    return train_df, val_df


def train_lightgbm_matcher(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "label",
    model_output_dir: Union[Path, str] = "output",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Trains a LightGBM binary classifier with class balancing (scale_pos_weight).

    Saves model artifact and feature importances to model_output_dir.
    """
    if feature_cols is None:
        # Exclude metadata/id/target columns
        meta_cols = {"source1_entity_id", "candidate_entity_id", "s1_id", "cand_id", "entity_id", "label", "is_match", "target", "s1_name", "cand_name"}
        feature_cols = [c for c in train_df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(train_df[c])]
        if not feature_cols:
            feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in train_df.columns]

    if target_col not in train_df.columns:
        for alt in ["is_match", "target", "matched"]:
            if alt in train_df.columns:
                target_col = alt
                break

    if target_col not in train_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in training DataFrame.")

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col].astype(int).values

    X_val = val_df[feature_cols].fillna(0) if len(val_df) > 0 else None
    y_val = val_df[target_col].astype(int).values if len(val_df) > 0 and target_col in val_df.columns else None

    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    scale_pos_weight = float(neg_count) / max(1, pos_count)

    print(f"\n[TRAIN] Candidate Class Balance:")
    print(f"   Positive candidate pairs (+1): {pos_count:,}")
    print(f"   Negative candidate pairs (0) : {neg_count:,}")
    print(f"   Computed scale_pos_weight     : {scale_pos_weight:.4f}")

    # Train LightGBM Classifier
    model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        verbosity=-1,
        n_jobs=-1
    )

    if X_val is not None and y_val is not None and len(X_val) > 0:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )
    else:
        model.fit(X_train, y_train)

    # Extract Feature Importances
    importances_split = model.booster_.feature_importance(importance_type="split")
    importances_gain = model.booster_.feature_importance(importance_type="gain")

    feat_imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": importances_gain,
        "importance_split": importances_split
    }).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)

    print("\n[IMPORTANCE] Top Feature Importances (by Gain):")
    for idx, row in feat_imp_df.head(10).iterrows():
        print(f"   {idx+1:2d}. {row['feature']:28s} | Gain: {row['importance_gain']:12.2f} | Split: {row['importance_split']:6d}")

    # Save artifacts
    out_dir = Path(model_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "lgb_matcher_model.joblib"
    imp_path = out_dir / "feature_importances.csv"

    artifact_dict = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "scale_pos_weight": scale_pos_weight,
        "pos_count": pos_count,
        "neg_count": neg_count
    }

    joblib.dump(artifact_dict, model_path)
    feat_imp_df.to_csv(imp_path, index=False, encoding="utf-8")

    print(f"\n[SAVED] Model artifact saved to : {model_path.resolve()}")
    print(f"[SAVED] Feature importances saved to: {imp_path.resolve()}")

    return {
        "model": model,
        "model_path": str(model_path.resolve()),
        "imp_path": str(imp_path.resolve()),
        "feature_cols": feature_cols,
        "scale_pos_weight": scale_pos_weight,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "feature_importances": feat_imp_df
    }


def main():
    print("=" * 72)
    print("           WEFOUR - TASK 2.1 LIGHTGBM MODEL TRAINING")
    print("=" * 72)

    # 1. Check feature_matrix.parquet existence
    matrix_path = find_feature_matrix_path()
    if matrix_path is None or not matrix_path.exists():
        print("\n[DEPENDENCY BLOCKER] feature_matrix.parquet does NOT exist.")
        print("  * Task 2.1 requires feature_matrix.parquet to train the LightGBM classifier.")
        print("  * Validation split file was successfully verified/generated at:")
        val_path = find_val_ids_path()
        if val_path:
            file_size = val_path.stat().st_size
            with open(val_path, "r", encoding="utf-8") as f:
                val_count = sum(1 for line in f if line.strip())
            print(f"    - Path: {val_path}")
            print(f"    - Size: {file_size:,} bytes")
            print(f"    - Entities: {val_count:,} Source1 IDs")
        print("\n  Stopping model training phase. Please run the feature generation pipeline first.")
        print("=" * 72)
        return

    # 2. Load feature_matrix.parquet
    print(f"\n[LOAD] Loading feature matrix from: {matrix_path}")
    df = pd.read_parquet(matrix_path)
    print(f"   Loaded {len(df):,} candidate rows with {len(df.columns)} columns.")

    # 3. Load validation split IDs
    val_ids = load_val_source1_ids()

    # 4. Split candidates by Source1 entity ID
    train_df, val_df = split_candidates_by_entity_id(df, val_ids)

    # 5. Train LightGBM model
    results = train_lightgbm_matcher(train_df, val_df)

    print("\n" + "=" * 72)
    print(" Task 2.1 Training Pipeline Completed Successfully!")
    print("=" * 72)


if __name__ == "__main__":
    main()
