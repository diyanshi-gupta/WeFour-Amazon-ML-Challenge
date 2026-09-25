#!/usr/bin/env python3
"""
Main pipeline execution script for WeFour - Amazon ML Challenge.
Runs end-to-end load -> normalize -> block -> validate offline -> generate test submission -> validate format.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Set, Dict, Optional
import pandas as pd

# Imports from src package
from src.normalize import normalize_name, normalize_address, apply_normalization
from src.blocking import generate_exact_name_candidates
from src.eval.f_beta import f_beta_score
from run_eval import evaluate_breakdown, print_summary_report

# CONFIGURATION FLAGS
# Set SAMPLE_SIZE to an integer (e.g., 50000) for fast local debugging, or None for full 2.2M evaluation
SAMPLE_SIZE: Optional[int] = 50000


def resolve_data_dir(relative_path: str, fallback_path: str, required_file: str) -> Path:
    """Finds directory containing required_file, prioritizing external dataset root."""
    candidates = [
        Path("C:/Users/yashi/Downloads/student_resource") / relative_path,
        Path(fallback_path),
        Path(relative_path),
    ]
    for c in candidates:
        if c.exists() and (c / required_file).exists():
            return c.resolve()
    return Path(relative_path).resolve()


def save_output_tsv(candidates_dict: Dict[str, Set[str]], s1_ids: Set[str], filepath: Path, col_name: str):
    """Saves candidate/matching dictionary to required TSV format."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id in sorted(s1_ids):
        matches = candidates_dict.get(s1_id, set())
        match_str = ",".join(sorted(matches)) if matches else ""
        rows.append({"source1_entity_id": s1_id, col_name: match_str})

    df = pd.DataFrame(rows)
    df.to_csv(filepath, sep="\t", index=False, encoding="utf-8")
    print(f"   Saved {len(df):,} rows to {filepath}")


def main():
    print("=" * 72)
    print("      WEFOUR AMAZON ML CHALLENGE - BASELINE END-TO-END PIPELINE")
    print("=" * 72)
    if SAMPLE_SIZE:
        print(f" [CONFIG] Running in SAMPLE MODE with SAMPLE_SIZE = {SAMPLE_SIZE:,} rows")
    else:
        print(" [CONFIG] Running in FULL DATASET MODE (all ~2.2M entities)")

    train_dir = resolve_data_dir("dataset/train", "c:/Users/yashi/Downloads/student_resource/dataset/train", "train_ground_truth.tsv")
    test_dir = resolve_data_dir("dataset/test", "c:/Users/yashi/Downloads/student_resource/dataset/test", "test_source1.tsv")

    print(f"\n[STAGE 1] Loading and Normalizing Training Data from: {train_dir}")
    s1_train_path = train_dir / "train_source1.tsv"
    s2_train_path = train_dir / "train_source2.tsv"
    s3_train_path = train_dir / "train_source3.tsv"
    gt_train_path = train_dir / "train_ground_truth.tsv"

    if not (s1_train_path.exists() and gt_train_path.exists()):
        print(f"ERROR: Training data files not found in {train_dir}")
        sys.exit(1)

    s1_train = pd.read_csv(s1_train_path, sep="\t", nrows=SAMPLE_SIZE, dtype=str)
    s2_train = pd.read_csv(s2_train_path, sep="\t", nrows=SAMPLE_SIZE * 2 if SAMPLE_SIZE else None, dtype=str)
    s3_train = pd.read_csv(s3_train_path, sep="\t", nrows=SAMPLE_SIZE * 2 if SAMPLE_SIZE else None, dtype=str)

    print("   Applying normalization to training data...")
    s1_train = apply_normalization(s1_train)
    s2_train = apply_normalization(s2_train)
    s3_train = apply_normalization(s3_train)

    print("\n[STAGE 2] Offline Candidate Blocking & Validation Evaluation")
    train_candidates = generate_exact_name_candidates(s1_train, s2_train, s3_train)

    val_ids_file = Path("dataset/train/val_source1_ids.txt")
    if not val_ids_file.exists():
        val_ids_file = train_dir / "val_source1_ids.txt"

    if val_ids_file.exists():
        with open(val_ids_file, "r", encoding="utf-8") as f:
            val_s1_set = set(line.strip() for line in f if line.strip())
        print(f"   Loaded {len(val_s1_set):,} validation entity IDs from {val_ids_file}")
    else:
        print("   Warning: val_source1_ids.txt not found. Using all loaded training IDs for validation evaluation.")
        s1_id_col = "source1_entity_id" if "source1_entity_id" in s1_train.columns else "entity_id"
        val_s1_set = set(s1_train[s1_id_col])

    # Filter evaluation to validation set IDs present in loaded dataset
    s1_id_col = "source1_entity_id" if "source1_entity_id" in s1_train.columns else "entity_id"
    available_val_ids = val_s1_set.intersection(set(s1_train[s1_id_col]))

    val_pred = {s1_id: train_candidates.get(s1_id, set()) for s1_id in available_val_ids}
    
    # Load Ground Truth for validation set
    gt_df = pd.read_csv(gt_train_path, sep="\t", dtype=str)
    gt_id_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else "entity_id"
    gt_df = gt_df[gt_df[gt_id_col].isin(available_val_ids)]

    val_gt = {}
    for _, row in gt_df.iterrows():
        s1_id = str(row[gt_id_col]).strip()
        m_str = str(row["matched_entity_ids"]).strip() if pd.notna(row["matched_entity_ids"]) else ""
        matches = set(m.strip() for m in m_str.split(",") if m.strip()) if m_str else set()
        val_gt[s1_id] = matches

    # Calculate validation metrics
    val_results = evaluate_breakdown(val_gt=val_gt, val_pred=val_pred, s1_df=s1_train)
    print_summary_report(val_results)

    print("\n[STAGE 3] Generating Predictions for Test Dataset from:", test_dir)
    s1_test_path = test_dir / "test_source1.tsv"
    s2_test_path = test_dir / "test_source2.tsv"
    s3_test_path = test_dir / "test_source3.tsv"

    if not s1_test_path.exists():
        print(f"ERROR: Test data file not found at {s1_test_path}")
        sys.exit(1)

    # Read ALL S1 IDs to guarantee 100% validator compliance (empty = no match)
    all_s1_test_df = pd.read_csv(s1_test_path, sep="\t", dtype=str, usecols=lambda c: c in ["source1_entity_id", "entity_id"])
    s1_test_id_col = "source1_entity_id" if "source1_entity_id" in all_s1_test_df.columns else "entity_id"
    all_test_s1_ids = set(all_s1_test_df[s1_test_id_col])
    print(f"   Loaded {len(all_test_s1_ids):,} total required test Source1 entity IDs.")

    s1_test = pd.read_csv(s1_test_path, sep="\t", nrows=SAMPLE_SIZE, dtype=str)
    s2_test = pd.read_csv(s2_test_path, sep="\t", nrows=SAMPLE_SIZE * 2 if SAMPLE_SIZE else None, dtype=str)
    s3_test = pd.read_csv(s3_test_path, sep="\t", nrows=SAMPLE_SIZE * 2 if SAMPLE_SIZE else None, dtype=str)

    print("   Applying normalization to test data...")
    s1_test = apply_normalization(s1_test)
    s2_test = apply_normalization(s2_test)
    s3_test = apply_normalization(s3_test)

    print("   Running exact candidate blocking on test set...")
    test_candidates = generate_exact_name_candidates(s1_test, s2_test, s3_test)

    out_matching = Path("output/matching_results.tsv")
    out_candidates = Path("output/candidate_pairs.tsv")

    # In crude baseline (Day 1), candidate blocking output IS the prediction matching output
    save_output_tsv(test_candidates, all_test_s1_ids, out_matching, "matched_entity_ids")
    save_output_tsv(test_candidates, all_test_s1_ids, out_candidates, "candidate_entity_ids")

    print("\n[STAGE 4] Executing Submission Validator")
    validator_script = Path("utils/validate_submission.py")
    if validator_script.exists():
        cmd = [
            sys.executable,
            str(validator_script),
            "--matching", str(out_matching),
            "--candidate", str(out_candidates),
            "--test-dir", str(test_dir),
        ]
        print(f"   Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        if res.returncode == 0:
            print("[OK] PIPELINE COMPLETED SUCCESSFULLY & VALIDATION PASSED!")
        else:
            print("[FAIL] PIPELINE COMPLETED WITH VALIDATION ERRORS.")
    else:
        print(f"   Warning: Validator script not found at {validator_script}")

    print("=" * 72)


if __name__ == "__main__":
    main()
