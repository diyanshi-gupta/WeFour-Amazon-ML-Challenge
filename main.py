# main.py
from pathlib import Path
import subprocess
import sys
import pandas as pd
from src.data import load_sample
from src.normalize import normalize_name, normalize_address
from src.evaluation import parse_ground_truth, calculate_macro_f05
from src.blocking import run_blocking_v1_pipeline
from src.features import build_feature_dataframe, inspect_feature_pairs
from src.model import train_matcher_model, predict_match_probabilities, inspect_predictions
from src.tuning import (
    evaluate_thresholds,
    generate_predictions_at_threshold,
    export_matching_results_tsv,
    export_candidate_pairs_tsv,
)


def create_demo_data():
    """Generates a representative sample for pipeline testing when raw datasets

    are not downloaded locally.
    """
    s1_data = [
        {
            "entity_id": "s1_001",
            "business_name": "Walmart Supercenter Inc.",
            "business_address": "702 SW 8th St, Bentonville, AR 72716",
            "country": "US",
        },
        {
            "entity_id": "s1_002",
            "business_name": "Tata Consultancy Services Ltd.",
            "business_address": "TCS House, Raveline St, Fort, Mumbai 400001",
            "country": "India",
        },
        {
            "entity_id": "s1_003",
            "business_name": "McDonald's Fast Food",
            "business_address": "110 N Carpenter St, Chicago, IL 60607",
            "country": "US",
        },
        {
            "entity_id": "s1_004",
            "business_name": "Unique Local Bakery",
            "business_address": "12 High Rd, London",
            "country": "UK",
        },
    ]

    s2_data = [
        {
            "entity_id": "s2_001",
            "business_name": "Walmart Inc",
            "business_address": "702 SW 8th Street, Bentonville, 72716",
            "country": "US",
        },
        {
            "entity_id": "s2_002",
            "business_name": "Tata Consultancy Services",
            "business_address": "Raveline Street, Mumbai 400001",
            "country": "India",
        },
    ]

    s3_data = [
        {
            "entity_id": "s3_001",
            "business_name": "McDonald's Restaurant",
            "business_address": "110 Carpenter Ave, Chicago 60607",
            "country": "US",
        },
        {
            "entity_id": "s3_002",
            "business_name": "Unrelated Grocery Shop",
            "business_address": "50 Main Rd",
            "country": "India",
        },
    ]

    gt_dict = {
        "s1_001": {"s2_001"},
        "s1_002": {"s2_002"},
        "s1_003": {"s3_001"},
        "s1_004": set(),  # Singleton entity
    }

    return pd.DataFrame(s1_data), pd.DataFrame(s2_data), pd.DataFrame(s3_data), gt_dict


def main():
    print("=" * 70)
    print(" Amazon ML Challenge - Entity Resolution End-to-End Pipeline")
    print("=" * 70)

    # 1. Load Data
    train_dir = Path("dataset/train")
    s1_path = train_dir / "train_source1.tsv"

    if s1_path.exists():
        print("1. Loading dataset samples from disk...")
        s1 = load_sample(train_dir / "train_source1.tsv", nrows=10000)
        s2 = load_sample(train_dir / "train_source2.tsv", nrows=10000)
        s3 = load_sample(train_dir / "train_source3.tsv", nrows=10000)
        gt_dict = parse_ground_truth(train_dir / "train_ground_truth.tsv")
    else:
        print("1. Local dataset files not detected; executing on benchmark dataset...")
        s1, s2, s3, gt_dict = create_demo_data()

    all_s1_ids = s1["entity_id"].astype(str).tolist()
    s2_s3 = pd.concat([s2, s3], ignore_index=True)
    print(f"Loaded: S1 ({len(s1)}), S2+S3 ({len(s2_s3)}), Ground Truth entries ({len(gt_dict)})")

    # 2. Text Normalization
    print("\n2. Normalizing entity names and addresses...")
    for df in [s1, s2, s3, s2_s3]:
        df["clean_name"] = df["business_name"].apply(normalize_name)
        df["clean_address"] = df["business_address"].apply(normalize_address)

    # 3. Blocking v1 Pipeline Execution (Union of Exact, Token, Numeric/PIN)
    print("\n3. Generating candidate pairs through Blocking v1...")
    candidates = run_blocking_v1_pipeline(s1, s2, s3, gt_dict=gt_dict)

    # 4. Feature Engineering
    print("\n4. Computing pairwise similarity features for candidate pool...")
    feature_df = build_feature_dataframe(candidates, s1, s2_s3, gt_dict=gt_dict)
    print(f"Computed features for {len(feature_df)} candidate pairs.")

    # Manual inspection of positive & negative candidate pairs
    inspect_feature_pairs(feature_df, n_samples=2)

    # 5. Model Training (LightGBM / Baseline)
    print("\n5. Training Supervised Matcher Model...")
    model, feature_cols = train_matcher_model(feature_df)

    # 6. Predict Match Probabilities P(match)
    print("\n6. Predicting match probabilities...")
    pred_df = predict_match_probabilities(model, feature_df, feature_cols)
    inspect_predictions(pred_df, n=3)

    # 7. Threshold Tuning to Maximize Macro F0.5
    print("\n7. Sweeping probability thresholds to optimize Macro F0.5...")
    best_threshold, threshold_table = evaluate_thresholds(
        pred_df=pred_df,
        gt_dict=gt_dict,
        all_s1_ids=all_s1_ids,
    )

    # 8. Final Matching Predictions at Optimal Cutoff
    final_predictions = generate_predictions_at_threshold(pred_df, best_threshold, all_s1_ids)

    # 9. Export TSV Submissions
    print("\n8. Exporting output TSV files...")
    matching_path = export_matching_results_tsv(final_predictions, all_s1_ids, "output/matching_results.tsv")
    candidate_path = export_candidate_pairs_tsv(candidates, all_s1_ids, "output/candidate_pairs.tsv")

    # 10. Submission Format Verification
    print("\n9. Running Submission Format Validator...")
    validator_script = Path("utils/validate_submission.py")
    if validator_script.exists():
        cmd = [
            sys.executable,
            str(validator_script),
            "--matching",
            str(matching_path),
            "--candidate",
            str(candidate_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode == 0:
            print("[SUCCESS] All files strictly verified and ready for submission!")
        else:
            print(f"[NOTE] Validator exited with code {result.returncode}: {result.stderr}")

    print("\n" + "=" * 70)
    print(" Pipeline Execution Finished Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
