# main.py
from pathlib import Path
import pandas as pd
from src.data import load_sample
from src.normalize import normalize_name, normalize_address
from src.evaluation import parse_ground_truth, calculate_macro_f05
from src.blocking import (
    generate_exact_name_candidates,
    calculate_blocking_metrics,
    run_blocking_v1_pipeline,
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
    print("=" * 60)
    print(" Amazon ML Challenge - Entity Resolution Pipeline")
    print("=" * 60)

    train_dir = Path("dataset/train")
    s1_path = train_dir / "train_source1.tsv"

    if s1_path.exists():
        print("1. Loading dataset samples from disk...")
        s1 = load_sample(train_dir / "train_source1.tsv", nrows=10000)
        s2 = load_sample(train_dir / "train_source2.tsv", nrows=10000)
        s3 = load_sample(train_dir / "train_source3.tsv", nrows=10000)
        gt_dict = parse_ground_truth(train_dir / "train_ground_truth.tsv")
    else:
        print("1. Local dataset not detected; running on representative test suite...")
        s1, s2, s3, gt_dict = create_demo_data()

    print(f"Loaded: S1 ({len(s1)}), S2 ({len(s2)}), S3 ({len(s3)}), Ground Truth entries ({len(gt_dict)})")

    # 2. Text Normalization
    print("\n2. Normalizing entity names and addresses...")
    for df in [s1, s2, s3]:
        df["clean_name"] = df["business_name"].apply(normalize_name)
        df["clean_address"] = df["business_address"].apply(normalize_address)

    # 3. Blocking v1 Pipeline Execution (Exact + Token + Numeric/Address + Union)
    print("\n3. Generating candidate pairs through Blocking v1...")
    candidates = run_blocking_v1_pipeline(s1, s2, s3, gt_dict=gt_dict)

    # 4. Evaluation via F0.5
    print("\n4. Preliminary F0.5 Scoring on candidate matches...")
    f05_score = calculate_macro_f05(gt_dict, candidates)
    print(f"Macro F0.5 Score of Candidate Pool: {f05_score:.4f}")

    print("\nPipeline execution completed successfully!")


if __name__ == "__main__":
    main()
