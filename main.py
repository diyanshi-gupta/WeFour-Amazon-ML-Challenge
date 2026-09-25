from pathlib import Path

from src.data import load_sample
from src.normalize import normalize_name, normalize_address
from src.evaluation import parse_ground_truth
from src.blocking import (
    generate_exact_name_candidates,
    calculate_blocking_metrics
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"


def main():

    # --------------------------------------------------
    # 1. Load sample data
    # --------------------------------------------------

    s1_df = load_sample(
        TRAIN_DIR / "train_source1.tsv"
    )

    s2_df = load_sample(
        TRAIN_DIR / "train_source2.tsv"
    )

    s3_df = load_sample(
        TRAIN_DIR / "train_source3.tsv"
    )

    print("Data loaded successfully.")
    print("S1 rows:", len(s1_df))
    print("S2 rows:", len(s2_df))
    print("S3 rows:", len(s3_df))


    # --------------------------------------------------
    # 2. Normalize names and addresses
    # --------------------------------------------------

    for df in [s1_df, s2_df, s3_df]:

        df["clean_name"] = (
            df["business_name"]
            .apply(normalize_name)
        )

        df["clean_address"] = (
            df["business_address"]
            .apply(normalize_address)
        )

    print("Normalization completed.")


    # --------------------------------------------------
    # 3. Load ground truth
    # --------------------------------------------------

    ground_truth = parse_ground_truth(
        TRAIN_DIR / "train_ground_truth.tsv"
    )

    print("Ground truth loaded.")


    # --------------------------------------------------
    # 4. Generate candidate pairs
    # --------------------------------------------------

    candidates = generate_exact_name_candidates(
        s1_df,
        s2_df,
        s3_df
    )

    print("Candidate generation completed.")


    # --------------------------------------------------
    # 5. Evaluate blocking
    # --------------------------------------------------

    blocking_recall, average_candidates = (
        calculate_blocking_metrics(
            candidates,
            ground_truth
        )
    )

    print()
    print("===== Blocking Results =====")
    print(
        "Blocking recall:",
        blocking_recall
    )

    print(
        "Average candidates per S1:",
        average_candidates
    )


if __name__ == "__main__":
    main()