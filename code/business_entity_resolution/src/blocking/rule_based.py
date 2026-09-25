# code/business_entity_resolution/src/blocking/rule_based.py
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union
import pandas as pd

try:
    from code.business_entity_resolution.src.schemas import (
        CANDIDATE_PAIR_COLUMNS,
        CandidatePair,
    )
except ImportError:
    from src.schemas import (  # type: ignore
        CANDIDATE_PAIR_COLUMNS,
        CandidatePair,
    )


def exact_name_country_blocking(
    s1_df: pd.DataFrame,
    s2_s3_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generates candidate pairs by matching normalized_name and country exactly.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Normalized Source 1 records conforming to NormalizedRecord schema.
        Must contain: 'entity_id', 'normalized_name', 'country'.
        (Falls back to 'clean_name' if 'normalized_name' is missing).
    s2_s3_df : pd.DataFrame
        Normalized Source 2 and Source 3 combined records.
        Must contain: 'entity_id', 'normalized_name', 'country', and optionally 'source'.

    Returns
    -------
    pd.DataFrame
        Candidate pairs DataFrame adhering strictly to CandidatePair schema:
        ['source1_entity_id', 'candidate_entity_id', 'source']
    """
    # 1. Standardize column names (support normalized_name or clean_name fallback)
    s1_copy = s1_df.copy()
    s2_s3_copy = s2_s3_df.copy()

    if "normalized_name" not in s1_copy.columns and "clean_name" in s1_copy.columns:
        s1_copy["normalized_name"] = s1_copy["clean_name"]
    if "normalized_name" not in s2_s3_copy.columns and "clean_name" in s2_s3_copy.columns:
        s2_s3_copy["normalized_name"] = s2_s3_copy["clean_name"]

    # 2. Infer source ('S2' or 'S3') if not explicitly provided
    if "source" not in s2_s3_copy.columns:

        def infer_source(eid: str) -> str:
            eid_str = str(eid).lower()
            if eid_str.startswith("s2"):
                return "S2"
            if eid_str.startswith("s3"):
                return "S3"
            return "unknown"

        s2_s3_copy["source"] = s2_s3_copy["entity_id"].apply(infer_source)

    # 3. Clean strings and filter out records with empty match keys
    s1_copy["norm_name_clean"] = s1_copy["normalized_name"].fillna("").astype(str).str.strip().str.lower()
    s1_copy["country_clean"] = s1_copy["country"].fillna("").astype(str).str.strip().str.upper()

    s2_s3_copy["norm_name_clean"] = s2_s3_copy["normalized_name"].fillna("").astype(str).str.strip().str.lower()
    s2_s3_copy["country_clean"] = s2_s3_copy["country"].fillna("").astype(str).str.strip().str.upper()

    # Filter out empty names or empty countries
    s1_valid = s1_copy[(s1_copy["norm_name_clean"] != "") & (s1_copy["country_clean"] != "")]
    cand_valid = s2_s3_copy[(s2_s3_copy["norm_name_clean"] != "") & (s2_s3_copy["country_clean"] != "")]

    # 4. Perform exact inner join on (normalized_name, country)
    s1_cols = ["entity_id", "norm_name_clean", "country_clean"]
    cand_cols = ["entity_id", "norm_name_clean", "country_clean", "source"]

    merged = pd.merge(
        s1_valid[s1_cols].rename(columns={"entity_id": "source1_entity_id"}),
        cand_valid[cand_cols].rename(columns={"entity_id": "candidate_entity_id"}),
        on=["norm_name_clean", "country_clean"],
        how="inner",
    )

    # 5. Project onto candidate pair schema columns
    result_df = merged[CANDIDATE_PAIR_COLUMNS].drop_duplicates().reset_index(drop=True)
    return result_df


def parse_ground_truth_dict(gt_path_or_df: Union[str, Path, pd.DataFrame]) -> Dict[str, Set[str]]:
    """Reads train_ground_truth.tsv into a mapping of s1_id -> set of matched entity IDs."""
    if isinstance(gt_path_or_df, (str, Path)):
        df = pd.read_csv(gt_path_or_df, sep="\t")
    else:
        df = gt_path_or_df

    gt_dict: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        if matched_str.strip() == "":
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = {m.strip() for m in matched_str.split(",") if m.strip()}
    return gt_dict


def evaluate_blocking_recall(
    candidate_pairs_df: pd.DataFrame,
    gt_dict: Dict[str, Set[str]],
    total_s1_count: Optional[int] = None,
) -> Tuple[float, float, int]:
    """Calculates candidate count and estimated blocking recall against ground truth.

    Returns
    -------
    (recall, avg_candidates_per_s1, total_candidates)
    """
    total_candidates = len(candidate_pairs_df)

    # Build S1 -> Set(candidate_ids) mapping
    cand_map: Dict[str, Set[str]] = {}
    for _, row in candidate_pairs_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        cand_id = str(row["candidate_entity_id"]).strip()
        cand_map.setdefault(s1_id, set()).add(cand_id)

    total_true_matches = 0
    retained_true_matches = 0

    for s1_id, true_matches in gt_dict.items():
        total_true_matches += len(true_matches)
        cands = cand_map.get(s1_id, set())
        retained_true_matches += len(true_matches & cands)

    num_s1 = total_s1_count if total_s1_count is not None else len(gt_dict)
    recall = (retained_true_matches / total_true_matches) if total_true_matches > 0 else 0.0
    avg_candidates = (total_candidates / num_s1) if num_s1 > 0 else 0.0

    print("=" * 65)
    print("Rule-Based Blocking Evaluation (Exact Name + Country)")
    print("=" * 65)
    print(f"Total Candidate Pairs Generated: {total_candidates:,}")
    print(f"Total Evaluated S1 Entities:     {num_s1:,}")
    print(f"Average Candidates per S1:       {avg_candidates:.2f}")
    print(f"Total True Matches in GT:        {total_true_matches:,}")
    print(f"Retained True Matches (Hits):    {retained_true_matches:,}")
    print(f"Estimated Blocking Recall:       {recall * 100:.2f}%")
    print("=" * 65)

    return recall, avg_candidates, total_candidates


# =============================================================================
# Execution / Benchmark Block
# =============================================================================
def _find_dataset_train_dir() -> Optional[Path]:
    """Searches for the dataset/train folder in common relative locations."""
    candidates = [
        Path("dataset/train"),
        Path("../dataset/train"),
        Path("../../dataset/train"),
        Path("../../../dataset/train"),
    ]
    for p in candidates:
        if p.exists() and (p / "train_source1.tsv").exists():
            return p.resolve()
    return None


def _create_benchmark_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    """Generates benchmark sample normalized records for verification when raw data is offline."""
    s1_data = [
        {
            "entity_id": "s1_001",
            "business_name": "Walmart Inc.",
            "business_address": "702 SW 8th St, Bentonville, AR 72716",
            "country": "US",
            "normalized_name": "walmart incorporated",
            "normalized_address": "702 southwest 8th street bentonville ar 72716",
        },
        {
            "entity_id": "s1_002",
            "business_name": "Tata Consultancy Services Ltd.",
            "business_address": "TCS House, Fort, Mumbai 400001",
            "country": "India",
            "normalized_name": "tata consultancy services limited",
            "normalized_address": "tcs house fort mumbai 400001",
        },
        {
            "entity_id": "s1_003",
            "business_name": "McDonald's Fast Food",
            "business_address": "110 N Carpenter St, Chicago, IL 60607",
            "country": "US",
            "normalized_name": "mcdonalds fast food",
            "normalized_address": "110 north carpenter street chicago il 60607",
        },
        {
            "entity_id": "s1_004",
            "business_name": "Singleton Bakery Ltd",
            "business_address": "12 High Rd",
            "country": "UK",
            "normalized_name": "singleton bakery limited",
            "normalized_address": "12 high road",
        },
    ]

    s2_s3_data = [
        {
            "entity_id": "s2_001",
            "business_name": "Walmart Inc.",
            "business_address": "702 SW 8th Street, 72716",
            "country": "US",
            "normalized_name": "walmart incorporated",
            "normalized_address": "702 southwest 8th street 72716",
            "source": "S2",
        },
        {
            "entity_id": "s2_002",
            "business_name": "Tata Consultancy Services Ltd",
            "business_address": "Mumbai 400001",
            "country": "India",
            "normalized_name": "tata consultancy services limited",
            "normalized_address": "mumbai 400001",
            "source": "S2",
        },
        {
            "entity_id": "s3_001",
            "business_name": "McDonald's Restaurant",  # Slightly different name -> missed by exact blocker
            "business_address": "110 Carpenter Ave",
            "country": "US",
            "normalized_name": "mcdonalds restaurant",
            "normalized_address": "110 carpenter avenue",
            "source": "S3",
        },
        {
            "entity_id": "s3_002",
            "business_name": "Unrelated Shop",
            "business_address": "50 Main Rd",
            "country": "India",
            "normalized_name": "unrelated shop",
            "normalized_address": "50 main road",
            "source": "S3",
        },
    ]

    gt = {
        "s1_001": {"s2_001"},
        "s1_002": {"s2_002"},
        "s1_003": {"s3_001"},
        "s1_004": set(),
    }

    return pd.DataFrame(s1_data), pd.DataFrame(s2_s3_data), gt


def main():
    train_dir = _find_dataset_train_dir()

    if train_dir is not None:
        print(f"Loading train set from: {train_dir}")
        from src.normalize import normalize_address, normalize_name

        s1_df = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", nrows=25000)
        s2_df = pd.read_csv(train_dir / "train_source2.tsv", sep="\t", nrows=25000)
        s3_df = pd.read_csv(train_dir / "train_source3.tsv", sep="\t", nrows=25000)

        s2_df["source"] = "S2"
        s3_df["source"] = "S3"
        s2_s3_df = pd.concat([s2_df, s3_df], ignore_index=True)

        # Apply normalization to match NormalizedRecord schema
        for df in [s1_df, s2_s3_df]:
            df["normalized_name"] = df["business_name"].apply(normalize_name)
            df["normalized_address"] = df["business_address"].apply(normalize_address)

        gt_dict = parse_ground_truth_dict(train_dir / "train_ground_truth.tsv")
        total_s1 = len(s1_df)
    else:
        print("Dataset directory not found on local disk. Running on benchmark sample...")
        s1_df, s2_s3_df, gt_dict = _create_benchmark_data()
        total_s1 = len(s1_df)

    # 1. Run Rule-Based Exact Blocker
    candidate_pairs = exact_name_country_blocking(s1_df, s2_s3_df)

    # 2. Evaluate Candidate Count & Blocking Recall
    evaluate_blocking_recall(candidate_pairs, gt_dict, total_s1_count=total_s1)


if __name__ == "__main__":
    main()
