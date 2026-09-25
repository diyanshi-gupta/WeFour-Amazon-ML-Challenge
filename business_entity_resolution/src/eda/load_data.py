import pandas as pd
import sys
from pathlib import Path

def load_and_verify_data(dataset_path: Path):
    source_cols = ['entity_id', 'business_name', 'business_address', 'country']
    gt_cols = ['source1_entity_id', 'matched_entity_ids']
    
    for split in ['train', 'test']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"Skipping {split_dir} as it does not exist.")
            continue
            
        print(f"\n{'='*50}\nProcessing Split: {split.upper()}\n{'='*50}")
        
        for source_idx in [1, 2, 3]:
            filename = f"{split}_source{source_idx}.tsv"
            filepath = split_dir / filename
            if not filepath.exists():
                print(f"Warning: {filename} not found.")
                continue
                
            print(f"\n--- Loading {filename} ---")
            df = pd.read_csv(filepath, sep='\t')
            
            # 1. Verify Column Counts & Names
            assert list(df.columns) == source_cols, f"Expected columns {source_cols}, but got {list(df.columns)}"
            
            # 2. Print Row Counts
            print(f"Row count: {len(df)}")
            
            # 3. Add assertions that entity_id prefixes match expected source
            expected_prefix = f"S{source_idx}-"
            assert df['entity_id'].notna().all(), f"Found missing entity_ids in {filename}"
            invalid_prefixes = df[~df['entity_id'].str.startswith(expected_prefix)]
            assert invalid_prefixes.empty, f"Found {len(invalid_prefixes)} entity_ids without prefix '{expected_prefix}' in {filename}"
            
            # 4. Print sample rows
            print(f"Sample rows:")
            print(df.head(2).to_string())
            
        # Check ground truth
        gt_filename = f"{split}_ground_truth.tsv"
        gt_filepath = split_dir / gt_filename
        if gt_filepath.exists():
            print(f"\n--- Loading {gt_filename} ---")
            df_gt = pd.read_csv(gt_filepath, sep='\t')
            
            # Verify columns
            assert list(df_gt.columns) == gt_cols, f"Expected columns {gt_cols}, but got {list(df_gt.columns)}"
            
            print(f"Row count: {len(df_gt)}")
            print(f"Sample rows:")
            print(df_gt.head(2).to_string())

if __name__ == "__main__":
    # Point to the root directory where `dataset` resides
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    dataset_dir = base_dir / "dataset"
    if len(sys.argv) > 1:
        dataset_dir = Path(sys.argv[1])
        
    print(f"Using dataset directory: {dataset_dir}")
    load_and_verify_data(dataset_dir)
