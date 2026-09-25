import pandas as pd
from pathlib import Path
import sys

def analyze_match_cardinality(dataset_path: Path):
    findings_dir = Path(__file__).resolve().parent.parent.parent.parent / "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)
    
    gt_filepath = dataset_path / 'train' / 'train_ground_truth.tsv'
    if not gt_filepath.exists():
        print(f"Warning: Ground truth file not found at {gt_filepath}.")
        print("Creating an empty/default markdown report for now.")
        total_entities = 0
        df = pd.DataFrame({'source1_entity_id': [], 'matched_entity_ids': []})
    else:
        df = pd.read_csv(gt_filepath, sep='\t')
        total_entities = len(df)
        
    # Handle NaNs as empty strings
    df['matched_entity_ids'] = df['matched_entity_ids'].fillna('')
    
    # Parse matches into lists
    df['matches'] = df['matched_entity_ids'].apply(lambda x: [m.strip() for m in str(x).split(',')] if str(x).strip() else [])
    
    # Match set sizes
    df['match_count'] = df['matches'].apply(len)
    
    singletons_count = (df['match_count'] == 0).sum()
    fraction_singletons = singletons_count / total_entities if total_entities > 0 else 0
    
    dist_0 = singletons_count
    dist_1 = (df['match_count'] == 1).sum()
    dist_2 = (df['match_count'] == 2).sum()
    dist_3_plus = (df['match_count'] >= 3).sum()
    
    # S2 vs S3 splits
    s2_count = 0
    s3_count = 0
    for matches in df['matches']:
        for m in matches:
            if m.startswith('S2-'):
                s2_count += 1
            elif m.startswith('S3-'):
                s3_count += 1
                
    total_matches = s2_count + s3_count
    s2_fraction = s2_count / total_matches if total_matches > 0 else 0
    s3_fraction = s3_count / total_matches if total_matches > 0 else 0
    
    # Write Markdown
    out_lines = ["# Match Cardinality Analysis\n\n"]
    
    out_lines.append("## Distribution of Match-Set Size\n\n")
    out_lines.append("| Match Size | Count | Fraction |\n")
    out_lines.append("|---|---|---|\n")
    
    if total_entities > 0:
        out_lines.append(f"| 0 (Singletons) | {dist_0} | {dist_0/total_entities:.2%} |\n")
        out_lines.append(f"| 1 | {dist_1} | {dist_1/total_entities:.2%} |\n")
        out_lines.append(f"| 2 | {dist_2} | {dist_2/total_entities:.2%} |\n")
        out_lines.append(f"| 3+ | {dist_3_plus} | {dist_3_plus/total_entities:.2%} |\n\n")
    else:
        out_lines.append("| 0 (Singletons) | 0 | 0.00% |\n")
        out_lines.append("| 1 | 0 | 0.00% |\n")
        out_lines.append("| 2 | 0 | 0.00% |\n")
        out_lines.append("| 3+ | 0 | 0.00% |\n\n")
    
    out_lines.append("## Match Source Split\n\n")
    out_lines.append(f"- **S2 Matches:** {s2_count} ({s2_fraction:.2%})\n")
    out_lines.append(f"- **S3 Matches:** {s3_count} ({s3_fraction:.2%})\n\n")
    
    out_lines.append("## Implications for Modeling\n\n")
    implication_paragraph = (
        f"{fraction_singletons:.2%} of entities are singletons (having no matches in S2 or S3). "
        "Correctly predicting an empty set for these singletons is worth protecting, given that "
        "the $F_{0.5}$ score heavily weights precision over recall. Generating false positives for these "
        "entities will aggressively penalize the metric, implying that the final modeling and thresholding "
        "steps should adopt a conservative approach, requiring very high confidence before linking records."
    )
    out_lines.append(implication_paragraph + "\n")
    
    out_md = findings_dir / "match_cardinality.md"
    with open(out_md, 'w') as f:
        f.writelines(out_lines)
        
    print(f"Match cardinality analysis saved to {out_md}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    dataset_dir = base_dir / "dataset"
    if len(sys.argv) > 1:
        dataset_dir = Path(sys.argv[1])
        
    analyze_match_cardinality(dataset_dir)
