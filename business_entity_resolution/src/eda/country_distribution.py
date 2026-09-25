import pandas as pd
import sys
from pathlib import Path

def analyze_country_distribution(dataset_path: Path):
    # Ensure findings directory exists
    findings_dir = Path(__file__).resolve().parent.parent.parent.parent / "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)
    
    out_lines = ["# Country Distribution Analysis\n\n"]
    
    for split in ['train', 'test']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"Skipping {split_dir} as it does not exist.")
            continue
            
        out_lines.append(f"## {split.upper()} Split\n\n")
        
        for source_idx in [1, 2, 3]:
            filename = f"{split}_source{source_idx}.tsv"
            filepath = split_dir / filename
            if not filepath.exists():
                print(f"Warning: {filename} not found.")
                continue
                
            df = pd.read_csv(filepath, sep='\t')
            country_counts = df['country'].value_counts(dropna=False)
            
            print(f"\n--- {filename} Country Counts ---")
            print(country_counts)
            
            out_lines.append(f"### {filename}\n")
            out_lines.append(f"```text\n{country_counts.to_string()}\n```\n\n")
            
            # Assert train covers only {US, India}
            if split == 'train':
                unique_countries = set(df['country'].dropna().unique())
                assert unique_countries.issubset({'US', 'India'}), f"Train set contains unexpected countries: {unique_countries}"
            
            # Specific checks for Test Source1
            if split == 'test' and source_idx == 1:
                # Confirm test contains France
                unique_test_countries = set(df['country'].dropna().unique())
                if 'France' in unique_test_countries:
                    print("\nConfirmed: Test set contains France.")
                
                # Report fraction of Test Source1 that is France
                if 'France' in df['country'].values:
                    fraction_france = (df['country'] == 'France').mean()
                    print(f"Fraction of Test Source 1 rows that are France: {fraction_france:.4f}")
                    out_lines.append(f"**Fraction of France in Test Source 1:** {fraction_france:.2%}\n\n")
                
    out_lines.append("## Conclusion on Generalization\n\n")
    out_lines.append(
        "France is fully zero-shot for anything that learns from country-specific patterns. "
        "Because France only appears in the test set and not in the training set (which only covers US and India), "
        "any country-specific rules, text normalization logic, or models trained exclusively on training data "
        "will need to generalize to unseen French data without any explicit training examples."
    )
    
    out_md = findings_dir / "country_distribution.md"
    with open(out_md, 'w') as f:
        f.writelines(out_lines)
    
    print(f"\nFindings saved to {out_md}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    dataset_dir = base_dir / "dataset"
    if len(sys.argv) > 1:
        dataset_dir = Path(sys.argv[1])
        
    print(f"Using dataset directory: {dataset_dir}")
    analyze_country_distribution(dataset_dir)
