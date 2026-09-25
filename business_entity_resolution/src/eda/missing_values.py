import pandas as pd
from pathlib import Path
import sys

def analyze_missing_values(dataset_path: Path):
    findings_dir = Path(__file__).resolve().parent.parent.parent.parent / "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)
    
    out_lines = ["# Missing & Sparse Value Analysis\n\n"]
    
    train_dir = dataset_path / 'train'
    if not train_dir.exists():
        print(f"Warning: Train directory not found at {train_dir}")
        out_lines.append("_Data not available to run metrics. Below is the expected report format._\n\n")
    
    for i in [1, 2, 3]:
        file_name = f"train_source{i}.tsv"
        filepath = train_dir / file_name
        
        out_lines.append(f"## {file_name}\n\n")
        
        if not filepath.exists():
            out_lines.append(f"File {file_name} not found.\n\n")
            continue
            
        df = pd.read_csv(filepath, sep='\t')
        total = len(df)
        
        if total == 0:
            continue
            
        # Missing/Empty for business_name
        name_missing = df['business_name'].isna() | (df['business_name'].astype(str).str.strip() == '')
        name_missing_count = name_missing.sum()
        
        # Missing/Empty for business_address
        address_missing = df['business_address'].isna() | (df['business_address'].astype(str).str.strip() == '')
        address_missing_count = address_missing.sum()
        
        # Sparse Address (under 5 tokens, not empty)
        def is_sparse(addr):
            if pd.isna(addr):
                return False
            s = str(addr).strip()
            if not s:
                return False
            return len(s.split()) < 5
            
        sparse_count = df['business_address'].apply(is_sparse).sum()
        
        out_lines.append(f"- **Total Rows:** {total}\n")
        out_lines.append(f"- **Missing/Empty `business_name`:** {name_missing_count} ({name_missing_count/total:.2%})\n")
        out_lines.append(f"- **Missing/Empty `business_address`:** {address_missing_count} ({address_missing_count/total:.2%})\n")
        out_lines.append(f"- **Sparse `business_address` (1-4 tokens):** {sparse_count} ({sparse_count/total:.2%})\n\n")
        
    out_lines.append("## Recommendations for Normalization\n\n")
    out_lines.append(
        "Based on the prevalence of missing and sparse fields, text normalization functions **must** safely handle "
        "empty or `NaN` values rather than crashing or guessing. \n\n"
        "1. **Never crash on NaN/Null:** Normalization logic should check for `pd.isna()` or empty strings and return "
        "an empty string (or a distinct missing-value indicator like `__MISSING__`) instead of crashing with a TypeError.\n"
        "2. **Treat Missing as Distinct:** A missing address should be treated as its own categorical feature or flagged "
        "as a distinct category rather than silently imputing a generic value. This allows blocking and modeling steps "
        "to differentiate between 'we don't know the address' vs. 'the address is identical to another empty address'.\n"
        "3. **Handle Sparse Addresses Differently:** Addresses with fewer than 5 tokens ('sparse addresses') do not contain "
        "enough signal for complex geographic matching. These should be classified into a distinct 'sparse address' category "
        "and handled with different blocking or feature extraction techniques than full addresses.\n"
    )
    
    out_md = findings_dir / "missing_values.md"
    with open(out_md, 'w') as f:
        f.writelines(out_lines)
        
    print(f"Missing values analysis saved to {out_md}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    dataset_dir = base_dir / "dataset"
    if len(sys.argv) > 1:
        dataset_dir = Path(sys.argv[1])
        
    analyze_missing_values(dataset_dir)
