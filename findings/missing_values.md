# Missing & Sparse Value Analysis

## train_source1.tsv

- **Total Rows:** 2206821
- **Missing/Empty `business_name`:** 0 (0.00%)
- **Missing/Empty `business_address`:** 0 (0.00%)
- **Sparse `business_address` (1-4 tokens):** 36128 (1.64%)

## train_source2.tsv

- **Total Rows:** 5034616
- **Missing/Empty `business_name`:** 2 (0.00%)
- **Missing/Empty `business_address`:** 168967 (3.36%)
- **Sparse `business_address` (1-4 tokens):** 244823 (4.86%)

## train_source3.tsv

- **Total Rows:** 5285603
- **Missing/Empty `business_name`:** 13 (0.00%)
- **Missing/Empty `business_address`:** 175916 (3.33%)
- **Sparse `business_address` (1-4 tokens):** 334723 (6.33%)

## Recommendations for Normalization

Based on the prevalence of missing and sparse fields, text normalization functions **must** safely handle empty or `NaN` values rather than crashing or guessing. 

1. **Never crash on NaN/Null:** Normalization logic should check for `pd.isna()` or empty strings and return an empty string (or a distinct missing-value indicator like `__MISSING__`) instead of crashing with a TypeError.
2. **Treat Missing as Distinct:** A missing address should be treated as its own categorical feature or flagged as a distinct category rather than silently imputing a generic value. This allows blocking and modeling steps to differentiate between 'we don't know the address' vs. 'the address is identical to another empty address'.
3. **Handle Sparse Addresses Differently:** Addresses with fewer than 5 tokens ('sparse addresses') do not contain enough signal for complex geographic matching. These should be classified into a distinct 'sparse address' category and handled with different blocking or feature extraction techniques than full addresses.
