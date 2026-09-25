# Missing & Sparse Value Analysis

_Data not available to run metrics. Below is the expected report format._

## train_source1.tsv

File train_source1.tsv not found.

## train_source2.tsv

File train_source2.tsv not found.

## train_source3.tsv

File train_source3.tsv not found.

## Recommendations for Normalization

Based on the prevalence of missing and sparse fields, text normalization functions **must** safely handle empty or `NaN` values rather than crashing or guessing. 

1. **Never crash on NaN/Null:** Normalization logic should check for `pd.isna()` or empty strings and return an empty string (or a distinct missing-value indicator like `__MISSING__`) instead of crashing with a TypeError.
2. **Treat Missing as Distinct:** A missing address should be treated as its own categorical feature or flagged as a distinct category rather than silently imputing a generic value. This allows blocking and modeling steps to differentiate between 'we don't know the address' vs. 'the address is identical to another empty address'.
3. **Handle Sparse Addresses Differently:** Addresses with fewer than 5 tokens ('sparse addresses') do not contain enough signal for complex geographic matching. These should be classified into a distinct 'sparse address' category and handled with different blocking or feature extraction techniques than full addresses.
