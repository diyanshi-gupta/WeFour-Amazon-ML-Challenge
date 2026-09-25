# Match Cardinality Analysis

## Distribution of Match-Set Size

| Match Size | Count | Fraction |
|---|---|---|
| 0 (Singletons) | 0 | 0.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3+ | 0 | 0.00% |

## Match Source Split

- **S2 Matches:** 0 (0.00%)
- **S3 Matches:** 0 (0.00%)

## Implications for Modeling

0.00% of entities are singletons (having no matches in S2 or S3). Correctly predicting an empty set for these singletons is worth protecting, given that the $F_{0.5}$ score heavily weights precision over recall. Generating false positives for these entities will aggressively penalize the metric, implying that the final modeling and thresholding steps should adopt a conservative approach, requiring very high confidence before linking records.
