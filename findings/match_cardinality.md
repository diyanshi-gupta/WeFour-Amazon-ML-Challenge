# Match Cardinality Analysis

## Distribution of Match-Set Size

| Match Size | Count | Fraction |
|---|---|---|
| 0 (Singletons) | 123247 | 5.58% |
| 1 | 119157 | 5.40% |
| 2 | 375212 | 17.00% |
| 3+ | 1589205 | 72.01% |

## Match Source Split

- **S2 Matches:** 3693619 (48.36%)
- **S3 Matches:** 3944746 (51.64%)

## Implications for Modeling

5.58% of entities are singletons (having no matches in S2 or S3). Correctly predicting an empty set for these singletons is worth protecting, given that the $F_{0.5}$ score heavily weights precision over recall. Generating false positives for these entities will aggressively penalize the metric, implying that the final modeling and thresholding steps should adopt a conservative approach, requiring very high confidence before linking records.
