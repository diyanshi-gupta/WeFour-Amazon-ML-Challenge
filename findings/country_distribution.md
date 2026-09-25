# Country Distribution Analysis

## Conclusion on Generalization

France is fully zero-shot for anything that learns from country-specific patterns. Because France only appears in the test set and not in the training set (which only covers US and India), any country-specific rules, text normalization logic, or models trained exclusively on training data will need to generalize to unseen French data without any explicit training examples.