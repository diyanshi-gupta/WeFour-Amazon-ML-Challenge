# Country Distribution Analysis

## TRAIN Split

### train_source1.tsv
```text
country
US       1323633
India     883188
```

### train_source2.tsv
```text
country
US       3016817
India    2017799
```

### train_source3.tsv
```text
country
US       3170056
India    2115547
```

## TEST Split

### test_source1.tsv
```text
country
India     809986
US        663106
France    259452
```

**Fraction of France in Test Source 1:** 14.98%

### test_source2.tsv
```text
country
India     2312565
US        1871330
France     703378
```

### test_source3.tsv
```text
country
India     2405000
US        1945701
France     731615
```

## Conclusion on Generalization

France is fully zero-shot for anything that learns from country-specific patterns. Because France only appears in the test set and not in the training set (which only covers US and India), any country-specific rules, text normalization logic, or models trained exclusively on training data will need to generalize to unseen French data without any explicit training examples.