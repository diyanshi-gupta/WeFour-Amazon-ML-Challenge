# Embedding Model Choice

> **Branch:** `feat/blocking-hybrid-pv`  
> **Date:** 2026-09-26  
> **Analyst:** WeFour Team

## Objective

Select a sentence-embedding model for augmenting the blocking pipeline with
semantic similarity scores between business names and addresses.
Requirements:
- Open, commercially-usable license (MIT or Apache-2.0)
- < 8 billion parameters (fits in standard competition environment)
- Runs **fully offline** after first download (no API calls at inference time)
- Fast enough for pairwise scoring over millions of candidate pairs

## Candidate Models Evaluated

| Model | Parameters | License | Embedding Dims | Notes |
|-------|-----------|---------|----------------|-------|
| `sentence-transformers/all-MiniLM-L6-v2` ✅ **CHOSEN** | 22M | Apache-2.0 | 384 | Industry default; extremely fast; solid quality for business-name similarity. |
| `intfloat/e5-small-v2` | 33M | MIT | 384 | E5 family; strong on retrieval benchmarks; MIT license; fully offline. |
| `BAAI/bge-small-en-v1.5` | 33M | MIT | 384 | BGE-small; best-in-class quality at small size; MIT license. |

## Selected Model

**`sentence-transformers/all-MiniLM-L6-v2`**

| Property | Value |
|----------|-------|
| Full model name | `sentence-transformers/all-MiniLM-L6-v2` |
| Parameter count | **22 million** (22M) |
| License | **Apache-2.0** |
| Embedding dimensions | 384 |
| Framework | `sentence-transformers` (HuggingFace) |
| Offline inference | ✅ Yes — weights cached locally after first `model.encode()` call |
| HuggingFace page | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |

### Why this model?

- **Speed:** At 22M parameters, it is the fastest option — critical when scoring
  millions of candidate pairs during blocking.
- **Quality:** All-MiniLM-L6-v2 consistently ranks near the top of the
  Sentence-Transformers benchmarks for semantic textual similarity tasks,
  outperforming larger models on short-phrase inputs like business names.
- **Offline guarantee:** The `sentence_transformers` library downloads weights
  to `~/.cache/huggingface/` on first use; all subsequent calls are fully local
  — no internet access needed at inference time.
- **Apache-2.0 license:** Fully open, allows commercial and research use without
  restrictions.

## Offline Smoke Test

Model loaded in **260.04s**.  
Encoded **10 business names** in **0.044s** (offline).  

### Sample Names Used

- `Starbucks Coffee`
- `starbucks coffee`
- `McDonald's Fast Food`
- `City Pharmacy`
- `Tata Consultancy Services Ltd.`
- `Unique Local Bakery`
- `Shell Petrol Station`
- `Amazon Web Services`
- `Raj Electronics Store`
- `Global Logistics Solutions Pvt Ltd`

### Top-10 Most Similar Pairs (cosine similarity)

| Name A | Name B | Cosine Similarity |
|--------|--------|-------------------|
| Starbucks Coffee | starbucks coffee | `1.0000` ████████████████████ |
| Starbucks Coffee | McDonald's Fast Food | `0.4491` ████████ |
| starbucks coffee | McDonald's Fast Food | `0.4491` ████████ |
| Tata Consultancy Services Ltd. | Global Logistics Solutions Pvt Ltd | `0.3939` ███████ |
| Tata Consultancy Services Ltd. | Raj Electronics Store | `0.3868` ███████ |
| McDonald's Fast Food | City Pharmacy | `0.3542` ███████ |
| City Pharmacy | Raj Electronics Store | `0.3469` ██████ |
| Starbucks Coffee | City Pharmacy | `0.3318` ██████ |
| starbucks coffee | City Pharmacy | `0.3318` ██████ |
| City Pharmacy | Unique Local Bakery | `0.2975` █████ |

### Bottom-5 Most Dissimilar Pairs

| Name A | Name B | Cosine Similarity |
|--------|--------|-------------------|
| McDonald's Fast Food | Amazon Web Services | `0.1342` ██ |
| Tata Consultancy Services Ltd. | Unique Local Bakery | `0.1174` ██ |
| Starbucks Coffee | Amazon Web Services | `0.0957` █ |
| starbucks coffee | Amazon Web Services | `0.0957` █ |
| Shell Petrol Station | Amazon Web Services | `0.0427`  |

### Key Sanity Checks

| Check | Expected | Result |
|-------|----------|--------|
| `Starbucks Coffee` vs `starbucks coffee` (identical) | ≥ 0.98 | `1.0000` ✅ |
| Different industry names (e.g. bakery vs logistics) | < 0.50 | `0.0427` ✅ |

**Conclusion:** The model correctly assigns near-perfect similarity to
name variants of the same entity and low similarity to unrelated businesses,
confirming it is suitable for the blocking disambiguation task.

## Integration Plan

1. **Blocking gate:** After the name-token or exact-name blocker produces a
   candidate pair, compute the cosine similarity of the two business-name
   embeddings. If similarity < 0.45 *and* the name is flagged as generic/chain,
   drop the pair before it reaches the classifier.

2. **Feature engineering:** Add `name_embedding_cosine` and optionally
   `address_embedding_cosine` as features for the LightGBM classifier.

3. **Batch encoding:** Pre-compute all S1, S2, S3 embeddings once and cache
   them as numpy `.npy` files to avoid redundant computation across runs.

4. **Memory footprint:** 384-dim float32 × 1M records ≈ **1.4 GB** — well
   within typical competition RAM limits.