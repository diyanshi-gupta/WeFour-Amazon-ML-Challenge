# Business Entity Resolution Challenge - Implementation Plan

## Overview
**Objective:** Map businesses from a reference source (Source 1) to noisy fragments in Source 2 and Source 3.
**Key Metric:** $F_{0.5}$ score (Weights Precision 2x over Recall - heavily penalizes false merges).
**Constraints:** Tab-separated files only, strictly NO external data/APIs, model must be MIT/Apache 2.0 licensed and under 8 Billion parameters.

---

## Phase 1: Exploration & Setup

### Step 1: EDA & Data Understanding
*   **Task:** Load all datasets using `sep='\t'`.
*   **Analysis:** 
    *   Analyze country distributions (Account for the 'France' anomaly in the test set).
    *   Analyze match cardinality from `train_ground_truth.tsv` (ratio of singletons vs. multiple matches).
    *   Assess missing values across names and addresses to inform the preprocessing strategy.

## Phase 2: Pipeline Development

### Step 2: Text Normalization (The Foundation)
*   **Task:** Build a reusable, consistent text cleaner for all sources.
*   **Operations:** Lowercase, strip punctuation, standardize spacing.
*   **Mapping:** Create custom mapping dictionaries strictly from training data to expand abbreviations (e.g., `rd` -> `road`) and harmonize legal suffixes (e.g., `pvt ltd` -> `private limited`).

### Step 3: Blocking / Candidate Generation (Maximizing Recall)
*   **Task:** Filter the $O(N \times M)$ search space down to a high-recall candidate pool.
*   **Strategy (Union of Approaches):**
    1.  TF-IDF character n-grams (e.g., 3-grams) + Cosine Similarity (FAISS) for top-K extraction.
    2.  Token blocking on extracted numerics (PIN/Zip codes) or rare tokens.
*   **Validation:** Measure *Blocking Recall* against `train_ground_truth.tsv` to ensure true matches aren't being dropped early.

### Step 4: Pairwise Feature Engineering (Creating the Dataset)
*   **Task:** For every candidate pair generated in Step 3, compute a rich set of features.
*   **Features:**
    *   *Name Similarities:* Jaro-Winkler, Levenshtein distance, Monge-Elkan.
    *   *Address Similarities:* Jaccard token overlap, TF-IDF Cosine distance.
    *   *Domain-Specific:* Exact country match flag, Numeric/PIN overlap score.

## Phase 3: Modeling & Refinement

### Step 5: Train & Validate the Matcher
*   **Task:** Train a binary classifier to predict Match (1) or No Match (0).
*   **Validation Split:** Stratify by country and match presence (singleton vs. multi-match).
*   **Algorithm:** LightGBM, XGBoost, or CatBoost (Highly robust on tabular similarity features).
*   **Threshold Tuning:** Iterate probability thresholds to find the exact cutoff that maximizes the **$F_{0.5}$ score**.

### Step 6: Error Analysis & Iteration
*   **Task:** Deep dive into validation errors to refine the pipeline.
*   **False Positives (Merged distinct businesses):** Highest penalty. Fix by raising probability threshold or adding stronger negative features.
*   **False Negatives (Missed true businesses):** Typically caused by blocking misses. Fix by relaxing blocking constraints in Step 3.

## Phase 4: Finalization

### Step 7: Generate Outputs & Validate Locally
*   **Task:** Run the finalized pipeline on `test_source1/2/3.tsv`.
*   **Outputs:** Produce `matching_results.tsv` and `candidate_pairs.tsv` in the `output/` directory.
*   **Validation:** Run the provided `utils/validate_submission.py` locally to guarantee strict format compliance.

### Step 8: Package the Final Submission
*   **Task:** Assemble the project zip file exactly as requested by the challenge.
*   **Structure:**
    *   `output/` (contains both TSV files).
    *   `code/business_entity_resolution/src/` (contains all runnable scripts).
    *   `code/business_entity_resolution/README.md` and `requirements.txt`.
    *   Completed `Documentation_template.md`.

---

## Git Version Control Strategy
To ensure top marks for version history, commits will be made incrementally, corresponding to the steps above. 
**Format:** Conventional Commits (e.g., `feat(eda): add initial data exploration script`).
**Protection:** A strict `.gitignore` will be used to ensure datasets and large model files are never pushed to the remote repository.
