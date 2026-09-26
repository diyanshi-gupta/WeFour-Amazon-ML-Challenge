# src/model.py
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

DEFAULT_FEATURE_COLS = [
    "exact_name_match",
    "name_token_jaccard",
    "name_3gram_similarity",
    "name_prefix_match",
    "name_len_diff_ratio",
    "exact_address_match",
    "address_token_jaccard",
    "address_missing",
    "country_match",
    "numeric_overlap_count",
    "has_shared_pin",
]


def train_matcher_model(
    train_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    use_lightgbm: bool = True,
):
    """Trains a binary classifier to predict P(match) for candidate pairs.

    Defaults to LightGBM with class balancing; gracefully falls back to
    RandomForestClassifier if LightGBM is unavailable.
    """
    if feature_cols is None:
        feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in train_df.columns]

    if "label" not in train_df.columns:
        raise ValueError("Training DataFrame must contain a 'label' column (1 for Match, 0 for Non-match).")

    X = train_df[feature_cols].fillna(0)
    y = train_df["label"].values

    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    print(f"Training dataset: {len(X)} candidate pairs ({pos_count} positive, {neg_count} negative)")

    # Adjust class balancing for severe candidate imbalance
    scale_pos = max(1.0, float(neg_count) / max(1, pos_count))

    model = None
    model_name = ""

    if use_lightgbm:
        try:
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                scale_pos_weight=scale_pos,
                random_state=42,
                verbosity=-1,
            )
            model.fit(X, y)
            model_name = "LightGBM Classifier"
        except Exception as e:
            print(f"LightGBM warning ({e}); falling back to RandomForest.")

    if model is None:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X, y)
        model_name = "RandomForest Classifier (Baseline)"

    print(f"Successfully trained {model_name}!")

    # Display Feature Importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print("\nTop Predictive Features:")
        for idx in sorted_idx[:6]:
            print(f"  * {feature_cols[idx]:25}: {importances[idx]}")

    return model, feature_cols


def predict_match_probabilities(
    model,
    feature_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Computes P(match) for all candidate pairs in feature_df."""
    if feature_cols is None:
        feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in feature_df.columns]

    X = feature_df[feature_cols].fillna(0)

    # If only 1 class was observed in tiny training samples, predict appropriately
    if hasattr(model, "classes_") and len(model.classes_) == 1:
        single_class = model.classes_[0]
        probs = np.full(len(X), 1.0 if single_class == 1 else 0.0)
    else:
        probs = model.predict_proba(X)[:, 1]

    result_df = feature_df.copy()
    result_df["prob_match"] = np.round(probs, 4)
    return result_df


def inspect_predictions(pred_df: pd.DataFrame, n: int = 5) -> None:
    """Verifies that high-probability pairs look like genuine matches

    and low-probability pairs look like distinct entities.
    """
    print("\n" + "=" * 70)
    print("Inspection of Model Predicted Probabilities P(match)")
    print("=" * 70)

    sorted_df = pred_df.sort_values(by="prob_match", ascending=False)

    print(f"\n[+] TOP {n} PREDICTED MATCHES (Highest P(match)):")
    for _, row in sorted_df.head(n).iterrows():
        s1 = row.get("s1_name", row["source1_entity_id"])
        cand = row.get("cand_name", row["candidate_entity_id"])
        prob = row["prob_match"]
        lbl = f" | True Label: {row['label']}" if "label" in row else ""
        print(f"  * P={prob:.4f}{lbl} | '{s1}' <--> '{cand}'")

    print(f"\n[-] LOWEST {n} PREDICTED MATCHES (Lowest P(match)):")
    for _, row in sorted_df.tail(n).iterrows():
        s1 = row.get("s1_name", row["source1_entity_id"])
        cand = row.get("cand_name", row["candidate_entity_id"])
        prob = row["prob_match"]
        lbl = f" | True Label: {row['label']}" if "label" in row else ""
        print(f"  * P={prob:.4f}{lbl} | '{s1}' <--> '{cand}'")

    print("=" * 70)
