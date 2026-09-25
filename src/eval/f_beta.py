"""
F-beta evaluation metric for Business Entity Resolution.
Supports macro-averaged F_0.5 computation across Source1 entities including singleton handling.
"""

from typing import Dict, Set


def f_beta_score(
    y_true: Dict[str, Set[str]],
    y_pred: Dict[str, Set[str]],
    beta: float = 0.5
) -> float:
    """
    Computes macro-averaged F_beta score across all Source1 entities in y_true.

    Args:
        y_true: Mapping from Source1 entity_id to set of true matched entity_ids.
        y_pred: Mapping from Source1 entity_id to set of predicted matched entity_ids.
        beta: Weight of precision vs recall (default: 0.5, precision weighted 2x recall).

    Returns:
        float: Macro-averaged F_beta score in range [0.0, 1.0].
    """
    if not y_true:
        return 0.0

    beta_sq = beta ** 2
    factor = 1.0 + beta_sq
    scores = []

    for s1_id, true_matches in y_true.items():
        pred_matches = y_pred.get(s1_id, set())

        # Singleton & empty predictions edge cases
        if not true_matches and not pred_matches:
            # Correct singleton prediction (empty ground truth & empty prediction)
            scores.append(1.0)
            continue
        if not true_matches and pred_matches:
            # False merge on a singleton (empty ground truth, but predicted matches)
            scores.append(0.0)
            continue
        if true_matches and not pred_matches:
            # Missed match (non-empty ground truth, but empty prediction)
            scores.append(0.0)
            continue

        # Non-empty ground truth and prediction
        tp = len(pred_matches & true_matches)
        if tp == 0:
            scores.append(0.0)
            continue

        precision = tp / len(pred_matches)
        recall = tp / len(true_matches)

        denom = (beta_sq * precision) + recall
        if denom == 0:
            scores.append(0.0)
        else:
            f_score = (factor * precision * recall) / denom
            scores.append(f_score)

    return sum(scores) / len(scores)
