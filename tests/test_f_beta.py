import pytest
from src.eval.f_beta import f_beta_score


def test_worked_example_problem_statement():
    """
    Worked example from problem statement:
    True match: 2 IDs ({'S2_10', 'S3_20'})
    Prediction: 3 IDs ({'S2_10', 'S3_20', 'S2_99'}) (2 correct, 1 FP)
    Precision = 2/3, Recall = 2/2 = 1.0
    F_0.5 = (1.25 * (2/3) * 1.0) / (0.25 * (2/3) + 1.0) = (5/6) / (7/6) = 5/7 ~ 0.7142857
    """
    y_true = {"S1_100": {"S2_10", "S3_20"}}
    y_pred = {"S1_100": {"S2_10", "S3_20", "S2_99"}}

    score = f_beta_score(y_true, y_pred, beta=0.5)
    assert pytest.approx(score, abs=1e-3) == 0.714
    assert pytest.approx(score, abs=1e-6) == 5.0 / 7.0


def test_singleton_perfect_prediction():
    """Correct singleton prediction (empty ground truth & empty prediction) yields 1.0."""
    y_true = {"S1_1": set()}
    y_pred = {"S1_1": set()}

    score = f_beta_score(y_true, y_pred, beta=0.5)
    assert score == 1.0


def test_singleton_false_merge():
    """False merge on a singleton (empty ground truth, non-empty prediction) yields 0.0."""
    y_true = {"S1_1": set()}
    y_pred = {"S1_1": {"S2_10"}}

    score = f_beta_score(y_true, y_pred, beta=0.5)
    assert score == 0.0


def test_missed_match():
    """Missed match (non-empty ground truth, empty prediction) yields 0.0."""
    y_true = {"S1_1": {"S2_10"}}
    y_pred = {"S1_1": set()}

    score = f_beta_score(y_true, y_pred, beta=0.5)
    assert score == 0.0


def test_zero_tp_overlap():
    """Prediction with zero true positive overlap yields 0.0."""
    y_true = {"S1_1": {"S2_10"}}
    y_pred = {"S1_1": {"S3_99"}}

    score = f_beta_score(y_true, y_pred, beta=0.5)
    assert score == 0.0


def test_macro_average_multiple_entities():
    """Macro-average across multiple entities."""
    y_true = {
        "S1_1": {"S2_10"},  # perfect match -> 1.0
        "S1_2": set(),       # correct singleton -> 1.0
        "S1_3": set(),       # false merge -> 0.0
        "S1_4": {"S2_20"}    # missed match -> 0.0
    }
    y_pred = {
        "S1_1": {"S2_10"},
        "S1_2": set(),
        "S1_3": {"S3_50"},
        "S1_4": set()
    }

    # Expected score = (1.0 + 1.0 + 0.0 + 0.0) / 4 = 0.5
    score = f_beta_score(y_true, y_pred, beta=0.5)
    assert score == 0.5
