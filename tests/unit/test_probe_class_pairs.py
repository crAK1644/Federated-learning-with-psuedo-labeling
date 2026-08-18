"""Checks for the stage-4 pair statistics that do not need the prepared dataset."""

import numpy as np

from scripts.probe_class_pairs import (
    CRITERIA,
    feature_degeneracy,
    pair_confusion_mass,
    verdict,
)


def test_confusion_mass_is_symmetric_and_normalised_by_support():
    # Class 0 has 100 samples and sends 20 to class 1; class 1 has 10 and sends 2 back.
    confusion = np.array([[80, 20, 0], [2, 8, 0], [0, 0, 50]])

    masses = pair_confusion_mass(confusion)

    assert masses[(0, 1)] == (20 + 2) / (100 + 10)
    assert masses[(0, 2)] == 0.0
    assert set(masses) == {(0, 1), (0, 2), (1, 2)}


def test_a_big_class_bleeding_into_a_small_one_does_not_outrank_a_worse_pair():
    """Raw off-diagonal counts would rank these the other way round.

    Classes 0/1 are large and swap 30 samples out of 2000; classes 2/3 are small and swap 20 out
    of 100. The second pair is far more confused per sample, and the ranking has to say so.
    """
    confusion = np.zeros((4, 4), dtype=int)
    confusion[0, 0], confusion[0, 1] = 970, 30
    confusion[1, 1] = 1000
    confusion[2, 2], confusion[2, 3] = 30, 20
    confusion[3, 3] = 50

    masses = pair_confusion_mass(confusion)

    assert masses[(2, 3)] > masses[(0, 1)]


def test_the_positive_control_criteria_reject_an_easy_pair():
    """A pair a linear model already solves leaves no room for aggregation to matter."""
    easy = {"labels": [1, 2], "head_to_head": {"linear": 0.999, "forest": 0.999}}
    coverage = {1: 27, 2: 27}
    support = {1: 900, 2: 900}

    checks = dict(verdict(easy, "positive", coverage, support))

    assert checks["non-linear separable"]
    assert not checks["linear struggles"]


def test_the_negative_control_criteria_reject_a_learnable_pair():
    learnable = {"labels": [4, 5], "head_to_head": {"forest": 0.99}}

    checks = dict(verdict(learnable, "negative", {}, {}))

    assert not checks["not learnable"]
    assert CRITERIA["negative_max_nonlinear_accuracy"] < 0.99


def test_degeneracy_counts_distinct_rows_not_samples():
    """A class stored as the same row over and over is destroyed, not hard."""
    features = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, 1, 1, 1])

    stats = feature_degeneracy(features, labels, num_classes=2)

    assert stats[0]["distinct_rows"] == 3
    assert stats[1]["distinct_rows"] == 1
    assert stats[1]["mean_feature_std"] == 0.0
