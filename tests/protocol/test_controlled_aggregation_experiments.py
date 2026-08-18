"""Regression guard for the stage-3 controlled aggregation experiments.

The script itself is the deliverable and runs with more replicates; this keeps the pre-registered
expectations from silently rotting when the estimator changes.
"""

import numpy as np

from scripts.controlled_aggregation_experiments import (
    NUM_CLASSES,
    build_tie_case,
    experiment_artificial_ties,
    run_experiments,
)


def test_every_pre_registered_expectation_holds():
    assert run_experiments(replicates=5)["failures"] == []


def test_the_tie_case_is_an_exact_two_two_split():
    """The experiment is worthless if the constructed ties are not actually tied."""
    _, annotations, tie_mask = build_tie_case(
        np.random.default_rng(0), num_anchor=100, num_tie=50, reliable=0.95, poor=0.45
    )
    tied = annotations[:, tie_mask]

    counts = np.stack([(tied == label).sum(axis=0) for label in range(NUM_CLASSES)])
    # Exactly two classes with two votes each, on every tie item.
    assert (np.sort(counts, axis=0)[-2:] == 2).all()


def test_the_negative_control_is_not_quietly_solvable():
    """Without anchors the two hypotheses are exact relabellings of each other.

    If this ever passes, the tie result above stops being evidence: it would mean the estimator can
    separate the pairs from the tie items alone, and the anchors were never what did the work.
    """
    blind = experiment_artificial_ties(replicates=5)["unidentifiable"]

    assert blind["ds_tie_accuracy"] < 0.6
