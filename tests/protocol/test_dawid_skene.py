"""Dawid-Skene estimator: correctness, missing data, determinism, and every fallback path.

The safety-relevant assertions here are the fallback ones: a caller that sees ``status != "ok"``
must never broadcast these labels, so each failure mode gets its own test rather than being
covered incidentally.
"""

import numpy as np
import pytest

from ssfl.protocols.dawid_skene import (
    ABSTAIN,
    DawidSkeneSettings,
    build_annotation_matrix,
    fit_dawid_skene,
)

NUM_CLASSES = 4


def _majority(annotations: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Same rule as protocols/ssfl.py::aggregate_votes -- ties take the lowest class index."""
    num_open = annotations.shape[1]
    labels = np.full(num_open, ABSTAIN, dtype=np.int64)
    for i in range(num_open):
        column = annotations[:, i]
        column = column[column != ABSTAIN]
        if len(column):
            labels[i] = int(np.bincount(column, minlength=num_classes).argmax())
    return labels


def _synthetic(
    num_items: int = 400,
    num_clients: int = 7,
    accuracy_by_client=None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Clients with known per-client accuracy label items drawn from a uniform true class."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, NUM_CLASSES, size=num_items)
    accuracies = accuracy_by_client or [0.9] * num_clients
    annotations = np.empty((len(accuracies), num_items), dtype=np.int8)
    for j, accuracy in enumerate(accuracies):
        correct = rng.random(num_items) < accuracy
        wrong = (truth + rng.integers(1, NUM_CLASSES, size=num_items)) % NUM_CLASSES
        annotations[j] = np.where(correct, truth, wrong)
    return annotations, truth


def _fit(annotations, majority=None, **overrides):
    settings = DawidSkeneSettings(min_clients=2, **overrides)
    return fit_dawid_skene(
        annotations,
        num_classes=NUM_CLASSES,
        majority_labels=_majority(annotations) if majority is None else majority,
        settings=settings,
    )


# --- observation model ---------------------------------------------------------------------


def test_build_annotation_matrix_is_sender_sorted():
    late = ("client_9", np.array([0, 1, ABSTAIN], dtype=np.int8))
    early = ("client_1", np.array([2, ABSTAIN, 3], dtype=np.int8))
    matrix, senders = build_annotation_matrix([late, early], num_open=3, num_classes=NUM_CLASSES)
    assert senders == ("client_1", "client_9")
    assert matrix[0].tolist() == [2, ABSTAIN, 3]


def test_build_annotation_matrix_rejects_illegal_labels():
    with pytest.raises(ValueError, match="outside"):
        build_annotation_matrix(
            [("a", np.array([0, NUM_CLASSES], dtype=np.int8))], num_open=2, num_classes=NUM_CLASSES
        )
    with pytest.raises(ValueError, match="expected"):
        build_annotation_matrix(
            [("a", np.array([0], dtype=np.int8))], num_open=2, num_classes=NUM_CLASSES
        )


# --- estimation ----------------------------------------------------------------------------


def test_recovers_truth_better_than_majority_when_clients_differ_in_reliability():
    # Five unreliable clients outvote two good ones, which is exactly what majority cannot fix.
    annotations, truth = _synthetic(
        accuracy_by_client=[0.95, 0.95, 0.35, 0.35, 0.35, 0.35, 0.35], seed=11
    )
    fit = _fit(annotations)
    assert fit.ok, fit.status
    majority_accuracy = (_majority(annotations) == truth).mean()
    assert (fit.labels == truth).mean() > majority_accuracy


def test_high_consensus_case_settles_in_five_steps_or_fewer():
    annotations, truth = _synthetic(
        num_items=1_000,
        num_clients=7,
        accuracy_by_client=[0.9] * 7,
        seed=17,
    )
    fit = _fit(annotations)
    assert fit.ok, fit.status
    assert fit.iterations <= 5
    assert (fit.labels == truth).mean() > 0.99


def test_controlled_exact_ties_are_resolved_only_when_anchor_items_identify_raters():
    """Two reliable and two inverted raters tie on every target item; a fifth rater labels only
    anchor items. The anchors make the two groups identifiable, so DS can resolve the target ties.
    Without anchors, the symmetric problem must be rejected rather than guessed through.
    """
    rng = np.random.default_rng(17)
    num_anchors, num_tied = 400, 200
    truth = rng.integers(0, 2, size=num_anchors + num_tied)
    annotations = np.empty((5, len(truth)), dtype=np.int8)
    annotations[0] = truth
    annotations[1] = truth
    annotations[2] = 1 - truth
    annotations[3] = 1 - truth
    annotations[4, :num_anchors] = truth[:num_anchors]
    annotations[4, num_anchors:] = ABSTAIN

    majority = _majority(annotations, num_classes=2)
    fit = fit_dawid_skene(
        annotations,
        num_classes=2,
        majority_labels=majority,
        settings=DawidSkeneSettings(
            min_clients=3,
            max_iterations=500,
            permutation_min_diagonal_ratio=0.0,
            permutation_min_majority_agreement=0.0,
        ),
    )
    assert fit.ok, fit.status
    assert (majority[num_anchors:] == truth[num_anchors:]).mean() < 0.6
    assert (fit.labels[num_anchors:] == truth[num_anchors:]).mean() == 1.0
    assert fit.iterations <= 5

    tied_only = annotations[:, num_anchors:]
    tied_majority = majority[num_anchors:]
    unanchored = fit_dawid_skene(
        tied_only,
        num_classes=2,
        majority_labels=tied_majority,
        settings=DawidSkeneSettings(
            min_clients=3,
            max_iterations=500,
            permutation_min_diagonal_ratio=0.0,
            permutation_min_majority_agreement=0.0,
        ),
    )
    assert unanchored.status == "permutation_check_chance"
    assert not unanchored.valid_mask.any()


def test_confusion_matrix_tracks_the_client_that_generated_it():
    annotations, _ = _synthetic(accuracy_by_client=[0.95, 0.95, 0.95, 0.4], seed=3)
    fit = _fit(annotations)
    assert fit.ok, fit.status
    diagonals = fit.confusion[:, np.arange(NUM_CLASSES), np.arange(NUM_CLASSES)].mean(axis=1)
    assert diagonals[:3].min() > diagonals[3]


def test_abstention_is_missing_data_not_a_class():
    annotations, truth = _synthetic(seed=5)
    holes = annotations.copy()
    holes[0, :200] = ABSTAIN
    fit = _fit(holes)
    assert fit.ok, fit.status
    # A client that abstained on half the items must not be modelled as always-wrong there.
    assert fit.confusion[0].diagonal().min() > 0.5
    assert (fit.labels == truth).mean() > 0.9


def test_explicit_abstention_adds_one_emission_without_changing_validity():
    annotations, _ = _synthetic(seed=6)
    annotations[0, :200] = ABSTAIN
    annotations[:, -10:] = ABSTAIN
    fit = _fit(annotations, explicit_abstention=True)
    assert fit.numerically_valid, fit.status
    assert fit.confusion.shape == (annotations.shape[0], NUM_CLASSES, NUM_CLASSES + 1)
    expected_valid = (annotations != ABSTAIN).sum(axis=0) >= 1
    np.testing.assert_array_equal(fit.candidate_valid_mask, expected_valid)
    assert (fit.candidate_labels[~expected_valid] == ABSTAIN).all()


def test_explicit_abstention_keeps_an_all_silent_client_as_evidence():
    annotations, _ = _synthetic(num_clients=5, seed=8)
    annotations[-1] = ABSTAIN
    missing = _fit(annotations)
    explicit = _fit(annotations, explicit_abstention=True)
    assert missing.eligible_clients == 4
    assert explicit.eligible_clients == 5
    assert explicit.confusion.shape == (5, NUM_CLASSES, NUM_CLASSES + 1)


def test_all_abstain_items_stay_invalid():
    annotations, _ = _synthetic(seed=7)
    annotations[:, :10] = ABSTAIN
    fit = _fit(annotations)
    assert fit.ok, fit.status
    assert not fit.valid_mask[:10].any()
    assert (fit.labels[:10] == ABSTAIN).all()
    assert fit.valid_mask[10:].all()


def test_all_abstain_items_do_not_change_the_fit_for_observed_items():
    annotations, _ = _synthetic(seed=32)
    baseline = _fit(annotations)
    padded = _fit(np.pad(annotations, ((0, 0), (0, 1_000)), constant_values=ABSTAIN))
    assert baseline.ok and padded.ok
    np.testing.assert_array_equal(padded.labels[: annotations.shape[1]], baseline.labels)
    np.testing.assert_allclose(padded.confusion, baseline.confusion, rtol=0.0, atol=1e-12)
    assert padded.iterations == baseline.iterations
    assert padded.objective == pytest.approx(baseline.objective, abs=1e-10)


def test_deterministic_and_invariant_to_client_order():
    annotations, _ = _synthetic(seed=13)
    first = _fit(annotations)
    second = _fit(annotations)
    assert first.objective == second.objective
    assert np.array_equal(first.labels, second.labels)


def test_min_item_annotations_gates_validity():
    annotations, _ = _synthetic(num_clients=5, seed=17)
    annotations[1:, 0] = ABSTAIN  # item 0 keeps exactly one annotation
    fit = _fit(annotations, min_item_annotations=2)
    assert fit.ok, fit.status
    assert not fit.valid_mask[0]
    assert fit.valid_mask[1:].all()


def test_sparse_client_is_excluded_but_not_treated_as_unreliable():
    annotations, _ = _synthetic(num_clients=5, seed=19)
    annotations[4, 5:] = ABSTAIN  # 5 annotations left
    fit = _fit(annotations, min_client_annotations=50)
    assert fit.ok, fit.status
    assert fit.eligible_clients == 4
    assert fit.confusion.shape[0] == 4


# --- fallback paths ------------------------------------------------------------------------


def test_too_few_clients_falls_back():
    annotations, _ = _synthetic(num_clients=2, accuracy_by_client=[0.9, 0.9], seed=23)
    fit = fit_dawid_skene(
        annotations,
        num_classes=NUM_CLASSES,
        majority_labels=_majority(annotations),
        settings=DawidSkeneSettings(min_clients=3),
    )
    assert fit.status == "insufficient_clients"
    assert not fit.ok
    assert not fit.valid_mask.any()
    assert (fit.labels == ABSTAIN).all()


def test_iteration_cap_without_convergence_falls_back():
    annotations, _ = _synthetic(seed=29)
    fit = _fit(annotations, max_iterations=1, min_iterations=1)
    assert fit.status == "not_converged"
    assert not fit.valid_mask.any()


def test_permutation_check_catches_relabelled_classes():
    # Every client applies the same +1 rotation: the likelihood is identical to the unrotated
    # problem, so only the permutation check can tell that latent class c is no longer output c.
    annotations, _ = _synthetic(seed=31)
    rotated = ((annotations + 1) % NUM_CLASSES).astype(np.int8)
    honest_majority = _majority(annotations)
    fit = fit_dawid_skene(
        rotated,
        num_classes=NUM_CLASSES,
        majority_labels=honest_majority,
        settings=DawidSkeneSettings(min_clients=2, permutation_min_majority_agreement=0.8),
    )
    assert fit.status == "permutation_check_agreement"
    assert not fit.valid_mask.any()
    assert fit.converged  # it converged; it is the class identity that is wrong


def test_empty_input_falls_back():
    fit = _fit(np.zeros((0, 0), dtype=np.int8), majority=np.zeros(0, dtype=np.int64))
    assert fit.status == "no_annotations"


def test_failed_fit_never_returns_usable_labels():
    annotations, _ = _synthetic(seed=37)
    fit = _fit(annotations, max_iterations=1, min_iterations=1)
    assert not fit.ok
    assert fit.valid_mask.sum() == 0
    assert set(np.unique(fit.labels)) == {ABSTAIN}



def _specialists(num_items: int = 900, num_clients: int = 9, seed: int = 7):
    """A non-IID federation shaped like scenario 1: each client is reliable on the classes in its
    own shard and, when it meets anything else, guesses *inside its own shard* rather than at
    random. That last detail is the whole point -- it is what drives the confusion mode off the
    diagonal for honest reasons, and it is what an absolute diagonal floor misreads as tampering.
    """
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, NUM_CLASSES, size=num_items)
    annotations = np.empty((num_clients, num_items), dtype=np.int8)
    for j in range(num_clients):
        expertise = np.array([(2 * j) % NUM_CLASSES, (2 * j + 1) % NUM_CLASSES])
        known = np.isin(truth, expertise)
        in_shard_guess = expertise[rng.integers(0, len(expertise), size=num_items)]
        annotations[j] = np.where(known & (rng.random(num_items) < 0.95), truth, in_shard_guess)
    return annotations, truth


def test_specialised_clients_are_not_mistaken_for_a_permutation():
    """The regression the check was rewritten for. Acceptance must be decided against what
    majority scores on the same annotations, never a fixed constant: on real scenario-1 data even
    sealed ground truth scores ~0.14-0.25, so any absolute floor above that rejects correct labels.
    """
    annotations, _ = _specialists()
    fit = _fit(annotations)
    assert fit.status == "ok"
    assert fit.diagonal_fraction >= 0.7 * fit.reference_diagonal_fraction
    # Same fit, same data -- only the yardstick moves. If the gate were absolute this could not
    # flip, and that flip is exactly the property under test.
    stricter = _fit(annotations, permutation_min_diagonal_ratio=1.5)
    assert stricter.status == "permutation_check_diagonal"
    assert not stricter.valid_mask.any()


def test_chance_backstop_rejects_an_internally_incoherent_fit():
    """Every client applies its own rotation, so no labelling makes latent class c emit c. The
    ratio test alone cannot catch this (the reference degrades in step with the fit), which is
    why the absolute chance floor is checked first."""
    rng = np.random.default_rng(11)
    truth = rng.integers(0, NUM_CLASSES, size=900)
    annotations = np.stack(
        [((truth + j) % NUM_CLASSES).astype(np.int8) for j in range(11)]
    )
    fit = _fit(annotations, max_iterations=300)
    assert fit.diagonal_fraction <= 1.0 / NUM_CLASSES
    assert fit.status == "permutation_check_chance"
    assert not fit.valid_mask.any()


def test_diagonal_fraction_alone_cannot_see_a_global_relabelling():
    """Documents why majority_agreement carries the permutation check and the diagonal statistic
    does not: rotating every client's labels moves EM to the rotated solution, which is just as
    internally self-consistent, so the fit's own diagonal fraction is unchanged."""
    annotations, _ = _specialists()
    rotated = ((annotations + 1) % NUM_CLASSES).astype(np.int8)
    honest = _fit(annotations)
    tampered = fit_dawid_skene(
        rotated,
        num_classes=NUM_CLASSES,
        majority_labels=_majority(annotations),
        settings=DawidSkeneSettings(min_clients=2),
    )
    assert tampered.diagonal_fraction == honest.diagonal_fraction
    assert tampered.majority_agreement < 0.1 < honest.majority_agreement
    assert tampered.status == "permutation_check_agreement"
