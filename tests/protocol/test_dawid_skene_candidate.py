"""The estimator/policy split: what a rejected fit still exposes, and class alignment.

``tests/protocol/test_dawid_skene.py`` covers the accept path and every fallback status. What is
proven here is the seam experiment 1 depends on (DENEY_1_SCENARIO_3_DENEY_PLANI.md sections 4-5):
a fit that failed a *policy* gate still carries the labels it would have broadcast, a fit that
failed *arithmetic* carries nothing, and the class alignment resolves latent-class identity
without ever consulting majority or ground truth.
"""

import numpy as np

from ssfl.protocols.dawid_skene import (
    ABSTAIN,
    DawidSkeneSettings,
    _align_classes,
    fit_dawid_skene,
)

from tests.protocol.test_dawid_skene import NUM_CLASSES, _fit, _majority, _synthetic


# --- candidate exposure ----------------------------------------------------------------------


def test_non_converged_fit_keeps_its_last_finite_posterior():
    """The DS-only arm broadcasts this. Non-convergence is a policy gate, not a numerical fault:
    the parameters were checked finite and normalized on every iteration EM actually ran."""
    annotations, _ = _synthetic(seed=29)
    fit = _fit(annotations, max_iterations=1, min_iterations=1)

    assert fit.status == "not_converged"
    assert fit.numerically_valid
    assert fit.candidate_labels is not None
    assert (fit.candidate_labels != ABSTAIN).any()
    assert fit.posterior is not None and np.isfinite(fit.posterior).all()
    # The candidate is the argmax of the posterior that was kept, on the items it considered valid.
    expected = np.where(
        fit.candidate_valid_mask, fit.posterior.argmax(axis=1), ABSTAIN
    ).astype(np.int64)
    np.testing.assert_array_equal(fit.candidate_labels, expected)


def test_rejected_fit_still_hides_its_labels_from_the_hybrid_contract():
    """``labels``/``valid_mask`` remain the accept-path fields. A caller that reads them without
    checking ``ok`` gets nothing usable, exactly as before the candidate was added."""
    annotations, _ = _synthetic(seed=29)
    fit = _fit(annotations, max_iterations=1, min_iterations=1)

    assert not fit.ok
    assert not fit.valid_mask.any()
    assert set(np.unique(fit.labels)) == {ABSTAIN}


def test_permutation_gate_rejection_is_still_numerically_valid():
    annotations, _ = _synthetic(seed=31)
    rotated = ((annotations + 1) % NUM_CLASSES).astype(np.int8)
    fit = fit_dawid_skene(
        rotated,
        num_classes=NUM_CLASSES,
        majority_labels=_majority(annotations),
        settings=DawidSkeneSettings(min_clients=2, permutation_min_majority_agreement=0.8),
    )

    assert fit.status == "permutation_check_agreement"
    assert fit.numerically_valid
    assert fit.candidate_labels is not None


def test_a_fit_that_never_reached_em_exposes_no_candidate():
    """Arithmetic failure, not policy: there is no posterior to broadcast, so the DS-only arm has
    nothing to fall back to and must stop the run rather than invent a label."""
    annotations, _ = _synthetic(num_clients=2, accuracy_by_client=[0.9, 0.9], seed=23)
    fit = fit_dawid_skene(
        annotations,
        num_classes=NUM_CLASSES,
        majority_labels=_majority(annotations),
        settings=DawidSkeneSettings(min_clients=3),
    )

    assert fit.status == "insufficient_clients"
    assert not fit.numerically_valid
    assert fit.candidate_labels is None
    assert fit.candidate_valid_mask is None


# --- class alignment -------------------------------------------------------------------------


def test_align_classes_inverts_a_permuted_fit():
    rng = np.random.default_rng(3)
    posterior = rng.random((20, NUM_CLASSES))
    posterior /= posterior.sum(axis=1, keepdims=True)
    prior = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
    # Confusion is near-identity, so the only optimal mapping is the identity...
    confusion = np.tile(np.eye(NUM_CLASSES) * 0.7 + 0.3 / NUM_CLASSES, (5, 1, 1))
    confusion /= confusion.sum(axis=2, keepdims=True)
    # ...until the latent axis is rotated, which is what a relabelled EM solution looks like.
    rotation = np.roll(np.arange(NUM_CLASSES), 1)
    rotated_posterior = posterior[:, rotation]
    rotated_confusion = confusion[:, rotation, :]
    rotated_prior = prior[rotation]

    aligned_posterior, _, aligned_confusion, permutation, score = _align_classes(
        rotated_posterior, rotated_prior, rotated_confusion
    )

    np.testing.assert_allclose(aligned_posterior, posterior)
    np.testing.assert_allclose(aligned_confusion, confusion)
    # Rotated latent slot c holds original class rotation[c], so that is where it maps back.
    assert permutation == tuple(int(k) for k in rotation)
    assert score > 0.0


def test_alignment_is_a_no_op_on_a_fit_that_was_already_aligned():
    """The hybrid arm gets the same candidate as the DS-only arm, so alignment must not perturb a
    healthy fit -- otherwise switching it on would silently change what hybrid broadcasts."""
    annotations, _ = _synthetic(
        accuracy_by_client=[0.95, 0.95, 0.35, 0.35, 0.35, 0.35, 0.35], seed=11
    )
    plain = _fit(annotations)
    aligned = _fit(annotations, class_alignment=True)

    assert plain.ok and aligned.ok
    assert aligned.alignment_permutation == tuple(range(NUM_CLASSES))
    assert aligned.alignment_score > 0.0
    np.testing.assert_array_equal(plain.labels, aligned.labels)


def test_alignment_reads_no_majority_labels():
    """Section 5 of the plan: the mapping is scored on the fitted confusions only. Feeding the
    estimator a deliberately wrong majority vector must not move the permutation it chooses."""
    annotations, _ = _synthetic(seed=13)
    settings = DawidSkeneSettings(min_clients=2, class_alignment=True)
    honest = fit_dawid_skene(
        annotations,
        num_classes=NUM_CLASSES,
        majority_labels=_majority(annotations),
        settings=settings,
    )
    misled = fit_dawid_skene(
        annotations,
        num_classes=NUM_CLASSES,
        majority_labels=(_majority(annotations) + 1) % NUM_CLASSES,
        settings=settings,
    )

    assert honest.alignment_permutation == misled.alignment_permutation
    np.testing.assert_array_equal(honest.candidate_labels, misled.candidate_labels)
