"""Online-EM Dawid-Skene: the three properties the streaming aggregator has to have.

The arm this covers changes the server-side aggregation function and nothing else, so the first
thing to protect is that runs made before it existed still mean what they meant. The other two
tests are the two load-bearing claims: majority vote is a point in this estimator's family, and
a client's accumulated statistics follow the client and not its row index.
"""

import numpy as np
import pytest

from ssfl.protocols.dawid_skene import (
    ABSTAIN,
    DawidSkeneSettings,
    _confusion_pseudocounts,
    fit_dawid_skene,
)

NUM_CLASSES = 6


def _annotations(seed: int = 11, num_clients: int = 12, num_items: int = 600):
    """Heterogeneous annotators: the regime Dawid-Skene exists for, and the one it fails in."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, NUM_CLASSES, num_items)
    matrix = np.empty((num_clients, num_items), dtype=np.int8)
    for j in range(num_clients):
        accuracy = rng.uniform(0.35, 0.9)
        wrong = rng.random(num_items) > accuracy
        matrix[j] = np.where(wrong, rng.integers(0, NUM_CLASSES, num_items), truth)
        matrix[j][rng.random(num_items) < 0.1] = ABSTAIN
    senders = tuple(f"client-{j:02d}" for j in range(num_clients))
    return matrix, senders, truth


def _majority(annotations: np.ndarray) -> np.ndarray:
    """Same tie rule as ``aggregate_votes``: lowest class index wins, all-abstain is ABSTAIN."""
    labels = np.full(annotations.shape[1], ABSTAIN, dtype=np.int64)
    for i, column in enumerate(annotations.T):
        voted = column[column != ABSTAIN]
        if voted.size:
            labels[i] = int(np.bincount(voted, minlength=NUM_CLASSES).argmax())
    return labels


def test_defaults_are_the_historical_batch_estimator():
    """No config override must change anything already on disk.

    The uniform prior has to reduce to the scalar pseudocount the M-step used before it existed,
    and the batch path must not produce state for a caller to thread anywhere.
    """
    settings = DawidSkeneSettings()
    assert settings.confusion_prior == "uniform"
    assert settings.state_decay == 0.0
    pseudocounts = _confusion_pseudocounts(settings, NUM_CLASSES, NUM_CLASSES)
    assert np.array_equal(
        pseudocounts, np.full((NUM_CLASSES, NUM_CLASSES), settings.confusion_pseudocount)
    )

    annotations, senders, _ = _annotations()
    fit = fit_dawid_skene(
        annotations, NUM_CLASSES, _majority(annotations), settings=settings, senders=senders
    )
    assert fit.ok
    assert fit.state is None
    assert fit.stop_reason != "online_em"


def test_diagonal_prior_is_a_reweighting_not_a_rescaling():
    """``diagonal`` must actually move mass onto the diagonal, at the same total pseudocount."""
    settings = DawidSkeneSettings(
        confusion_prior="diagonal", confusion_prior_diagonal=0.9, confusion_pseudocount=20.0
    )
    pseudocounts = _confusion_pseudocounts(settings, NUM_CLASSES, NUM_CLASSES)
    assert pseudocounts.sum(axis=1) == pytest.approx(np.full(NUM_CLASSES, 20.0))
    assert np.diag(pseudocounts) == pytest.approx(np.full(NUM_CLASSES, 18.0))
    off_diagonal = pseudocounts[~np.eye(NUM_CLASSES, dtype=bool)]
    assert off_diagonal == pytest.approx(np.full(off_diagonal.size, 2.0 / (NUM_CLASSES - 1)))

    with pytest.raises(ValueError, match="unknown confusion_prior"):
        _confusion_pseudocounts(DawidSkeneSettings(confusion_prior="nonsense"), 3, 3)


def test_majority_vote_is_a_point_in_the_family():
    """With no accumulated evidence every client shares one confusion matrix.

    The per-vote log-likelihood ratio is then a constant, so the argmax is the vote count. This is
    what makes the arm safe: the worst it can degrade to is the baseline it replaces, and the
    first round it ever runs is exactly majority vote.
    """
    annotations, senders, _ = _annotations()
    majority = _majority(annotations)
    settings = DawidSkeneSettings(
        confusion_prior="diagonal",
        confusion_prior_diagonal=0.9,
        confusion_pseudocount=20.0,
        state_decay=0.9,
        state_init_rounds=1e9,
    )
    fit = fit_dawid_skene(
        annotations, NUM_CLASSES, majority, settings=settings, senders=senders
    )
    assert fit.numerically_valid
    assert fit.stop_reason == "online_em"
    assert fit.iterations == 1
    # Ties excluded, and only ties: the equality is exact on every item whose vote count has a
    # unique winner. On a tie the two disagree because the log-likelihood is accumulated client by
    # client in float64, so equal vote counts do not produce bit-equal scores and argmax's
    # lowest-index rule breaks the wrong way. That is a property of float addition, not of the
    # aggregator -- the batch path has it too -- and it is why this asserts the algebra rather
    # than byte equality.
    counts = np.zeros((annotations.shape[1], NUM_CLASSES), dtype=np.int64)
    for row in annotations:
        voted = np.nonzero(row != ABSTAIN)[0]
        counts[voted, row[voted]] += 1
    top_two = np.sort(counts, axis=1)[:, -2:]
    decisive = top_two[:, 1] > top_two[:, 0]

    mask = fit.candidate_valid_mask & (majority != ABSTAIN) & decisive
    assert mask.sum() > 0.9 * fit.candidate_valid_mask.sum()
    assert np.array_equal(fit.candidate_labels[mask], majority[mask])


def test_state_follows_the_sender_not_the_row():
    """A client that drops out and rejoins must get its own statistics back.

    Rows are re-indexed whenever the participating set changes, so keying the accumulated counts
    by position would quietly hand one client another client's confusion matrix.
    """
    annotations, senders, _ = _annotations()
    settings = DawidSkeneSettings(
        confusion_prior="diagonal",
        confusion_prior_diagonal=0.9,
        confusion_pseudocount=20.0,
        state_decay=0.9,
        state_init_rounds=5.0,
    )
    first = fit_dawid_skene(
        annotations, NUM_CLASSES, _majority(annotations), settings=settings, senders=senders
    )
    assert first.state is not None
    assert set(first.state.counts) == set(senders)
    assert first.state.rounds == 1

    # Round two with the same votes but half the clients absent, then round three with everyone
    # back. If state were positional, the survivors' statistics would land on the wrong senders.
    keep = [0, 2, 4, 6, 8, 10]
    subset = annotations[keep]
    subset_senders = tuple(senders[j] for j in keep)
    second = fit_dawid_skene(
        subset, NUM_CLASSES, _majority(subset), settings=settings,
        senders=subset_senders, state=first.state,
    )
    assert second.state is not None
    # Absent clients keep their round-one statistics untouched rather than decaying or vanishing.
    for j, sender in enumerate(senders):
        if sender in subset_senders:
            continue
        assert np.array_equal(second.state.counts[sender], first.state.counts[sender])
    for sender in subset_senders:
        assert not np.array_equal(second.state.counts[sender], first.state.counts[sender])

    third = fit_dawid_skene(
        annotations, NUM_CLASSES, _majority(annotations), settings=settings,
        senders=senders, state=second.state,
    )
    assert third.numerically_valid
    assert third.state.rounds == 3

    # Reply order cannot change the answer: build_annotation_matrix sorts, and the state lookup is
    # by sender, so a permuted batch is the same batch.
    order = [3, 7, 1, 9, 5, 11, 0, 4, 8, 2, 10, 6]
    permuted = fit_dawid_skene(
        annotations[order], NUM_CLASSES, _majority(annotations), settings=settings,
        senders=tuple(senders[j] for j in order), state=second.state,
    )
    assert np.array_equal(permuted.candidate_labels, third.candidate_labels)


def test_streaming_without_senders_fails_closed():
    """No sender ids means no way to key the statistics, so the fit must refuse rather than guess."""
    annotations, _, _ = _annotations()
    settings = DawidSkeneSettings(state_decay=0.9, state_init_rounds=5.0)
    fit = fit_dawid_skene(annotations, NUM_CLASSES, _majority(annotations), settings=settings)
    assert fit.status == "state_requires_senders"
    assert not fit.numerically_valid
    assert not fit.valid_mask.any()
