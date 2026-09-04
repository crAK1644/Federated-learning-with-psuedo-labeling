"""Hard-label Dawid-Skene estimator for server-side label aggregation.

Pure NumPy, no Flower and no torch import -- the same testability contract as the rest of
``protocols/``. ``strategies/ssfl.py`` is the only production caller.

Model: every open-set sample ``i`` has a hidden true class ``z_i``; every client ``j`` has a
class-conditional confusion matrix ``M_j[c, k] = P(client j says k | truth is c)``. The default
observation model treats ``ABSTAIN`` as missing data for backward compatibility. The experimental
explicit mode adds it as a ``K+1``-th emitted outcome while keeping the latent truth space at K
classes. EM alternates a posterior E-step with a pseudocount-regularized M-step, both in float64;
the E-step works in log space with log-sum-exp normalization because a product over many clients
of probabilities near the epsilon floor underflows float64 directly.

This module is the estimator, not the policy. It reports what it found; ``strategies/ssfl.py``
decides what to broadcast. That split is what lets the hybrid arm and the Dawid-Skene-only arm of
DENEY_1_SCENARIO_3_DENEY_PLANI.md consume one identical fit and differ only in what they do with
it: hybrid broadcasts on ``status == "ok"``, Dawid-Skene-only broadcasts on ``numerically_valid``.

Safety contract (DAWID_SKENE_REVIEW.md F3, and the approval brief's safeguards):

* Every failure path returns ``status != ok`` so a caller that wants the hybrid policy falls back
  to deterministic majority. ``labels``/``valid_mask`` stay empty unless the status is ``ok``.
* ``numerically_valid`` is the weaker, policy-free question: did EM produce finite parameters and
  a normalized posterior? A fit can be numerically valid and still fail a gate (non-convergence,
  or a majority-anchored permutation check). Such a fit exposes ``candidate_labels`` /
  ``candidate_valid_mask``; a numerically broken one exposes nothing at all.
* The likelihood is invariant to permuting the hidden classes, and the returned integer is used
  directly as a distillation target, so a permutation check (diagonal dominance + agreement with
  majority) runs after convergence and is a fallback condition, not a warning.
* Nothing here reads open-set ground truth. The estimator only ever sees client labels.

``status``, the exclusion vocabulary and what "fallback" does and does not mean are defined
once in DAWID_SKENE_GLOSSARY.md; the terms in this module follow it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

ABSTAIN = -1


@dataclass(frozen=True)
class DawidSkeneSettings:
    """Estimator knobs. Mirrors the ``dawid_skene_*`` fields of ``ExperimentConfig`` so this
    module stays importable without pydantic; ``from_config`` does the one-way translation."""

    max_iterations: int = 100
    min_iterations: int = 2
    tolerance: float = 1e-6
    initialization_pseudocount: float = 0.01
    confusion_pseudocount: float = 0.1
    class_prior_pseudocount: float = 1.0
    min_item_annotations: int = 1
    min_client_annotations: int = 1
    min_clients: int = 3
    posterior_threshold: float = 0.0
    damping: float = 1.0
    epsilon: float = 1e-12
    # Fraction of majority's own diagonal fraction the fit must reach, NOT an absolute floor --
    # see the permutation check in fit_dawid_skene for why an absolute floor cannot work here.
    permutation_min_diagonal_ratio: float = 0.7
    permutation_min_majority_agreement: float = 0.5
    # Deterministic latent-class -> label-space permutation after EM. Scored on the fitted
    # confusion matrices alone: no majority labels, no ground truth. See ``_align_classes``.
    class_alignment: bool = False
    # False reproduces the original missing-at-random likelihood exactly. True gives each client
    # a K+1 output alphabet and lets its decision to abstain carry class-conditional evidence.
    explicit_abstention: bool = False
    # ``one_coin`` constrains each client to a single accuracy parameter. Conditional on an
    # error, probability is uniform over the other K-1 labels. This prevents a highly flexible
    # per-client confusion tensor from turning correlated errors into spurious expertise.
    confusion_model: str = "full"
    one_coin_min_accuracy: float = 0.9
    one_coin_pseudocount: float = 1.0
    # Window 1 is the round-local method. Window 2 adds the immediately preceding round with
    # ``temporal_decay`` weight; current observations always retain weight 1.0.
    temporal_window: int = 1
    temporal_decay: float = 0.8

    @classmethod
    def from_config(cls, config) -> "DawidSkeneSettings":
        abstention_mode = getattr(config, "dawid_skene_abstention_mode", "missing")
        abstention_mode = getattr(abstention_mode, "value", abstention_mode)
        return cls(
            max_iterations=config.dawid_skene_max_iterations,
            min_iterations=config.dawid_skene_min_iterations,
            tolerance=config.dawid_skene_tolerance,
            initialization_pseudocount=config.dawid_skene_initialization_pseudocount,
            confusion_pseudocount=config.dawid_skene_confusion_pseudocount,
            class_prior_pseudocount=config.dawid_skene_class_prior_pseudocount,
            min_item_annotations=config.dawid_skene_min_item_annotations,
            min_client_annotations=config.dawid_skene_min_client_annotations,
            min_clients=config.dawid_skene_min_clients,
            posterior_threshold=config.dawid_skene_posterior_threshold,
            damping=config.dawid_skene_damping,
            epsilon=config.dawid_skene_epsilon,
            permutation_min_diagonal_ratio=config.dawid_skene_permutation_min_diagonal_ratio,
            permutation_min_majority_agreement=(
                config.dawid_skene_permutation_min_majority_agreement
            ),
            class_alignment=config.dawid_skene_class_alignment,
            explicit_abstention=abstention_mode == "explicit",
            confusion_model=config.dawid_skene_confusion_model.value,
            one_coin_min_accuracy=config.dawid_skene_one_coin_min_accuracy,
            one_coin_pseudocount=config.dawid_skene_one_coin_pseudocount,
            temporal_window=config.dawid_skene_temporal_window,
            temporal_decay=config.dawid_skene_temporal_decay,
        )


@dataclass(frozen=True)
class DawidSkeneFit:
    """``status == "ok"`` is the only case in which ``labels``/``valid_mask`` may be broadcast.

    ``confusion``, ``prior`` and ``posterior`` are restricted diagnostics (per-client behaviour
    and per-sample uncertainty): the caller must not put them on the wire and must not write them
    to the default audit output. They exist for offline validation against a reference
    implementation, which needs the fitted distributions and not just the argmax labels.
    """

    status: str  # "ok", or the fallback reason
    labels: np.ndarray  # int64, len num_open, ABSTAIN where not usable
    valid_mask: np.ndarray  # bool, len num_open
    iterations: int
    converged: bool
    stop_reason: str
    log_likelihood: float
    objective: float
    eligible_clients: int
    excluded_clients: tuple[str, ...]
    diagonal_fraction: float
    reference_diagonal_fraction: float
    majority_agreement: float
    max_posterior_mean: float
    # Row order of ``confusion``; restricted along with it, and empty unless the fit reached EM.
    eligible_senders: tuple[str, ...] = ()
    # EM finished with finite parameters and a normalized posterior. Weaker than ``ok``: it says
    # the numbers are usable, not that the fit passed the hybrid arm's gates.
    numerically_valid: bool = False
    # The decision this fit would broadcast, populated whenever ``numerically_valid``. Identical
    # to ``labels``/``valid_mask`` when the status is ``ok``.
    candidate_labels: np.ndarray | None = field(default=None, repr=False)
    candidate_valid_mask: np.ndarray | None = field(default=None, repr=False)
    # ``alignment_permutation[c]`` is the output class latent class ``c`` was mapped onto; empty
    # when alignment is off. ``alignment_score`` is the total confusion mass the mapping captured.
    alignment_permutation: tuple[int, ...] = ()
    alignment_score: float = 0.0
    confusion: np.ndarray | None = field(default=None, repr=False)
    prior: np.ndarray | None = field(default=None, repr=False)
    posterior: np.ndarray | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _failed(reason: str, num_open: int, **overrides) -> DawidSkeneFit:
    base = dict(
        status=reason,
        labels=np.full(num_open, ABSTAIN, dtype=np.int64),
        valid_mask=np.zeros(num_open, dtype=bool),
        iterations=0,
        converged=False,
        stop_reason=reason,
        log_likelihood=float("nan"),
        objective=float("nan"),
        eligible_clients=0,
        excluded_clients=(),
        diagonal_fraction=0.0,
        reference_diagonal_fraction=0.0,
        majority_agreement=0.0,
        max_posterior_mean=0.0,
    )
    base.update(overrides)
    return DawidSkeneFit(**base)


def _diagonal_fraction(posterior, slices, onehots, num_classes: int, pseudocount: float) -> float:
    """Fraction of supported ``(client, class)`` pairs whose modal emission is the class itself.

    Both the fitted posterior and the majority reference go through this one function: the
    permutation check compares the two numbers, so computing them any differently would make the
    comparison meaningless.
    """
    hits = 0
    total = 0
    for j, (index, _) in enumerate(slices):
        # "Supported" = the reference actually places mass on class c for items this client saw;
        # unsupported rows are pure pseudocount and would report whatever the prior says.
        supported = posterior[index].sum(axis=0) > 1.0
        if not supported.any():
            continue
        counts = posterior[index].T @ onehots[j] + pseudocount
        modal = counts.argmax(axis=1)
        hits += int((modal[supported] == np.nonzero(supported)[0]).sum())
        total += int(supported.sum())
    return float(hits / total) if total else 0.0


def _align_classes(
    posterior: np.ndarray, prior: np.ndarray, confusion: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], float]:
    """Permute the latent classes onto the label space the clients actually emit.

    The likelihood is invariant to relabelling the hidden classes, so EM can converge to a fit in
    which latent class 0 means ``gafgyt.udp``. The hybrid arm catches that with a majority-anchored
    agreement gate, but the Dawid-Skene-only arm has no majority path by construction, so the
    permutation has to be resolved inside the estimator instead of accepted or rejected outside it
    (DENEY_1_SCENARIO_3_DENEY_PLANI.md section 5).

    Score: the total confusion mass a mapping captures, ``sum_j Theta_j[c, k]`` for latent ``c``
    mapped to output ``k``. That reads only the fitted confusions -- no majority labels and no
    ground truth -- and the maximizing assignment is exact via Hungarian rather than greedy, which
    for 11 classes can differ. ``linear_sum_assignment`` is deterministic on identical input; the
    scores are float64 sums over 89 clients, so exact ties are not something to design around.
    """
    # In explicit-abstention mode the last output column means silence, not a label identity, and
    # must not participate in the latent-class-to-emitted-label assignment.
    num_classes = posterior.shape[1]
    score = confusion[:, :, :num_classes].sum(axis=0)
    rows, permutation = linear_sum_assignment(-score)
    aligned_posterior = np.empty_like(posterior)
    aligned_posterior[:, permutation] = posterior
    aligned_prior = np.empty_like(prior)
    aligned_prior[permutation] = prior
    # Confusion rows are latent classes and get permuted; columns are emitted labels and do not.
    aligned_confusion = np.empty_like(confusion)
    aligned_confusion[:, permutation, :] = confusion
    return (
        aligned_posterior,
        aligned_prior,
        aligned_confusion,
        tuple(int(k) for k in permutation),
        float(score[rows, permutation].sum()),
    )


def build_annotation_matrix(
    labels_by_sender: list[tuple[str, np.ndarray]], num_open: int, num_classes: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    """``(J, num_open)`` int8 matrix of client labels in sorted-sender order, plus that order.

    Sender-sorted rather than reply-arrival order: float64 EM is not order-invariant (addition is
    not associative), and Ray/Flower reply arrival order is not reproducible across runs of the
    same seeded simulation. This is the same argument that makes ``aggregate_soft`` sort.
    """
    ordered = sorted(labels_by_sender, key=lambda item: item[0])
    senders = tuple(sender for sender, _ in ordered)
    matrix = np.full((len(ordered), num_open), ABSTAIN, dtype=np.int8)
    for row, (_, labels) in enumerate(ordered):
        values = np.asarray(labels)
        if values.shape != (num_open,):
            raise ValueError(f"sender {senders[row]}: expected ({num_open},), got {values.shape}")
        legal = (values == ABSTAIN) | ((values >= 0) & (values < num_classes))
        if not legal.all():
            raise ValueError(f"sender {senders[row]}: label outside [0,{num_classes}) and != -1")
        matrix[row] = values.astype(np.int8)
    return matrix, senders


def _fit_one_coin_dawid_skene(
    annotations: np.ndarray,
    num_classes: int,
    majority_labels: np.ndarray,
    settings: DawidSkeneSettings,
    senders: tuple[str, ...],
    annotation_history: tuple[np.ndarray, ...],
) -> DawidSkeneFit:
    """Fit the constrained one-coin Dawid-Skene likelihood.

    Each eligible client has one accuracy ``q_j``. Given a mistake, its mass is uniform over the
    remaining labels. Abstention is class-independent, so it is missing from the class likelihood
    rather than being allowed to overwhelm the emitted labels as a K+1-th class-conditional event.
    """
    if settings.explicit_abstention:
        raise ValueError("one_coin does not support class-conditional explicit abstention")
    if settings.temporal_window < 1:
        raise ValueError("temporal_window must be >= 1")
    if not 0.0 < settings.temporal_decay <= 1.0:
        raise ValueError("temporal_decay must be in (0, 1]")
    history = tuple(np.asarray(previous) for previous in annotation_history)
    for previous in history:
        if previous.shape != annotations.shape:
            raise ValueError(
                "each temporal annotation matrix must have the same shape as the current round"
            )
        if np.any((previous != ABSTAIN) & ((previous < 0) | (previous >= num_classes))):
            raise ValueError("temporal annotations contain a label outside the class range")
    history = history[-(settings.temporal_window - 1) :] if settings.temporal_window > 1 else ()
    num_open = annotations.shape[1]
    observed = annotations != ABSTAIN
    per_client = observed.sum(axis=1)
    eligible = per_client >= settings.min_client_annotations
    excluded = tuple(senders[j] for j in np.nonzero(~eligible)[0] if j < len(senders))
    eligible_senders = tuple(senders[j] for j in np.nonzero(eligible)[0] if j < len(senders))
    if int(eligible.sum()) < settings.min_clients:
        return _failed("insufficient_clients", num_open, excluded_clients=excluded)

    annotations = annotations[eligible]
    observed = observed[eligible]
    history = tuple(previous[eligible] for previous in history)
    item_counts = observed.sum(axis=0).astype(np.int64)
    if not item_counts.any():
        return _failed("no_observations", num_open, excluded_clients=excluded)
    num_eligible = annotations.shape[0]
    eps = settings.epsilon
    current_slices = [
        (np.nonzero(observed[j])[0], annotations[j][observed[j]].astype(np.int64))
        for j in range(num_eligible)
    ]
    current_onehots = [
        np.eye(num_classes, dtype=np.float64)[labels] for _, labels in current_slices
    ]
    temporal_annotations = (*history, annotations)
    temporal_slices = []
    for round_annotations in temporal_annotations:
        round_observed = round_annotations != ABSTAIN
        temporal_slices.append(
            [
                (
                    np.nonzero(round_observed[j])[0],
                    round_annotations[j][round_observed[j]].astype(np.int64),
                )
                for j in range(num_eligible)
            ]
        )
    round_weights = settings.temporal_decay ** np.arange(
        len(temporal_slices) - 1, -1, -1, dtype=np.float64
    )
    # Keep one effective round of evidence. Without normalization, window 2 would silently make
    # the posterior 1.8x sharper than the round-local control in addition to adding temporal
    # information, confounding smoothing with a change in likelihood temperature.
    round_weights /= round_weights.sum()

    weighted_votes = np.zeros((num_open, num_classes), dtype=np.float64)
    for weight, round_slices in zip(round_weights, temporal_slices, strict=True):
        for index, labels in round_slices:
            np.add.at(weighted_votes, (index, labels), weight)
    evidence_items = weighted_votes.sum(axis=1) > 0.0
    posterior = weighted_votes + settings.initialization_pseudocount
    posterior /= posterior.sum(axis=1, keepdims=True)

    prior = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
    accuracies = np.full(num_eligible, settings.one_coin_min_accuracy, dtype=np.float64)
    confusion = np.empty((num_eligible, num_classes, num_classes), dtype=np.float64)
    previous_objective = -np.inf
    log_likelihood = float("nan")
    objective = float("nan")
    iterations = 0
    converged = False
    stop_reason = "iteration_cap"
    beta = settings.one_coin_pseudocount

    for iterations in range(1, settings.max_iterations + 1):
        prior_counts = posterior[evidence_items].sum(axis=0) + settings.class_prior_pseudocount
        new_prior = prior_counts / prior_counts.sum()
        new_accuracies = np.empty_like(accuracies)
        for j in range(num_eligible):
            expected_correct = 0.0
            effective_count = 0.0
            for weight, round_slices in zip(round_weights, temporal_slices, strict=True):
                index, labels = round_slices[j]
                expected_correct += weight * float(posterior[index, labels].sum())
                effective_count += weight * len(index)
            estimate = (expected_correct + beta) / (effective_count + 2.0 * beta)
            new_accuracies[j] = max(float(estimate), settings.one_coin_min_accuracy)

        if settings.damping < 1.0:
            new_prior = settings.damping * new_prior + (1.0 - settings.damping) * prior
            new_accuracies = (
                settings.damping * new_accuracies + (1.0 - settings.damping) * accuracies
            )
        prior, accuracies = new_prior, new_accuracies

        off_diagonal = (1.0 - accuracies) / (num_classes - 1)
        confusion[:] = off_diagonal[:, None, None]
        diagonal = np.arange(num_classes)
        confusion[:, diagonal, diagonal] = accuracies[:, None]
        if not (np.isfinite(prior).all() and np.isfinite(confusion).all()):
            return _failed("non_finite_parameters", num_open, iterations=iterations)

        log_score = np.tile(np.log(np.maximum(prior, eps)), (num_open, 1))
        log_confusion = np.log(np.maximum(confusion, eps))
        for weight, round_slices in zip(round_weights, temporal_slices, strict=True):
            for j, (index, labels) in enumerate(round_slices):
                log_score[index] += weight * log_confusion[j][:, labels].T
        row_max = log_score.max(axis=1, keepdims=True)
        exp_shifted = np.exp(log_score - row_max)
        row_sum = exp_shifted.sum(axis=1, keepdims=True)
        posterior = exp_shifted / row_sum
        if not np.isfinite(posterior).all():
            return _failed("non_finite_posterior", num_open, iterations=iterations)

        per_item_loglik = (np.log(row_sum) + row_max).ravel()
        log_likelihood = float(per_item_loglik[evidence_items].sum())
        objective = float(
            log_likelihood
            + settings.class_prior_pseudocount * np.log(np.maximum(prior, eps)).sum()
            + beta
            * (
                np.log(np.maximum(accuracies, eps))
                + np.log(np.maximum(1.0 - accuracies, eps))
            ).sum()
        )
        if not np.isfinite(objective):
            return _failed("non_finite_objective", num_open, iterations=iterations)
        scale = max(abs(previous_objective), 1.0)
        delta = objective - previous_objective
        if np.isfinite(previous_objective):
            if delta < -settings.tolerance * scale:
                return _failed("objective_decreased", num_open, iterations=iterations)
            if iterations >= settings.min_iterations and delta <= settings.tolerance * scale:
                converged = True
                stop_reason = "tolerance"
                previous_objective = objective
                break
        previous_objective = objective

    alignment_permutation: tuple[int, ...] = ()
    alignment_score = 0.0
    if settings.class_alignment:
        posterior, prior, confusion, alignment_permutation, alignment_score = _align_classes(
            posterior, prior, confusion
        )

    candidate = posterior.argmax(axis=1).astype(np.int64)
    max_posterior = posterior.max(axis=1)
    valid_mask = (item_counts >= settings.min_item_annotations) & (
        max_posterior >= settings.posterior_threshold
    )
    labels = np.where(valid_mask, candidate, ABSTAIN).astype(np.int64)
    diagonal_fraction = _diagonal_fraction(
        posterior,
        current_slices,
        current_onehots,
        num_classes,
        settings.confusion_pseudocount,
    )
    observed_majority = majority_labels != ABSTAIN
    majority_posterior = np.zeros((num_open, num_classes), dtype=np.float64)
    majority_posterior[
        np.nonzero(observed_majority)[0], majority_labels[observed_majority]
    ] = 1.0
    reference_diagonal_fraction = _diagonal_fraction(
        majority_posterior,
        current_slices,
        current_onehots,
        num_classes,
        settings.confusion_pseudocount,
    )
    comparable = valid_mask & observed_majority
    majority_agreement = (
        float((labels[comparable] == majority_labels[comparable]).mean())
        if comparable.any()
        else 0.0
    )
    diagnostics = dict(
        iterations=iterations,
        converged=converged,
        stop_reason=stop_reason,
        log_likelihood=log_likelihood,
        objective=objective,
        eligible_clients=num_eligible,
        excluded_clients=excluded,
        eligible_senders=eligible_senders,
        diagonal_fraction=diagonal_fraction,
        reference_diagonal_fraction=reference_diagonal_fraction,
        majority_agreement=majority_agreement,
        max_posterior_mean=float(max_posterior[valid_mask].mean()) if valid_mask.any() else 0.0,
        alignment_permutation=alignment_permutation,
        alignment_score=alignment_score,
        numerically_valid=True,
        candidate_labels=labels,
        candidate_valid_mask=valid_mask,
        confusion=confusion,
        prior=prior,
        posterior=posterior,
    )
    if not converged:
        return _failed("not_converged", num_open, **diagnostics)
    if diagonal_fraction <= 1.0 / num_classes:
        return _failed("permutation_check_chance", num_open, **diagnostics)
    if diagonal_fraction < settings.permutation_min_diagonal_ratio * reference_diagonal_fraction:
        return _failed("permutation_check_diagonal", num_open, **diagnostics)
    if majority_agreement < settings.permutation_min_majority_agreement:
        return _failed("permutation_check_agreement", num_open, **diagnostics)
    return DawidSkeneFit(status="ok", labels=labels, valid_mask=valid_mask, **diagnostics)


def fit_dawid_skene(
    annotations: np.ndarray,
    num_classes: int,
    majority_labels: np.ndarray,
    settings: DawidSkeneSettings | None = None,
    senders: tuple[str, ...] = (),
    annotation_history: tuple[np.ndarray, ...] = (),
) -> DawidSkeneFit:
    """Fit hard-label Dawid-Skene on an ``annotations`` matrix of shape ``(J, N)``.

    ``majority_labels`` is the deterministic majority result for the same batch; it is used only
    for the permutation check and the reported disagreement rate, never as an EM input.

    With ``explicit_abstention=False`` the historical likelihood is reproduced exactly and
    ``ABSTAIN`` contributes nothing. With it enabled, latent truth still has ``num_classes``
    states but the emitted alphabet has ``num_classes + 1`` outcomes, the final one representing
    abstention. The broadcast validity rule remains based on non-abstaining votes, so changing the
    likelihood cannot change which open samples are scored or distilled. ``annotation_history``
    is used only by temporal one-coin DS; it contains older matrices in chronological order and
    never changes the current round's validity mask.
    """
    settings = settings or DawidSkeneSettings()
    annotations = np.asarray(annotations)
    num_open = annotations.shape[1] if annotations.ndim == 2 else 0
    if annotations.ndim != 2 or num_open == 0:
        return _failed("no_annotations", num_open)
    if settings.confusion_model == "one_coin":
        return _fit_one_coin_dawid_skene(
            annotations,
            num_classes,
            majority_labels,
            settings,
            senders,
            annotation_history,
        )
    if annotation_history:
        raise ValueError("annotation_history is supported only by one_coin Dawid-Skene")
    if settings.confusion_model != "full":
        raise ValueError(f"unknown Dawid-Skene confusion model {settings.confusion_model!r}")

    observed = annotations != ABSTAIN
    # Under the historical model only emitted labels are observations, so the client floor counts
    # non-abstaining labels. Under the explicit model every position is an emission (including
    # silence); dropping an all-abstain client here would delete exactly the evidence this mode is
    # meant to model before EM ever sees it.
    per_client = (
        np.full(annotations.shape[0], num_open, dtype=np.int64)
        if settings.explicit_abstention
        else observed.sum(axis=1)
    )
    eligible = per_client >= settings.min_client_annotations
    excluded = tuple(
        senders[j] for j in np.nonzero(~eligible)[0] if j < len(senders)
    )
    eligible_senders = tuple(senders[j] for j in np.nonzero(eligible)[0] if j < len(senders))
    if int(eligible.sum()) < settings.min_clients:
        return _failed("insufficient_clients", num_open, excluded_clients=excluded)

    annotations = annotations[eligible]
    observed = observed[eligible]
    item_counts = observed.sum(axis=0).astype(np.int64)
    if not item_counts.any():
        return _failed("no_observations", num_open, excluded_clients=excluded)
    observed_items = item_counts > 0

    num_eligible = annotations.shape[0]
    eps = settings.epsilon
    num_outputs = num_classes + int(settings.explicit_abstention)

    # Per-client observed index/label slices, computed once: the E and M steps both need them and
    # they are the only per-iteration allocation that would otherwise repeat J times per iteration.
    slices = [
        (np.nonzero(observed[j])[0], annotations[j][observed[j]].astype(np.int64))
        for j in range(num_eligible)
    ]
    onehots = [np.eye(num_classes, dtype=np.float64)[labels] for _, labels in slices]
    if settings.explicit_abstention:
        emissions = np.where(annotations == ABSTAIN, num_classes, annotations).astype(np.int64)
        output_eye = np.eye(num_outputs, dtype=np.float64)
        emission_onehots = [output_eye[emissions[j]] for j in range(num_eligible)]
        likelihood_items = np.ones(num_open, dtype=bool)
    else:
        emissions = None
        emission_onehots = None
        likelihood_items = observed_items

    # Initialization: smoothed per-sample vote proportions. This biases latent class c toward
    # output class c using the repository's own class indices, without reading any true label --
    # it does not mathematically prevent label switching, which is what the post-fit check is for.
    votes = np.zeros((num_open, num_classes), dtype=np.float64)
    for index, labels in slices:
        np.add.at(votes, (index, labels), 1.0)
    posterior = votes + settings.initialization_pseudocount
    posterior /= posterior.sum(axis=1, keepdims=True)

    prior = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
    confusion = np.tile(
        np.eye(num_classes, num_outputs, dtype=np.float64), (num_eligible, 1, 1)
    )
    previous_objective = -np.inf
    log_likelihood = float("nan")
    objective = float("nan")
    iterations = 0
    converged = False
    stop_reason = "iteration_cap"

    for iterations in range(1, settings.max_iterations + 1):
        # --- M-step: posterior-weighted counts + Dirichlet pseudocounts ---------------------
        # An item with no annotations is absent from the Dawid-Skene likelihood. Its initialized
        # posterior is therefore only a placeholder for the fixed-width output array and must not
        # be counted when estimating the class prior. Including it would let appended all-abstain
        # columns pull the fitted prior towards uniform and change otherwise identical results.
        prior_counts = posterior[likelihood_items].sum(axis=0) + settings.class_prior_pseudocount
        new_prior = prior_counts / prior_counts.sum()
        new_confusion = np.empty_like(confusion)
        for j in range(num_eligible):
            if settings.explicit_abstention:
                counts = posterior.T @ emission_onehots[j] + settings.confusion_pseudocount
            else:
                index, _ = slices[j]
                counts = posterior[index].T @ onehots[j] + settings.confusion_pseudocount
            new_confusion[j] = counts / counts.sum(axis=1, keepdims=True)

        if settings.damping < 1.0:
            new_prior = settings.damping * new_prior + (1.0 - settings.damping) * prior
            new_confusion = (
                settings.damping * new_confusion + (1.0 - settings.damping) * confusion
            )
        prior, confusion = new_prior, new_confusion

        if not (np.isfinite(prior).all() and np.isfinite(confusion).all()):
            return _failed("non_finite_parameters", num_open, iterations=iterations)
        if abs(prior.sum() - 1.0) > 1e-8 or np.abs(confusion.sum(axis=2) - 1.0).max() > 1e-8:
            return _failed("normalization_invariant_failed", num_open, iterations=iterations)

        # --- E-step: log-space posterior over hidden classes --------------------------------
        log_score = np.tile(np.log(np.maximum(prior, eps)), (num_open, 1))
        log_confusion = np.log(np.maximum(confusion, eps))
        for j in range(num_eligible):
            if settings.explicit_abstention:
                log_score += log_confusion[j][:, emissions[j]].T
            else:
                index, labels = slices[j]
                log_score[index] += log_confusion[j][:, labels].T

        row_max = log_score.max(axis=1, keepdims=True)
        exp_shifted = np.exp(log_score - row_max)
        row_sum = exp_shifted.sum(axis=1, keepdims=True)
        posterior = exp_shifted / row_sum
        if not np.isfinite(posterior).all():
            return _failed("non_finite_posterior", num_open, iterations=iterations)

        # Unobserved items contribute log(1) to the likelihood, not log(prior): they carry no
        # evidence, so counting their prior mass would let the objective drift with coverage.
        per_item_loglik = (np.log(row_sum) + row_max).ravel()
        log_likelihood = float(per_item_loglik[likelihood_items].sum())
        # Regularized objective: the M-step is a Dirichlet MAP update, so the monotone quantity is
        # the observed-data log-likelihood plus the matching log-prior terms -- not a raw one.
        objective = float(
            log_likelihood
            + settings.class_prior_pseudocount * np.log(np.maximum(prior, eps)).sum()
            + settings.confusion_pseudocount * np.log(np.maximum(confusion, eps)).sum()
        )
        if not np.isfinite(objective):
            return _failed("non_finite_objective", num_open, iterations=iterations)

        scale = max(abs(previous_objective), 1.0)
        delta = objective - previous_objective
        if np.isfinite(previous_objective):
            if delta < -settings.tolerance * scale:
                return _failed("objective_decreased", num_open, iterations=iterations)
            if iterations >= settings.min_iterations and delta <= settings.tolerance * scale:
                converged = True
                stop_reason = "tolerance"
                previous_objective = objective
                break
        previous_objective = objective

    # --- Class alignment ----------------------------------------------------------------------
    alignment_permutation: tuple[int, ...] = ()
    alignment_score = 0.0
    if settings.class_alignment:
        posterior, prior, confusion, alignment_permutation, alignment_score = _align_classes(
            posterior, prior, confusion
        )

    # --- Decision -----------------------------------------------------------------------------
    # np.argmax already returns the lowest index on a tie, matching aggregate_votes' tie rule.
    candidate = posterior.argmax(axis=1).astype(np.int64)
    max_posterior = posterior.max(axis=1)
    valid_mask = (item_counts >= settings.min_item_annotations) & (
        max_posterior >= settings.posterior_threshold
    )
    labels = np.where(valid_mask, candidate, ABSTAIN).astype(np.int64)

    # --- Permutation check (F3): is latent class c still output class c? ----------------------
    # Three gates, and it matters which one does what:
    #   majority_agreement -- the actual permutation detector. It is the only statistic anchored
    #     to something outside the fit, so it is the only one a global relabelling cannot fool.
    #   diagonal_fraction  -- coherence of latent classes against the emitted label space, scored
    #     RELATIVE to what majority achieves on the same annotations. It cannot see a global
    #     relabelling at all (EM moves to the rotated solution, which is equally self-consistent,
    #     so the number is unchanged -- see test_diagonal_fraction_alone_cannot_see_a_global
    #     _relabelling). An absolute floor here is worse than useless: it conflates specialisation
    #     with tampering, and on scenario 1 even sealed ground truth only scores ~0.14-0.25.
    #   chance floor       -- backstop for a fit with no coherent class-to-label mapping at all,
    #     checked first because the ratio degrades in step with the fit when both collapse.
    # All three are majority-anchored or majority-scored, so all three are hybrid-arm policy. They
    # set the status; whether the status blocks a broadcast is the caller's decision.
    diagonal_fraction = _diagonal_fraction(
        posterior, slices, onehots, num_classes, settings.confusion_pseudocount
    )
    observed_majority = majority_labels != ABSTAIN
    majority_posterior = np.zeros((num_open, num_classes), dtype=np.float64)
    majority_posterior[
        np.nonzero(observed_majority)[0], majority_labels[observed_majority]
    ] = 1.0
    reference_diagonal_fraction = _diagonal_fraction(
        majority_posterior, slices, onehots, num_classes, settings.confusion_pseudocount
    )
    comparable = valid_mask & (majority_labels != ABSTAIN)
    majority_agreement = (
        float((labels[comparable] == majority_labels[comparable]).mean())
        if comparable.any()
        else 0.0
    )
    diagnostics = dict(
        iterations=iterations,
        converged=converged,
        stop_reason=stop_reason,
        log_likelihood=log_likelihood,
        objective=objective,
        eligible_clients=num_eligible,
        excluded_clients=excluded,
        eligible_senders=eligible_senders,
        diagonal_fraction=diagonal_fraction,
        reference_diagonal_fraction=reference_diagonal_fraction,
        majority_agreement=majority_agreement,
        max_posterior_mean=float(max_posterior[valid_mask].mean()) if valid_mask.any() else 0.0,
        alignment_permutation=alignment_permutation,
        alignment_score=alignment_score,
        # EM ran to a finite, normalized end. Every check from here down is policy, not arithmetic,
        # so the candidate travels with the rejection and a caller with no majority path can use it.
        numerically_valid=True,
        candidate_labels=labels,
        candidate_valid_mask=valid_mask,
        confusion=confusion,
        prior=prior,
        posterior=posterior,
    )

    if not converged:
        return _failed("not_converged", num_open, **diagnostics)
    # Chance backstop, checked first: a permuted fit lands near 1/num_classes, and if the majority
    # reference is itself degenerate the ratio test below would otherwise admit anything.
    if diagonal_fraction <= 1.0 / num_classes:
        return _failed("permutation_check_chance", num_open, **diagnostics)
    if (
        diagonal_fraction
        < settings.permutation_min_diagonal_ratio * reference_diagonal_fraction
    ):
        return _failed("permutation_check_diagonal", num_open, **diagnostics)
    if majority_agreement < settings.permutation_min_majority_agreement:
        return _failed("permutation_check_agreement", num_open, **diagnostics)

    return DawidSkeneFit(
        status="ok",
        labels=labels,
        valid_mask=valid_mask,
        **diagnostics,
    )
