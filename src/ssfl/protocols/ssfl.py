"""SSFL protocol: two-phase per-round exchange (proposal + distillation).

Pure functions/dataclasses over models, tensors, and datasets -- no Flower dependency, per the M4
gate. Flower ``ClientApp``/``ServerApp`` wiring (M5) calls straight into this module. Callers keep
holding the same classifier/discriminator ``nn.Module`` instances across rounds (in-process here,
via ``Context.state`` once wired to Flower) -- that is what gives "persistent private client
models" for free at this layer, without this module needing to know how state is stored.

Privacy boundary (REPRODUCIBILITY.md / DATA_CARD.md): :class:`ProposalResult` -- the only thing a
client returns from the proposal phase -- carries pseudo-labels and scalar metrics only. It never
contains model parameters, gradients, private features, or optimizer state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from ssfl.config import ThresholdPolicy
from ssfl.models import SSFLModel
from ssfl.protocols.message import Envelope
from ssfl.training import evaluate, make_loader, predict_probs, train_supervised

ABSTAIN = -1


@dataclass(frozen=True)
class ProposalResult:
    client_id: str
    pseudo_labels: np.ndarray  # int64, ABSTAIN(-1) or class index, len == num_open
    confidences: np.ndarray  # float32, len == num_open (classifier max-prob; audit-only)
    threshold: float
    classifier_loss: float
    discriminator_loss: float


def compute_threshold(confidences: np.ndarray, policy: ThresholdPolicy) -> float:
    if policy == ThresholdPolicy.median:
        return float(np.median(confidences))
    value = policy.fixed_value
    assert value is not None, f"non-median policy {policy} must define a fixed_value"
    return value


def client_proposal_step(
    client_id: str,
    classifier: SSFLModel,
    discriminator: SSFLModel,
    private_dataset,
    open_dataset,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    threshold_policy: ThresholdPolicy,
    seed: int,
) -> ProposalResult:
    """Phase A: train classifier on private data, score the open set, train the discriminator on
    private(familiar)/low-confidence-open(unfamiliar), then hard-filter open predictions."""
    private_loader = make_loader(private_dataset, batch_size, shuffle=True, seed=seed)
    classifier_result = train_supervised(classifier, private_loader, device, epochs, lr)

    open_loader = make_loader(open_dataset, batch_size, shuffle=False, seed=seed)
    open_probs = predict_probs(classifier, open_loader, device)
    confidences = open_probs.max(dim=1).values.numpy()
    threshold = compute_threshold(confidences, threshold_policy)

    # Discriminator targets: private examples are "familiar" (class 0), low-confidence open
    # examples are "unfamiliar" (class 1) -- the paper's discriminator dataset construction.
    unfamiliar_mask = confidences < threshold
    unfamiliar_x = open_dataset.x[torch.from_numpy(unfamiliar_mask)]
    disc_x = torch.cat([private_dataset.x, unfamiliar_x], dim=0)
    disc_y = torch.cat(
        [
            torch.zeros(private_dataset.x.shape[0], dtype=torch.long),
            torch.ones(unfamiliar_x.shape[0], dtype=torch.long),
        ]
    )
    disc_loader = make_loader(TensorDataset(disc_x, disc_y), batch_size, shuffle=True, seed=seed)
    discriminator_result = train_supervised(discriminator, disc_loader, device, epochs, lr)

    familiar_probs = predict_probs(discriminator, open_loader, device)
    familiar_mask = familiar_probs.argmax(dim=1).numpy() == 0
    class_predictions = open_probs.argmax(dim=1).numpy()
    pseudo_labels = np.where(familiar_mask, class_predictions, ABSTAIN).astype(np.int64)

    return ProposalResult(
        client_id=client_id,
        pseudo_labels=pseudo_labels,
        confidences=confidences.astype(np.float32),
        threshold=threshold,
        classifier_loss=classifier_result.final_loss,
        discriminator_loss=discriminator_result.final_loss,
    )


@dataclass(frozen=True)
class AggregationResult:
    global_labels: np.ndarray  # int64, len num_open, ABSTAIN(-1) where invalid
    valid_mask: np.ndarray  # bool, len num_open
    votes_per_class: np.ndarray  # (num_open, num_classes) int64, for audit
    participating_counts: np.ndarray  # (num_open,) int64, non-abstaining voters per index
    tie_count: int
    all_abstain_count: int
    rejected: tuple[tuple[str, str], ...]  # (sender_id, reason) for entries dropped pre-vote


def aggregate_votes(
    proposals: list[tuple[Envelope, ProposalResult]], num_open: int, num_classes: int
) -> AggregationResult:
    """Per-index majority vote; ties -> lowest class index; all-abstain -> ABSTAIN + invalid.

    Idempotent by construction: a duplicated envelope for a sender already counted is dropped
    (into ``rejected``) rather than counted twice, so re-aggregating a batch that accidentally
    contains a retried message produces the same result as aggregating it once.
    """
    seen_senders: set[str] = set()
    votes = np.zeros((num_open, num_classes), dtype=np.int64)
    rejected: list[tuple[str, str]] = []
    for envelope, result in proposals:
        if envelope.sender_id in seen_senders:
            rejected.append((envelope.sender_id, "duplicate sender in aggregation batch"))
            continue
        seen_senders.add(envelope.sender_id)
        labels = result.pseudo_labels
        idx = np.nonzero(labels != ABSTAIN)[0]
        votes[idx, labels[idx]] += 1

    global_labels = np.full(num_open, ABSTAIN, dtype=np.int64)
    valid_mask = np.zeros(num_open, dtype=bool)
    tie_count = 0
    all_abstain_count = 0
    for i in range(num_open):
        row = votes[i]
        total = int(row.sum())
        if total == 0:
            all_abstain_count += 1
            continue
        max_votes = row.max()
        winners = np.nonzero(row == max_votes)[0]
        if len(winners) > 1:
            tie_count += 1
        global_labels[i] = int(winners.min())
        valid_mask[i] = True

    return AggregationResult(
        global_labels=global_labels,
        valid_mask=valid_mask,
        votes_per_class=votes,
        participating_counts=votes.sum(axis=1),
        tie_count=tie_count,
        all_abstain_count=all_abstain_count,
        rejected=tuple(rejected),
    )


def _valid_open_dataset(open_dataset, aggregation: AggregationResult) -> TensorDataset:
    valid_idx = np.nonzero(aggregation.valid_mask)[0]
    x = open_dataset.x[torch.from_numpy(valid_idx)]
    y = torch.from_numpy(aggregation.global_labels[valid_idx]).long()
    return TensorDataset(x, y)


def client_distillation_step(
    classifier: SSFLModel,
    open_dataset,
    aggregation: AggregationResult,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
):
    """Phase B (client): hard-label CE on the valid subset of the global open labels. Returns
    metrics only -- no model parameters cross this function's boundary back to a message."""
    dataset = _valid_open_dataset(open_dataset, aggregation)
    loader = make_loader(dataset, batch_size, shuffle=True, seed=seed)
    return train_supervised(classifier, loader, device, epochs, lr)


def server_distillation_step(
    server_classifier: SSFLModel,
    open_dataset,
    aggregation: AggregationResult,
    test_dataset,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
):
    """Phase B (server): train the server's own persistent classifier on the same valid open
    labels, then evaluate it on the full test set."""
    dataset = _valid_open_dataset(open_dataset, aggregation)
    loader = make_loader(dataset, batch_size, shuffle=True, seed=seed)
    train_result = train_supervised(server_classifier, loader, device, epochs, lr)

    test_loader = make_loader(test_dataset, batch_size, shuffle=False, seed=seed)
    eval_metrics = evaluate(server_classifier, test_loader, device)
    return train_result, eval_metrics
