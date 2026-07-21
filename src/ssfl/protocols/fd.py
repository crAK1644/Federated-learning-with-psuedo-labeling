"""FD baseline: per-class averaged client soft predictions with leave-self-out teacher targets.

Loss weighting (ground-truth CE vs teacher-CE) is an equal 1:1 sum -- the paper does not specify a
weighting scheme; see REPRODUCIBILITY.md assumption #14.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from ssfl.models import NUM_CLASSES, SSFLModel
from ssfl.training import TrainResult, build_optimizer, make_loader, predict_probs, teacher_distribution_loss


@dataclass(frozen=True)
class ClassLogitUpload:
    client_id: str
    class_probs: np.ndarray  # (NUM_CLASSES, NUM_CLASSES) float32; row c = mean predicted prob
    #                          vector over this client's own private examples of class c (zero
    #                          row if the client has none of that class).
    class_present: np.ndarray  # (NUM_CLASSES,) int32, 1 if the client has >=1 example of class c


def client_class_logits_step(
    client_id: str, classifier: SSFLModel, private_dataset, device: torch.device, batch_size: int, seed: int
) -> ClassLogitUpload:
    loader = make_loader(private_dataset, batch_size, shuffle=False, seed=seed)
    probs = predict_probs(classifier, loader, device).numpy()
    labels = private_dataset.y.numpy()

    class_probs = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    class_present = np.zeros(NUM_CLASSES, dtype=np.int32)
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.any():
            class_probs[c] = probs[mask].mean(axis=0)
            class_present[c] = 1
    return ClassLogitUpload(client_id, class_probs, class_present)


@dataclass(frozen=True)
class FDAggregation:
    global_sum: np.ndarray  # (NUM_CLASSES, NUM_CLASSES) float32
    contributor_counts: np.ndarray  # (NUM_CLASSES,) int32; 0 == class missing federation-wide


def aggregate_class_logits(uploads: list[ClassLogitUpload]) -> FDAggregation:
    """Idempotent: a duplicate ``client_id`` is deduped before summing."""
    seen: set[str] = set()
    deduped: list[ClassLogitUpload] = []
    for u in uploads:
        if u.client_id in seen:
            continue
        seen.add(u.client_id)
        deduped.append(u)

    global_sum = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    contributor_counts = np.zeros(NUM_CLASSES, dtype=np.int32)
    for u in deduped:
        for c in range(NUM_CLASSES):
            if u.class_present[c]:
                global_sum[c] += u.class_probs[c]
                contributor_counts[c] += 1
    return FDAggregation(global_sum, contributor_counts)


def leave_self_out_targets(
    aggregation: FDAggregation, upload: ClassLogitUpload
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class teacher target for one client: the average of every *other* contributing
    client's class-probability vector. A class has no valid target (``valid[c] = False``) if this
    client is the only contributor, or if no client contributed that class at all -- the "missing
    class" case the caller must mask out before training."""
    targets = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    valid = np.zeros(NUM_CLASSES, dtype=bool)
    for c in range(NUM_CLASSES):
        count = int(aggregation.contributor_counts[c])
        if upload.class_present[c]:
            other_count = count - 1
            if other_count > 0:
                targets[c] = (aggregation.global_sum[c] - upload.class_probs[c]) / other_count
                valid[c] = True
        elif count > 0:
            targets[c] = aggregation.global_sum[c] / count
            valid[c] = True
    return targets, valid


def client_distillation_step(
    classifier: SSFLModel,
    private_dataset,
    targets: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> TrainResult:
    """Equal-weight ground-truth CE + teacher-CE, restricted to private examples whose class has
    a valid leave-self-out target (assumption #14: 1:1 weighting)."""
    labels = private_dataset.y.numpy()
    keep = valid[labels]
    result = TrainResult()
    if not keep.any():
        return result

    keep_t = torch.from_numpy(keep)
    x = private_dataset.x[keep_t]
    y = private_dataset.y[keep_t]
    teacher = torch.from_numpy(targets[labels[keep]]).float()

    loader = make_loader(TensorDataset(x, y, teacher), batch_size, shuffle=True, seed=seed)
    classifier.to(device)
    classifier.train()
    optimizer = build_optimizer(classifier, lr)
    for _ in range(epochs):
        total_loss, total_count = 0.0, 0
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            optimizer.zero_grad()
            logits = classifier(xb)
            loss = F.cross_entropy(logits, yb) + teacher_distribution_loss(logits, tb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.shape[0]
            total_count += xb.shape[0]
        result.epoch_losses.append(total_loss / total_count)
    return result
