"""Shared supervised + distillation training loops, evaluation, and batched open-data
prediction. Independent of Flower -- protocols (M4) call these directly against local models.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ssfl.seeding import dataloader_worker_init_fn, make_generator


@dataclass
class TrainResult:
    epoch_losses: list[float] = field(default_factory=list)

    @property
    def final_loss(self) -> float:
        return self.epoch_losses[-1] if self.epoch_losses else float("nan")


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    """Deterministic DataLoader: a seeded generator drives shuffling and worker seeding
    independent of whatever else has touched global RNG state."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=make_generator(seed) if shuffle else None,
        worker_init_fn=dataloader_worker_init_fn if shuffle else None,
    )


def build_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """Fresh Adam every call -- SSFL_IMPLEMENTATION_PLAN.md M3: 'fresh optimizer per Flower task
    while model weights persist.'"""
    return torch.optim.Adam(model.parameters(), lr=lr)


def _amp_context(device: torch.device, enabled: bool):
    # ponytail: mixed precision is a CUDA-only perf profile, off by default (paper mode never
    # sets it); on CPU/MPS this silently no-ops rather than erroring on unsupported autocast.
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return contextlib.nullcontext()


def _run_epochs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    loss_fn,
    grad_clip_norm: float | None,
    use_amp: bool,
) -> TrainResult:
    model.to(device)
    model.train()
    optimizer = build_optimizer(model, lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    result = TrainResult()
    for _ in range(epochs):
        total_loss, total_count = 0.0, 0
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            with _amp_context(device, use_amp):
                loss = loss_fn(model(x), target)
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x.shape[0]
            total_count += x.shape[0]
        result.epoch_losses.append(total_loss / total_count)
    return result


def train_supervised(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    grad_clip_norm: float | None = None,
    use_amp: bool = False,
) -> TrainResult:
    """Hard-label cross-entropy training (loader yields ``(x, class_index)``)."""
    return _run_epochs(
        model, loader, device, epochs, lr, nn.CrossEntropyLoss(), grad_clip_norm, use_amp
    )


def _teacher_distribution_loss(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def train_distillation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    grad_clip_norm: float | None = None,
    use_amp: bool = False,
) -> TrainResult:
    """Soft-target teacher-distribution loss (loader yields ``(x, prob_distribution)``) -- used
    for FD/DS-FL distillation and SSFL's soft-label ablation."""
    return _run_epochs(
        model, loader, device, epochs, lr, _teacher_distribution_loss, grad_clip_norm, use_amp
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.shape[0]
    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Batched softmax probabilities over an unlabeled loader (open data), in loader order."""
    model.to(device)
    model.eval()
    chunks = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        chunks.append(torch.softmax(model(x.to(device)), dim=-1).cpu())
    return torch.cat(chunks, dim=0)
