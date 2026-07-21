"""Centralized classification metrics (accuracy, macro P/R/F1, confusion matrix) plus a
per-round ledger written to ``metrics.parquet``.

Pseudo-label accuracy against true open-set labels is intentionally NOT computed here -- that
would require reading labels the SSFL protocol must never expose to training (see
REPRODUCIBILITY.md). Any such audit belongs in the M8 reporting layer, which reads the sealed
``audit/source_rows.parquet`` separately from run artifacts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion_matrix: np.ndarray


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> ClassificationMetrics:
    labels = list(range(num_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    return ClassificationMetrics(
        accuracy=accuracy,
        macro_precision=float(precision),
        macro_recall=float(recall),
        macro_f1=float(f1),
        confusion_matrix=cm,
    )


@dataclass
class MetricsLedger:
    """One row per centralized-eval call. Confusion matrices aren't scalar so they're kept out of
    the parquet table and flushed to a sidecar ``confusion_matrices.npz`` keyed by round."""

    rows: list[dict] = field(default_factory=list)
    _confusion_matrices: dict[int, np.ndarray] = field(default_factory=dict)

    def record(
        self,
        *,
        algorithm: str,
        scenario: int,
        round: int,
        loss: float,
        train_loss: float | None = None,
        metrics: ClassificationMetrics | None = None,
        valid_rate: float | None = None,
    ) -> None:
        row = {
            "algorithm": algorithm,
            "scenario": scenario,
            "round": round,
            "loss": loss,
            "train_loss": train_loss,
            "valid_rate": valid_rate,
            "wall_clock": time.time(),
        }
        if metrics is not None:
            row.update(
                {
                    "accuracy": metrics.accuracy,
                    "macro_precision": metrics.macro_precision,
                    "macro_recall": metrics.macro_recall,
                    "macro_f1": metrics.macro_f1,
                }
            )
            self._confusion_matrices[round] = metrics.confusion_matrix
        self.rows.append(row)

    def write(self, run_dir: Path) -> None:
        import pandas as pd

        pd.DataFrame(self.rows).to_parquet(run_dir / "metrics.parquet", index=False)
        if self._confusion_matrices:
            np.savez(
                run_dir / "confusion_matrices.npz",
                **{f"round_{r}": cm for r, cm in self._confusion_matrices.items()},
            )


if __name__ == "__main__":
    y_true = np.array([0, 1, 1, 2, 0])
    y_pred = np.array([0, 1, 0, 2, 0])
    m = compute_classification_metrics(y_true, y_pred, num_classes=3)
    assert 0.0 <= m.accuracy <= 1.0 and abs(m.accuracy - 4 / 5) < 1e-9
    assert m.confusion_matrix.shape == (3, 3)
    assert m.confusion_matrix.sum() == len(y_true)

    ledger = MetricsLedger()
    ledger.record(algorithm="ssfl", scenario=1, round=1, loss=0.5, metrics=m)
    assert ledger.rows[0]["accuracy"] == m.accuracy
    assert 1 in ledger._confusion_matrices
    print("metrics.py self-check OK")
