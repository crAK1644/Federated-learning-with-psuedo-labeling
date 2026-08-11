"""Stage 0 offline gate: does Dawid-Skene beat majority on annotation matrices already recorded?

This is the go/no-go the approval brief puts before any live active run, and the place where
pseudocounts / iteration cap / warm-up are screened -- hours instead of GPU-days, and no locked
live run spent on hyperparameter search.

Input is the ``annotations`` array written by ``SSFLStrategy`` when
``dawid_skene_save_annotations: true`` (see configs/dawid_skene_shadow.yaml). Sealed open-set
labels are read from the existing preparation audit table for scoring only: they never enter the
estimator, the settings, or any fallback decision, and no label file is written back into
``artifacts/data`` (that would change the dataset manifest hash and therefore every run_id).

    uv run python scripts/ds_stage0.py <run_dir> [--grid] [--classes 2,4,6]
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ssfl.protocols.dawid_skene import ABSTAIN, DawidSkeneSettings, fit_dawid_skene

sys.path.insert(0, str(Path(__file__).parent))
from ds_headroom import sealed_open_labels  # noqa: E402

NUM_CLASSES = 11

# Screened offline, never in a live run. Keep this small: each entry is a full EM fit per round.
GRID = [
    ("baseline", {}),
    ("conf_pseudocount_0.01", {"confusion_pseudocount": 0.01}),
    ("conf_pseudocount_1.0", {"confusion_pseudocount": 1.0}),
    ("flat_class_prior", {"class_prior_pseudocount": 10.0}),
    ("iter_cap_500", {"max_iterations": 500}),
    ("tolerance_1e-8", {"tolerance": 1e-8}),
]


def _majority(audit) -> tuple[np.ndarray, np.ndarray]:
    """Majority labels for the round, whichever key this run recorded them under."""
    if "majority_labels" in audit:
        return audit["majority_labels"].astype(np.int64), audit["majority_valid_mask"].astype(bool)
    return audit["global_labels"].astype(np.int64), audit["valid_mask"].astype(bool)


def _accuracy(labels: np.ndarray, mask: np.ndarray, sealed: np.ndarray) -> float:
    return float((labels[mask] == sealed[mask]).mean()) if mask.any() else 0.0


def evaluate_round(path: str, sealed: np.ndarray, settings: DawidSkeneSettings) -> dict:
    audit = np.load(path)
    annotations = audit["annotations"]
    majority_labels, majority_mask = _majority(audit)

    started = time.perf_counter()
    fit = fit_dawid_skene(
        annotations,
        num_classes=NUM_CLASSES,
        majority_labels=majority_labels,
        settings=settings,
    )
    seconds = time.perf_counter() - started

    # A failed fit falls back to majority in production, so score it that way here too -- the
    # question is what the arm actually delivers, not what the estimator would have delivered.
    labels, mask = (fit.labels, fit.valid_mask) if fit.ok else (majority_labels, majority_mask)
    return {
        "round": int(re.search(r"round_(\d+)", path).group(1)),
        "clients": int(annotations.shape[0]),
        "majority_acc": _accuracy(majority_labels, majority_mask, sealed),
        "ds_acc": _accuracy(labels, mask, sealed),
        "status": fit.status,
        "iterations": fit.iterations,
        "seconds": seconds,
        "diag_fraction": fit.diagonal_fraction,
        "majority_agreement": fit.majority_agreement,
        "ds_valid_rate": float(mask.mean()),
        "annotations_per_client": float((annotations != ABSTAIN).sum(axis=1).mean()),
        "min_annotations_per_client": int((annotations != ABSTAIN).sum(axis=1).min()),
    }


def per_class_recall(path: str, sealed: np.ndarray, settings: DawidSkeneSettings) -> pd.DataFrame:
    audit = np.load(path)
    majority_labels, majority_mask = _majority(audit)
    fit = fit_dawid_skene(
        audit["annotations"],
        num_classes=NUM_CLASSES,
        majority_labels=majority_labels,
        settings=settings,
    )
    labels, mask = (fit.labels, fit.valid_mask) if fit.ok else (majority_labels, majority_mask)
    rows = []
    for class_id in range(NUM_CLASSES):
        selected = sealed == class_id
        rows.append(
            {
                "class": class_id,
                "n": int(selected.sum()),
                "majority_recall": _accuracy(
                    majority_labels, majority_mask & selected, sealed
                ),
                "ds_recall": _accuracy(labels, mask & selected, sealed),
            }
        )
    frame = pd.DataFrame(rows)
    frame["delta"] = frame.ds_recall - frame.majority_recall
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data", default=Path("artifacts/data"), type=Path)
    parser.add_argument("--grid", action="store_true", help="screen the hyperparameter grid")
    args = parser.parse_args()

    sealed = sealed_open_labels(args.data)
    audits = sorted(
        glob.glob(str(args.run_dir / "attempts" / "*" / "aggregation_audit" / "*.npz")),
        key=lambda p: int(re.search(r"round_(\d+)", p).group(1)),
    )
    audits = [p for p in audits if "annotations" in np.load(p)]
    if not audits:
        raise SystemExit(
            f"no audit npz with an 'annotations' array under {args.run_dir} -- rerun with "
            "dawid_skene_save_annotations: true"
        )

    pd.set_option("display.width", 200)
    variants = GRID if args.grid else GRID[:1]
    summaries = []
    for name, overrides in variants:
        settings = DawidSkeneSettings(**overrides)
        table = pd.DataFrame([evaluate_round(p, sealed, settings) for p in audits])
        print(f"\n=== {name} ===")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        summaries.append(
            {
                "variant": name,
                "majority_acc": table.majority_acc.mean(),
                "ds_acc": table.ds_acc.mean(),
                "delta": table.ds_acc.mean() - table.majority_acc.mean(),
                "fallback_rounds": int((table.status != "ok").sum()),
                "mean_seconds": table.seconds.mean(),
                "mean_iterations": table.iterations.mean(),
            }
        )

    print("\n=== gate summary (mean over recorded rounds) ===")
    print(pd.DataFrame(summaries).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    final_round = re.search(r"round_(\d+)", audits[-1]).group(1)
    print(f"\n=== per-class recall, round {final_round} ===")
    print(
        per_class_recall(audits[-1], sealed, DawidSkeneSettings()).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    print(
        "\nGo/no-go is a judgement on this table, not an automatic threshold: the brief's stated "
        "target is recovery of the collapsed classes, not a small mean-accuracy gain."
    )


if __name__ == "__main__":
    main()
