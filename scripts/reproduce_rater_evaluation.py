"""Reproduce the qualitative evaluation in the rater Dawid-Skene presentation.

The presentation's evaluation section (from about 9 minutes) simulates five raters over four
classes. Informative raters are correct with probability 0.7 and spread mistakes uniformly over
the other classes. "Spammers" instead emit class 3 with probability 0.7. It reports two effects:

1. with five informative raters, majority vote slightly beats full Dawid-Skene because estimating
   every confusion-matrix entry adds variance; and
2. with systematic spammers, Dawid-Skene wins once enough informative raters identify the signal.

This script tests those effects with this repository's MAP-EM estimator. It is a behavioural
reproduction, not a claim of number-for-number parity with ``rater``: that package uses Bayesian
Stan inference (MCMC or optimisation) and different default priors.

    uv run python scripts/reproduce_rater_evaluation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ssfl.protocols.dawid_skene import DawidSkeneSettings, fit_dawid_skene


def _simulate(
    rng: np.random.Generator,
    *,
    num_items: int,
    num_classes: int,
    num_raters: int,
    informative_raters: int,
    accuracy: float,
    spam_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    truth = rng.integers(num_classes, size=num_items)
    annotations = np.empty((num_raters, num_items), dtype=np.int8)
    for rater in range(num_raters):
        if rater < informative_raters:
            preferred = truth
        else:
            preferred = np.full(num_items, spam_label, dtype=np.int64)
        other = rng.integers(num_classes - 1, size=num_items)
        other += other >= preferred
        annotations[rater] = np.where(rng.random(num_items) < accuracy, preferred, other)
    return truth, annotations


def _majority_random_tie(
    annotations: np.ndarray, num_classes: int, rng: np.random.Generator
) -> np.ndarray:
    votes = np.eye(num_classes, dtype=np.int16)[annotations].sum(axis=0)
    labels = np.empty(annotations.shape[1], dtype=np.int64)
    for item, row in enumerate(votes):
        winners = np.flatnonzero(row == row.max())
        labels[item] = int(rng.choice(winners))
    return labels


def run_evaluation(
    *,
    num_items: int = 100,
    replicates: int = 300,
    seed: int = 20_260_817,
    num_classes: int = 4,
    num_raters: int = 5,
    accuracy: float = 0.7,
) -> list[dict[str, float | int]]:
    if num_classes < 2 or not 0 <= accuracy <= 1:
        raise ValueError("num_classes must be >= 2 and accuracy must be in [0, 1]")
    if num_raters < 3 or num_items < 1 or replicates < 1:
        raise ValueError("num_raters must be >= 3; num_items and replicates must be >= 1")

    # The reference evaluation is about the estimator, not the production safety policy. Disable
    # the two project-specific post-fit gates while retaining numerical and convergence checks.
    settings = DawidSkeneSettings(
        max_iterations=500,
        min_clients=3,
        permutation_min_diagonal_ratio=0.0,
        permutation_min_majority_agreement=0.0,
    )
    rows: list[dict[str, float | int]] = []
    root = np.random.SeedSequence(seed)
    conditions = root.spawn(num_raters + 1)
    for informative_raters, condition_seed in zip(
        range(num_raters, -1, -1), conditions, strict=True
    ):
        majority_accuracy: list[float] = []
        ds_accuracy: list[float] = []
        fallback_accuracy: list[float] = []
        iterations: list[int] = []
        passed = 0
        for replicate_seed in condition_seed.spawn(replicates):
            data_seed, tie_seed = replicate_seed.spawn(2)
            truth, annotations = _simulate(
                np.random.default_rng(data_seed),
                num_items=num_items,
                num_classes=num_classes,
                num_raters=num_raters,
                informative_raters=informative_raters,
                accuracy=accuracy,
                spam_label=min(2, num_classes - 1),
            )
            majority = _majority_random_tie(
                annotations, num_classes, np.random.default_rng(tie_seed)
            )
            majority_score = float((majority == truth).mean())
            fit = fit_dawid_skene(annotations, num_classes, majority, settings)
            majority_accuracy.append(majority_score)
            if fit.ok:
                score = float((fit.labels == truth).mean())
                ds_accuracy.append(score)
                fallback_accuracy.append(score)
                iterations.append(fit.iterations)
                passed += 1
            else:
                fallback_accuracy.append(majority_score)

        rows.append(
            {
                "informative_raters": informative_raters,
                "spammers": num_raters - informative_raters,
                "majority_accuracy": float(np.mean(majority_accuracy)),
                "ds_accuracy_when_fit_passed": float(np.mean(ds_accuracy)) if ds_accuracy else 0.0,
                "ds_with_fallback_accuracy": float(np.mean(fallback_accuracy)),
                "fit_pass_rate": passed / replicates,
                "median_iterations": float(np.median(iterations)) if iterations else 0.0,
                "replicates": replicates,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=100)
    parser.add_argument("--replicates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20_260_817)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    rows = run_evaluation(num_items=args.items, replicates=args.replicates, seed=args.seed)
    header = (
        "informative spammers majority  DS(fit) DS/fallback pass_rate median_iterations"
    )
    print(header)
    for row in rows:
        print(
            f"{row['informative_raters']:>11} {row['spammers']:>8} "
            f"{row['majority_accuracy']:.4f}   {row['ds_accuracy_when_fit_passed']:.4f}   "
            f"{row['ds_with_fallback_accuracy']:.4f}     {row['fit_pass_rate']:.3f} "
            f"{row['median_iterations']:.1f}"
        )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
