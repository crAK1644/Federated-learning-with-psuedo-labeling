"""Does the Dawid-Skene objective prefer the collapsed fit over the truth?

This separates two very different diagnoses that the accuracy numbers alone cannot:

  truth scores HIGHER  -> EM converged to a worse local optimum. An optimization/initialization
                          problem, and fixable (better init, restarts, annealing).
  fit   scores HIGHER  -> the model genuinely ranks the wrong labelling above the right one.
                          A model mismatch, and no amount of optimization saves it.

Scores the sealed ground truth through the exact same objective the estimator maximizes, by
seeding the posterior at truth and running one M-step + E-step.
"""

import glob
import sys

import numpy as np

sys.path.insert(0, "scripts")
from pathlib import Path

from ds_stage0 import sealed_open_labels

from ssfl.protocols.dawid_skene import ABSTAIN, DawidSkeneSettings, fit_dawid_skene

K = 11
S = DawidSkeneSettings()


def objective_at(posterior, annotations, num_classes=K, settings=S):
    """The estimator's own objective: observed-data log-likelihood + matching Dirichlet log-priors.

    Mirrors fit_dawid_skene's M-step then E-step exactly, so the numbers are comparable.
    """
    observed = annotations != ABSTAIN
    slices = [
        (np.nonzero(observed[j])[0], annotations[j][observed[j]].astype(np.int64))
        for j in range(annotations.shape[0])
    ]
    onehots = [np.eye(num_classes)[lab] for _, lab in slices]
    item_counts = observed.sum(axis=0)
    eps = settings.epsilon

    prior_counts = posterior.sum(axis=0) + settings.class_prior_pseudocount
    prior = prior_counts / prior_counts.sum()
    confusion = np.empty((annotations.shape[0], num_classes, num_classes))
    for j in range(annotations.shape[0]):
        index, _ = slices[j]
        counts = posterior[index].T @ onehots[j] + settings.confusion_pseudocount
        confusion[j] = counts / counts.sum(axis=1, keepdims=True)

    log_score = np.tile(np.log(np.maximum(prior, eps)), (annotations.shape[1], 1))
    log_confusion = np.log(np.maximum(confusion, eps))
    for j in range(annotations.shape[0]):
        index, labels = slices[j]
        log_score[index] += log_confusion[j][:, labels].T
    row_max = log_score.max(axis=1, keepdims=True)
    row_sum = np.exp(log_score - row_max).sum(axis=1, keepdims=True)
    loglik = float((np.log(row_sum) + row_max).ravel()[item_counts > 0].sum())
    return loglik + (
        settings.class_prior_pseudocount * np.log(np.maximum(prior, eps)).sum()
        + settings.confusion_pseudocount * np.log(np.maximum(confusion, eps)).sum()
    )


truth = sealed_open_labels(Path("artifacts/data"))
for path in sorted(
    glob.glob("artifacts/runs/ssfl-s1-ds50_shadow-*/attempts/*/aggregation_audit/*.npz"),
    key=lambda p: int(p.split("round_")[1].split(".")[0]),
):
    rnd = int(path.split("round_")[1].split(".")[0])
    z = np.load(path)
    A = z["annotations"]
    maj = z["majority_labels"].astype(np.int64)

    fit = fit_dawid_skene(A, num_classes=K, majority_labels=maj, settings=S)

    onehot_truth = np.eye(K)[truth]
    onehot_maj = np.eye(K)[np.where(maj == ABSTAIN, 0, maj)]
    onehot_maj[maj == ABSTAIN] = 1.0 / K

    obj_truth = objective_at(onehot_truth, A)
    obj_maj = objective_at(onehot_maj, A)
    if fit.labels is not None and fit.valid_mask is not None and fit.valid_mask.any():
        onehot_fit = np.eye(K)[np.where(fit.labels == ABSTAIN, 0, fit.labels)]
        onehot_fit[fit.labels == ABSTAIN] = 1.0 / K
        obj_fit = objective_at(onehot_fit, A)
        acc_fit = (fit.labels == truth).mean()
    else:
        obj_fit, acc_fit = float("nan"), float("nan")

    print(
        f"round {rnd:2d}  status={fit.status:<28} "
        f"acc: truth=1.0000 maj={(maj == truth).mean():.4f} ds={acc_fit:.4f}"
    )
    print(
        f"          objective: truth={obj_truth:14.1f}  majority={obj_maj:14.1f}  "
        f"ds_fit={obj_fit:14.1f}   ds_beats_truth={obj_fit > obj_truth}"
    )
