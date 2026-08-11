"""Shared blind spots: classes every client misreads the same way.

The real baseline has class 4 at 0.000 recall and class 2 at 0.100 -- no client ever emits them
correctly, because all 27 are distilled from the same global model each round. Dawid-Skene's model
has no term for "all annotators share a systematic blind spot": the latent class for a never-emitted
class has no distinguishing evidence, so EM is free to reassign it. Question: does that alone
reproduce the live collapse (whole classes to 0.000, DS well below majority)?
"""

import numpy as np
from ssfl.protocols.dawid_skene import ABSTAIN, DawidSkeneSettings, fit_dawid_skene

K, N, J = 6, 3000, 15


def majority(ann):
    out = np.full(ann.shape[1], ABSTAIN, dtype=np.int64)
    for i in range(ann.shape[1]):
        col = ann[:, i][ann[:, i] != ABSTAIN]
        if len(col):
            out[i] = int(np.bincount(col, minlength=K).argmax())
    return out


def synth(num_blind, acc=0.85, seed=0):
    """`num_blind` classes are misread by EVERY client, each into a fixed other class."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, K, size=N)
    blind = {c: (c + 1) % K for c in range(K - num_blind, K)}  # shared, systematic
    ann = np.empty((J, N), dtype=np.int8)
    for j in range(J):
        row = np.empty(N, dtype=np.int8)
        for c in range(K):
            sel = truth == c
            if c in blind:
                # Every client sends it to the same wrong place, with a little private noise.
                ok = rng.random(sel.sum()) < acc
                row[sel] = np.where(ok, blind[c], rng.integers(0, K, size=sel.sum()))
            else:
                ok = rng.random(sel.sum()) < acc
                row[sel] = np.where(ok, c, rng.integers(0, K, size=sel.sum()))
        ann[j] = row
    return ann, truth


print(f"{'blind_classes':>13} {'maj_acc':>8} {'ds_acc':>8} {'delta':>8} {'status':<28} "
      f"{'latent_used':>11} {'classes_at_0':>12} {'maj_agree':>10}")
for num_blind in (0, 1, 2, 3):
    ann, truth = synth(num_blind, seed=3)
    maj = majority(ann)
    fit = fit_dawid_skene(ann, num_classes=K, majority_labels=maj,
                          settings=DawidSkeneSettings(min_clients=2))
    m_acc = float((maj == truth).mean())
    if fit.labels is not None and fit.valid_mask is not None and fit.valid_mask.any():
        d_acc = float((fit.labels == truth).mean())
        used = len(np.unique(fit.labels[fit.valid_mask]))
        dead = sum((fit.labels[truth == c] == c).mean() < 0.01 for c in range(K))
    else:
        d_acc, used, dead = float("nan"), 0, -1
    agree = fit.majority_agreement if fit.majority_agreement is not None else float("nan")
    print(f"{num_blind:13d} {m_acc:8.4f} {d_acc:8.4f} {d_acc - m_acc:8.4f} {fit.status:<28} "
          f"{used:11d} {dead:12d} {agree:10.4f}")
