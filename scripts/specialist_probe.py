"""Do specialist (severely non-IID) clients reproduce the class collapse?

Scenario 1 gives each client only 2-3 of the 11 classes. A client that never saw class c does not
abstain on it -- it confidently emits some class it does know. That is a legitimate Dawid-Skene
annotator (arbitrary confusion matrix), so the question is identifiability, not assumption
violation: can EM tell two classes apart when no annotator distinguishes them?
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


def synth(known_per_client, acc=0.9, seed=0, imbalance=False):
    rng = np.random.default_rng(seed)
    if imbalance:  # mimic the real open set: a few big classes, several tiny ones
        w = np.array([6.0, 6.0, 0.4, 3.0, 0.5, 6.0])[:K]
        truth = rng.choice(K, size=N, p=w / w.sum())
    else:
        truth = rng.integers(0, K, size=N)
    ann = np.empty((J, N), dtype=np.int8)
    for j in range(J):
        known = rng.choice(K, size=known_per_client, replace=False)
        # Systematic, fixed misread for every class this client never saw.
        misread = {c: int(rng.choice(known)) for c in range(K) if c not in known}
        row = np.empty(N, dtype=np.int8)
        for c in range(K):
            sel = truth == c
            if c in known:
                ok = rng.random(sel.sum()) < acc
                row[sel] = np.where(ok, c, rng.choice(known, size=sel.sum()))
            else:
                row[sel] = misread[c]
        ann[j] = row
    return ann, truth


print(f"{'known/client':>12} {'imbal':>6} {'maj_acc':>8} {'ds_acc':>8} {'delta':>8} "
      f"{'status':<28} {'latent_used':>11} {'classes_at_0':>12}")
for imbalance in (False, True):
    for known in (6, 5, 4, 3, 2):
        ann, truth = synth(known, seed=7, imbalance=imbalance)
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
        print(f"{known:12d} {str(imbalance):>6} {m_acc:8.4f} {d_acc:8.4f} {d_acc - m_acc:8.4f} "
              f"{fit.status:<28} {used:11d} {dead:12d}")
