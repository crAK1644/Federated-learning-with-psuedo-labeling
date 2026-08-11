"""Disjoint annotator coverage: the one ingredient the correlation/specialist/blindspot probes lacked.

Annotators here are ideal by every Dawid-Skene criterion -- conditionally independent, highly
accurate, no shared blind spot, no correlation. The single manipulation is WHICH classes each
client votes on. As coverage narrows, some class pairs end up with no annotator in common, and a
pair no annotator ever labels differently is non-identifiable: one latent class explains both, and
merging them raises the likelihood. Majority vote is unaffected, because it never has to decide
whether two classes are the same class.
"""

import numpy as np

from ssfl.protocols.dawid_skene import ABSTAIN, DawidSkeneSettings, fit_dawid_skene

K, N, J = 11, 8900, 27


def majority(ann):
    out = np.full(ann.shape[1], ABSTAIN, dtype=np.int64)
    for i in range(ann.shape[1]):
        col = ann[:, i][ann[:, i] != ABSTAIN]
        if len(col):
            out[i] = int(np.bincount(col, minlength=K).argmax())
    return out


def synth(classes_per_client, acc=0.95, seed=0):
    """Each client votes only on `classes_per_client` classes and abstains on the rest."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, K, size=N)
    ann = np.full((J, N), ABSTAIN, dtype=np.int8)
    covers = np.zeros((J, K), dtype=bool)
    for j in range(J):
        covers[j, rng.choice(K, size=classes_per_client, replace=False)] = True
        vote = covers[j][truth]
        ok = rng.random(N) < acc
        wrong = (truth + rng.integers(1, K, size=N)) % K
        ann[j] = np.where(vote, np.where(ok, truth, wrong), ABSTAIN)
    # A pair is unseparable when no client votes on both classes.
    pairs = [(a, b) for a in range(K) for b in range(a + 1, K)
             if not (covers[:, a] & covers[:, b]).any()]
    return ann, truth, len(pairs)


print(f"{'classes/client':>14} {'blind_pairs':>12} {'abstain':>8} {'maj_acc':>8} {'ds_acc':>8} "
      f"{'delta':>8} {'status':<18} {'latent_used':>11} {'maj_agree':>10}")
for cpc in (11, 9, 7, 5, 4, 3, 2):
    ann, truth, blind = synth(cpc, seed=5)
    maj = majority(ann)
    fit = fit_dawid_skene(ann, num_classes=K, majority_labels=maj, settings=DawidSkeneSettings())
    m_acc = float((maj == truth).mean())
    if fit.labels is not None and fit.valid_mask is not None and fit.valid_mask.any():
        d_acc = float((fit.labels == truth).mean())
        used = len(np.unique(fit.labels[fit.valid_mask]))
    else:
        d_acc, used = float("nan"), 0
    agree = fit.majority_agreement if fit.majority_agreement is not None else float("nan")
    print(f"{cpc:14d} {blind:12d} {(ann == ABSTAIN).mean():8.4f} {m_acc:8.4f} {d_acc:8.4f} "
          f"{d_acc - m_acc:8.4f} {fit.status:<18} {used:11d} {agree:10.4f}")
