"""Does annotator correlation alone reproduce the class collapse seen in the live run?

Holds marginal per-client accuracy fixed and varies only how much clients copy a shared
"global model" prediction -- the federated reality (distilled shared model, overlapping shards)
that violates Dawid-Skene's conditional-independence assumption. If correlation is the mechanism
#33 claims, DS should beat majority at rho=0 and collapse classes as rho rises.
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


def synth(rho, acc=0.75, seed=0):
    """rho = probability a client emits the shared global prediction instead of its own."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, K, size=N)
    # The shared model: same marginal accuracy, but every client copying it errs identically.
    g_ok = rng.random(N) < acc
    shared = np.where(g_ok, truth, (truth + rng.integers(1, K, size=N)) % K)
    ann = np.empty((J, N), dtype=np.int8)
    for j in range(J):
        own_ok = rng.random(N) < acc
        own = np.where(own_ok, truth, (truth + rng.integers(1, K, size=N)) % K)
        ann[j] = np.where(rng.random(N) < rho, shared, own)
    return ann, truth


print(f"{'rho':>5} {'maj_acc':>8} {'ds_acc':>8} {'delta':>8} {'status':<28} {'latent_used':>11} "
      f"{'worst_class_recall':>18}")
for rho in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95):
    ann, truth = synth(rho)
    maj = majority(ann)
    fit = fit_dawid_skene(ann, num_classes=K, majority_labels=maj,
                          settings=DawidSkeneSettings(min_clients=2))
    m_acc = float((maj == truth).mean())
    if fit.labels is not None and fit.valid_mask is not None and fit.valid_mask.any():
        d_acc = float((fit.labels == truth).mean())
        used = len(np.unique(fit.labels[fit.valid_mask]))
        recalls = [float((fit.labels[truth == c] == c).mean()) for c in range(K)]
        worst = min(recalls)
    else:  # a rejected fit returns all-ABSTAIN by design
        d_acc, used, worst = float("nan"), 0, float("nan")
    print(f"{rho:5.2f} {m_acc:8.4f} {d_acc:8.4f} {d_acc - m_acc:8.4f} {fit.status:<28} "
          f"{used:11d} {worst:18.4f}")
