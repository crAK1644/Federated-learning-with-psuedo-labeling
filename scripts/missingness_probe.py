"""Is informative abstention the mechanism? Isolate it by toggling only that.

Both arms are conditionally independent annotators with identical per-class accuracy -- everything
Dawid-Skene assumes holds, except in arm A the probability that a client votes at all depends on
the item's true class (which is what a confidence threshold produces, and what the real round-50
matrix shows: 4.30 to 12.27 voters per item purely by class).

  informative  : coverage matches the real per-class profile
  uniform      : same overall observation rate and same per-class accuracy, coverage flat

If the collapse appears only in the informative arm, missingness is the cause -- it is the one
Dawid-Skene assumption that the federated open-set protocol structurally violates.
"""

import numpy as np

from ssfl.protocols.dawid_skene import ABSTAIN, DawidSkeneSettings, fit_dawid_skene

K, J = 11, 27
SIZES = [900] * 6 + [700] * 5
# (voters per item, of which correct) measured on the real round-50 annotation matrix
PROFILE = [(6.09, 6.06), (12.13, 10.05), (12.27, 4.22), (12.02, 11.98), (7.00, 1.00),
           (6.99, 5.99), (7.45, 2.88), (11.12, 10.89), (4.30, 4.00), (7.37, 5.82), (4.40, 3.83)]
MEAN_N = float(np.average([n for n, _ in PROFILE], weights=SIZES))


def synth(informative, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(K), SIZES)
    ann = np.full((J, len(truth)), ABSTAIN, dtype=np.int8)
    for c, (n_c, k_c) in enumerate(PROFILE):
        sel = np.nonzero(truth == c)[0]
        p_obs = (n_c if informative else MEAN_N) / J
        p_correct = k_c / n_c  # per-class accuracy held identical across arms
        observed = rng.random((J, len(sel))) < p_obs
        correct = rng.random((J, len(sel))) < p_correct
        wrong = (c + rng.integers(1, K, size=(J, len(sel)))) % K
        block = np.where(correct, c, wrong).astype(np.int8)
        ann[:, sel] = np.where(observed, block, ABSTAIN)
    return ann, truth


def majority(ann):
    out = np.full(ann.shape[1], ABSTAIN, dtype=np.int64)
    for i in range(ann.shape[1]):
        col = ann[:, i][ann[:, i] != ABSTAIN]
        if len(col):
            out[i] = int(np.bincount(col, minlength=K).argmax())
    return out


print(f"{'arm':<13} {'abstain':>8} {'maj_acc':>8} {'ds_acc':>8} {'delta':>8} {'status':<18} "
      f"{'thin_maj':>9} {'thin_ds':>8} {'thick_maj':>10} {'thick_ds':>9}")
for informative in (True, False):
    ann, truth = synth(informative, seed=5)
    maj = majority(ann)
    fit = fit_dawid_skene(ann, num_classes=K, majority_labels=maj, settings=DawidSkeneSettings())
    n_ann = (ann != ABSTAIN).sum(axis=0)
    thin, thick = n_ann <= 6, n_ann >= 11
    if fit.labels is not None and fit.valid_mask is not None and fit.valid_mask.any():
        d = fit.labels
        row = (float((d == truth).mean()), float((d[thin] == truth[thin]).mean()),
               float((d[thick] == truth[thick]).mean()))
    else:
        row = (float("nan"),) * 3
    print(f"{'informative' if informative else 'uniform':<13} "
          f"{(ann == ABSTAIN).mean():8.4f} {(maj == truth).mean():8.4f} {row[0]:8.4f} "
          f"{row[0] - (maj == truth).mean():8.4f} {fit.status:<18} "
          f"{(maj[thin] == truth[thin]).mean():9.4f} {row[1]:8.4f} "
          f"{(maj[thick] == truth[thick]).mean():10.4f} {row[2]:9.4f}")
