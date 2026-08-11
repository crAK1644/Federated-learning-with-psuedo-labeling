"""Where does the DS fit's likelihood advantage over the truth come from?

Decomposes the per-item log-likelihood difference (fit-fitted params minus truth-fitted params)
by true class and by how many annotators labelled the item. If the advantage is bought by
sacrificing thinly-annotated classes to better explain heavily-annotated ones, the missingness
is doing the damage -- Dawid-Skene treats an unobserved entry as carrying no information, and
here the number of annotators is a direct function of the true class.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from ds_stage0 import sealed_open_labels

from ssfl.protocols.dawid_skene import ABSTAIN, DawidSkeneSettings, fit_dawid_skene

K = 11
S = DawidSkeneSettings()
NAMES = ["benign", "gafgyt.combo", "gafgyt.junk", "gafgyt.scan", "gafgyt.tcp", "gafgyt.udp",
         "mirai.ack", "mirai.scan", "mirai.syn", "mirai.udp", "mirai.udpplain"]


def fit_params(posterior, A):
    """M-step only: the parameters this labelling implies."""
    obs = A != ABSTAIN
    prior_counts = posterior.sum(axis=0) + S.class_prior_pseudocount
    prior = prior_counts / prior_counts.sum()
    conf = np.empty((A.shape[0], K, K))
    for j in range(A.shape[0]):
        idx = np.nonzero(obs[j])[0]
        oh = np.eye(K)[A[j][obs[j]].astype(np.int64)]
        c = posterior[idx].T @ oh + S.confusion_pseudocount
        conf[j] = c / c.sum(axis=1, keepdims=True)
    return prior, conf


def per_item_loglik(prior, conf, A):
    obs = A != ABSTAIN
    log_score = np.tile(np.log(np.maximum(prior, S.epsilon)), (A.shape[1], 1))
    lc = np.log(np.maximum(conf, S.epsilon))
    for j in range(A.shape[0]):
        idx = np.nonzero(obs[j])[0]
        lab = A[j][obs[j]].astype(np.int64)
        log_score[idx] += lc[j][:, lab].T
    m = log_score.max(axis=1, keepdims=True)
    return (np.log(np.exp(log_score - m).sum(axis=1, keepdims=True)) + m).ravel()


truth = sealed_open_labels(Path("artifacts/data"))
path = next(Path(".").glob(
    "artifacts/runs/ssfl-s1-ds50_shadow-*/attempts/*/aggregation_audit/*round_50*.npz"))
z = np.load(path)
A, maj = z["annotations"], z["majority_labels"].astype(np.int64)
fit = fit_dawid_skene(A, num_classes=K, majority_labels=maj, settings=S)

ll_truth = per_item_loglik(*fit_params(np.eye(K)[truth], A), A)
ll_fit = per_item_loglik(*fit_params(np.eye(K)[fit.labels], A), A)
gain = ll_fit - ll_truth
n_ann = (A != ABSTAIN).sum(axis=0)

print(f"total likelihood gain of DS fit over truth: {gain.sum():+.1f}\n")
print(f"{'true class':<16} {'n':>5} {'ann/item':>9} {'maj_rec':>8} {'ds_rec':>7} "
      f"{'ll_gain':>10} {'gain/item':>10}")
for c in range(K):
    sel = truth == c
    print(f"{NAMES[c]:<16} {sel.sum():5d} {n_ann[sel].mean():9.2f} "
          f"{(maj[sel] == c).mean():8.3f} {(fit.labels[sel] == c).mean():7.3f} "
          f"{gain[sel].sum():+10.1f} {gain[sel].mean():+10.3f}")

print(f"\n{'annotators on item':<20} {'n':>6} {'maj_acc':>8} {'ds_acc':>7} {'ll_gain':>10}")
for lo, hi in ((0, 5), (5, 7), (7, 9), (9, 11), (11, 28)):
    sel = (n_ann >= lo) & (n_ann < hi)
    if not sel.any():
        continue
    print(f"{f'{lo}-{hi - 1}':<20} {sel.sum():6d} {(maj[sel] == truth[sel]).mean():8.3f} "
          f"{(fit.labels[sel] == truth[sel]).mean():7.3f} {gain[sel].sum():+10.1f}")

print("\nwhere DS sends each class (rows=truth, cols=ds label, % of row):")
print("                 " + " ".join(f"{c:>5d}" for c in range(K)))
for c in range(K):
    sel = truth == c
    row = np.bincount(fit.labels[sel][fit.labels[sel] >= 0], minlength=K) / max(sel.sum(), 1)
    print(f"{NAMES[c]:<16} " + " ".join(f"{v * 100:5.1f}" for v in row))
