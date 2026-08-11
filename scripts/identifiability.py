"""Can any annotator tell the merged classes apart?

Dawid-Skene separates two true classes only if some annotator's label distribution differs between
them. That annotator must vote on BOTH -- a client that only ever sees class a, and a different
client that only ever sees class b, are jointly consistent with a single latent class covering both.
No amount of data fixes that: the classes are non-identifiable and merging them strictly increases
the likelihood, because it frees the parameters that were splitting them.

Reports, for every class pair, how much voting evidence separates them.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from ds_stage0 import sealed_open_labels

from ssfl.protocols.dawid_skene import ABSTAIN

K = 11
NAMES = ["benign", "gaf.combo", "gaf.junk", "gaf.scan", "gaf.tcp", "gaf.udp",
         "mir.ack", "mir.scan", "mir.syn", "mir.udp", "mir.udpplain"]
MERGED = {(0, 3), (4, 5), (5, 8), (4, 8), (1, 2), (2, 10), (1, 10)}

truth = sealed_open_labels(Path("artifacts/data"))
path = next(Path(".").glob(
    "artifacts/runs/ssfl-s1-ds50_shadow-*/attempts/*/aggregation_audit/*round_50*.npz"))
A = np.load(path)["annotations"]
J = A.shape[0]

# Per client and true class: how many items it votes on, and its label distribution there.
votes = np.zeros((J, K, K))  # [client][true class][emitted label]
for j in range(J):
    obs = A[j] != ABSTAIN
    np.add.at(votes[j], (truth[obs], A[j][obs].astype(np.int64)), 1)
seen = votes.sum(axis=2)  # [client][true class] items voted on

print("separation between class pairs (higher = more distinguishable)")
print(f"{'pair':<26} {'both':>5} {'sep':>6} {'merged?':>8}")
rows = []
for a in range(K):
    for b in range(a + 1, K):
        both = np.nonzero((seen[:, a] >= 20) & (seen[:, b] >= 20))[0]
        if len(both):
            pa = votes[both, a] / seen[both, a][:, None]
            pb = votes[both, b] / seen[both, b][:, None]
            sep = float(np.abs(pa - pb).sum(axis=1).max() / 2)  # best client's total variation
        else:
            sep = 0.0
        rows.append((sep, a, b, len(both)))

for sep, a, b, both in sorted(rows)[:12]:
    tag = "MERGED" if (a, b) in MERGED or (b, a) in MERGED else ""
    print(f"{NAMES[a] + ' vs ' + NAMES[b]:<26} {both:5d} {sep:6.3f} {tag:>8}")

merged_seps = [s for s, a, b, _ in rows if (a, b) in MERGED]
other_seps = [s for s, a, b, _ in rows if (a, b) not in MERGED]
print(f"\nmean separation, pairs DS merged : {np.mean(merged_seps):.3f}  (n={len(merged_seps)})")
print(f"mean separation, pairs DS kept   : {np.mean(other_seps):.3f}  (n={len(other_seps)})")
print(f"pairs with zero separating client: "
      f"{sum(1 for s, *_ in rows if s == 0.0)} of {len(rows)}")

print(f"\n{'class':<14} {'clients voting on it':>21} {'mean items/client':>18}")
for c in range(K):
    active = seen[:, c] >= 20
    print(f"{NAMES[c]:<14} {active.sum():21d} {seen[active, c].mean() if active.any() else 0:18.1f}")
