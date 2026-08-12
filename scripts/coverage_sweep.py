"""Does client class-coverage broaden as the global model converges?

The 200-round run's stated hypothesis (configs/dawid_skene_shadow_200.yaml): #36 measured the
blind pairs at round 50, on a model 150 rounds from convergence. If clients emit labels over more
classes as they converge, the blind pairs close, identifiability is restored, and Dawid-Skene
recovers for a stated reason. This sweeps the same measurement as scripts/identifiability.py
across every dumped round of both arms.

    uv run python scripts/coverage_sweep.py

A pair is BLIND when no client votes on both classes (>= MIN_ITEMS items each): no annotator's
label distribution can differ between them, so one latent class explains both and merging them
raises the likelihood. That is the #36 mechanism, counted per round.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from ds_stage0 import sealed_open_labels

from ssfl.protocols.dawid_skene import ABSTAIN

K = 11
MIN_ITEMS = 20
ROUNDS = [1, 25, 50, 100, 150, 200]
ARMS = {"shadow": "ssfl-s1-ds200_shadow-*", "active": "ssfl-s1-ds200_active-*"}

truth = sealed_open_labels(Path("artifacts/data"))


def coverage(path: Path) -> tuple[int, float, float, int]:
    """(blind pairs, mean separation, mean classes emitted per client, clients)."""
    A = np.load(path)["annotations"]
    J = A.shape[0]
    votes = np.zeros((J, K, K))  # [client][true class][emitted label]
    for j in range(J):
        obs = A[j] != ABSTAIN
        np.add.at(votes[j], (truth[obs], A[j][obs].astype(np.int64)), 1)
    seen = votes.sum(axis=2)  # [client][true class] items voted on

    blind, seps = 0, []
    for a in range(K):
        for b in range(a + 1, K):
            both = np.nonzero((seen[:, a] >= MIN_ITEMS) & (seen[:, b] >= MIN_ITEMS))[0]
            if not len(both):
                blind += 1
                seps.append(0.0)
                continue
            pa = votes[both, a] / seen[both, a][:, None]
            pb = votes[both, b] / seen[both, b][:, None]
            seps.append(float(np.abs(pa - pb).sum(axis=1).max() / 2))
    # Classes a client actually emits a label for -- the quantity that has to grow.
    emitted = (votes.sum(axis=1) >= MIN_ITEMS).sum(axis=1)
    return blind, float(np.mean(seps)), float(np.mean(emitted)), J


for arm, pattern in ARMS.items():
    print(f"\n=== {arm}")
    print(f"{'round':>6} {'blind':>6} {'mean sep':>9} {'classes/client':>15} {'clients':>8}")
    for rnd in ROUNDS:
        found = sorted(Path(".").glob(f"{pattern}/attempts/*/aggregation_audit/*round_{rnd}.npz"))
        if not found:
            print(f"{rnd:6d}  (no dump)")
            continue
        blind, sep, emitted, J = coverage(found[0])
        print(f"{rnd:6d} {blind:6d} {sep:9.3f} {emitted:15.2f} {J:8d}")
