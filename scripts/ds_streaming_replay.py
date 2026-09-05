"""Offline gate for the streaming Dawid-Skene aggregator: replay it on recorded annotations.

The three finished scenario-3 arms wrote the full ``(J, N)`` client label matrix into every
round's aggregation audit, so the whole aggregator can be re-run on real votes with no GPU and no
training. This script drives the PRODUCTION estimator -- ``fit_dawid_skene`` with
``state_decay > 0`` -- round by round, threading the online-EM state exactly the way
``strategies/ssfl.py`` does, and scores the result against the sealed open-set labels.

What it can and cannot tell you: the votes are fixed, recorded under whatever aggregator that arm
actually ran. So this measures the aggregator, holding the training trajectory constant. It cannot
measure the closed loop -- a better label stream would have trained different clients, which would
have produced different votes. That is what the GPU arm is for. Replaying on ``--arm ds_only``
and ``--arm hybrid`` is the honest counterweight: those substrates are what the votes look like
once the clients have already been trained on bad labels.

Sealed labels are used here only for offline measurement. They never enter aggregation, training,
or fallback logic (DAWID_SKENE_FEASIBILITY_PLAN.md section 1).

    uv run python scripts/ds_streaming_replay.py --arm mv
    uv run python scripts/ds_streaming_replay.py --arm mv --sweep
"""

from __future__ import annotations

import argparse
import glob
import itertools
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ds_headroom import sealed_open_labels  # noqa: E402
from ssfl.protocols.dawid_skene import (  # noqa: E402
    ABSTAIN,
    DawidSkeneSettings,
    fit_dawid_skene,
)

NUM_CLASSES = 11

# The three finished 200-round scenario-3 arms. Same data, same clients, same protocol; they
# differ only in what the server broadcast, which is exactly the substrate difference this script
# exists to expose.
ARMS = {
    "mv": "ssfl-s3-experiment1_s3_mv_200-79e618b3d32e06ee",
    "ds_only": "ssfl-s3-experiment1_s3_ds_only_200-b09be0a7fd6fcfe2",
    "hybrid": "ssfl-s3-experiment1_s3_hybrid_200-1f17d410103f5438",
}

# The plan's pre-GPU gate, on the MV arm. Numbers, not adjectives, so the run either earns the
# GPU time or it does not.
GATE_MEAN_DELTA = 0.02
GATE_MAX_NEGATIVE_ROUNDS = 5
GATE_LAST20_DELTA = 0.015


def load_rounds(run_dir: Path) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """``(round, annotations, majority_labels, majority_valid_mask)`` in round order.

    ``majority_labels`` is only present in an audit written by an arm that fitted Dawid-Skene; the
    majority arm's own broadcast IS the majority result, so ``global_labels`` is the same thing
    there.
    """
    paths = glob.glob(str(run_dir / "attempts" / "*" / "aggregation_audit" / "*.npz"))
    if not paths:
        raise SystemExit(f"no aggregation audits under {run_dir}")
    rounds = []
    for path in sorted(paths, key=lambda p: int(re.search(r"round_(\d+)", p).group(1))):
        audit = np.load(path)
        if "annotations" not in audit.files:
            continue
        key = "majority_labels" if "majority_labels" in audit.files else "global_labels"
        mask_key = "majority_valid_mask" if key == "majority_labels" else "valid_mask"
        rounds.append(
            (
                int(re.search(r"round_(\d+)", path).group(1)),
                audit["annotations"],
                audit[key].astype(np.int64),
                audit[mask_key].astype(bool),
            )
        )
    return rounds


def replay(rounds, sealed: np.ndarray, settings: DawidSkeneSettings) -> dict:
    """Run the aggregator forward over the recorded rounds, carrying state, and score each one."""
    state = None
    deltas, ds_accuracy, mv_accuracy, statuses = [], [], [], []
    for _, votes, majority, majority_mask in rounds:
        senders = tuple(f"c{j:03d}" for j in range(votes.shape[0]))
        fit = fit_dawid_skene(
            votes,
            num_classes=NUM_CLASSES,
            majority_labels=majority,
            settings=settings,
            senders=senders,
            state=state,
        )
        statuses.append(fit.status)
        if fit.state is not None:
            state = fit.state
        labels = fit.candidate_labels if fit.numerically_valid else majority
        mask = fit.candidate_valid_mask if fit.numerically_valid else majority_mask
        # Score both aggregators on the same items, so the delta is the aggregator and not a
        # difference in which open samples each one was willing to label.
        scored = mask & majority_mask & (majority != ABSTAIN)
        ds = float((labels[scored] == sealed[scored]).mean())
        mv = float((majority[scored] == sealed[scored]).mean())
        ds_accuracy.append(ds)
        mv_accuracy.append(mv)
        deltas.append(ds - mv)
    deltas = np.array(deltas)
    return {
        "mean": float(deltas.mean()),
        "last20": float(deltas[-20:].mean()),
        "min": float(deltas.min()),
        "negative": int((deltas < 0).sum()),
        "final_ds": ds_accuracy[-1],
        "final_mv": mv_accuracy[-1],
        "rounds": len(deltas),
        "statuses": statuses,
        "state": state,
    }


def settings_for(alpha: float, decay: float, init_rounds: float, theta: float):
    """Everything except the four streaming knobs is the DS-only arm's own configuration."""
    return DawidSkeneSettings(
        confusion_prior="diagonal",
        confusion_prior_diagonal=theta,
        confusion_pseudocount=alpha,
        state_decay=decay,
        state_init_rounds=init_rounds,
        class_alignment=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="mv", choices=sorted(ARMS))
    parser.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--data", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--init-rounds", type=float, default=150.0)
    parser.add_argument("--theta", type=float, default=0.9)
    parser.add_argument("--sweep", action="store_true", help="grid over the four knobs first")
    parser.add_argument("--gate", action="store_true", help="exit non-zero if the gate fails")
    args = parser.parse_args()

    sealed = sealed_open_labels(args.data)
    rounds = load_rounds(args.runs / ARMS[args.arm])
    print(f"arm={args.arm}  rounds with annotations: {len(rounds)}")

    if args.sweep:
        grid = list(
            itertools.product(
                [20.0, 50.0],
                [0.7, 0.8, 0.9, 0.95, 0.98],
                [150.0, 300.0, 600.0, 1200.0],
                [0.8, 0.9, 0.95],
            )
        )
        scored = []
        for alpha, decay, init_rounds, theta in grid:
            result = replay(rounds, sealed, settings_for(alpha, decay, init_rounds, theta))
            scored.append((result["mean"], result["last20"], result["min"], result["negative"],
                           alpha, decay, init_rounds, theta))
        scored.sort(reverse=True)
        print(f"{'mean':>8} {'last20':>8} {'min':>8} {'neg':>4}  alpha  rho   R0     theta")
        for row in scored[:12]:
            print(f"{row[0]:+8.4f} {row[1]:+8.4f} {row[2]:+8.4f} {row[3]:4d}  "
                  f"{row[4]:<6g} {row[5]:<5g} {row[6]:<6g} {row[7]}")
        print("...worst:")
        for row in scored[-3:]:
            print(f"{row[0]:+8.4f} {row[1]:+8.4f} {row[2]:+8.4f} {row[3]:4d}  "
                  f"{row[4]:<6g} {row[5]:<5g} {row[6]:<6g} {row[7]}")
        print()

    result = replay(
        rounds, sealed, settings_for(args.alpha, args.decay, args.init_rounds, args.theta)
    )
    print(f"chosen alpha={args.alpha} rho={args.decay} R0={args.init_rounds} theta={args.theta}")
    print(f"  mean delta {result['mean']:+.4f}   last-20 {result['last20']:+.4f}   "
          f"worst round {result['min']:+.4f}   negative rounds {result['negative']}/"
          f"{result['rounds']}")
    print(f"  final round: streaming DS {result['final_ds']:.4f}  majority {result['final_mv']:.4f}")
    bad = sorted({s for s in result["statuses"] if s != "ok"})
    print(f"  non-ok statuses: {bad or 'none'}")

    if args.gate:
        failures = []
        if result["mean"] < GATE_MEAN_DELTA:
            failures.append(f"mean delta {result['mean']:+.4f} < {GATE_MEAN_DELTA}")
        if result["negative"] > GATE_MAX_NEGATIVE_ROUNDS:
            failures.append(f"{result['negative']} negative rounds > {GATE_MAX_NEGATIVE_ROUNDS}")
        if result["last20"] < GATE_LAST20_DELTA:
            failures.append(f"last-20 delta {result['last20']:+.4f} < {GATE_LAST20_DELTA}")
        if failures:
            print("GATE FAILED: " + "; ".join(failures))
            return 1
        print("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
