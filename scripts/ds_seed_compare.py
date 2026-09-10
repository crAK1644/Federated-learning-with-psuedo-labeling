"""Headline comparison of the two-arm DS-vs-MV replicate on one partition.

The seed-2023 pair produced +0.0349 last-20 accuracy for streaming Dawid-Skene over majority vote.
That is n=1: one partition, one seed, one run per arm. This script reads a *different* partition's
pair and prints the same quantities, so the two can be set side by side.

What transfers across partitions is the SIGN and rough size of the DS-minus-MV gap. Absolute
accuracy does not: the seed-2023 draw happened to leave gafgyt.tcp/gafgyt.udp with no separator
client, which caps every vote-based aggregator at 0.8989 on that draw alone. So the report below
does not assume that pair -- it finds whichever class pair dominates each arm's errors and says
whether it is the same one.

Usage:
    uv run python scripts/ds_seed_compare.py --data artifacts/data-2026 --seed 2026
    uv run python scripts/ds_seed_compare.py --self-check      # reproduces the seed-2023 numbers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import ds_streaming_data as D  # noqa: E402


def top_pair(cm: np.ndarray) -> tuple[int, int, int]:
    """The class pair carrying the most errors, symmetrised: argmax over cm[a,b] + cm[b,a]."""
    best, pair = -1, (0, 1)
    for a in range(cm.shape[0]):
        for b in range(a + 1, cm.shape[0]):
            both = int(cm[a, b] + cm[b, a])
            if both > best:
                best, pair = both, (a, b)
    return pair[0], pair[1], best


def report(data: Path, seed: int, *, quiet: bool = False, discover: bool = True) -> dict[str, float]:
    # The seed-2023 arms were archived off disk, so they are not discoverable by scanning
    # artifacts/runs -- the module's hardcoded table is the only way to reach them. Everything
    # else retargets.
    have_labels = True
    if discover:
        have_labels = D.retarget(data, seed, require_labels=False)
    names = D.class_names()
    order = [arm for arm in D.ORDER if arm in ("stream", "mv")]
    rounds = {arm: D.num_rounds(arm) for arm in order}
    last = min(rounds.values())

    out: dict[str, float] = {}
    say = (lambda *a: None) if quiet else print

    say(f"partition {data.name}  seed {seed}")
    for arm in order:
        say(f"  {D.LABELS[arm]:<22} {D.ARMS[arm][0]}  ({rounds[arm]} rounds)")
    if len(set(rounds.values())) > 1:
        say(f"  ! arms ran different round counts {rounds}; comparing at {last}")

    # --- model accuracy -------------------------------------------------------------------
    say("\nmodel accuracy (test set, mean of last 20 rounds)")
    say(f"  {'arm':<22} {'last-20':>9} {'macro-F1':>9} {'final':>9} {'best':>9}")
    acc = {}
    for arm in order:
        m = D.metrics(arm).sort_values("round")
        acc[arm] = D.last_mean(m, "accuracy")
        out[f"acc_{arm}"] = acc[arm]
        out[f"f1_{arm}"] = D.last_mean(m, "macro_f1")
        say(
            f"  {D.LABELS[arm]:<22} {acc[arm]:9.4f} {out[f'f1_{arm}']:9.4f} "
            f"{m['accuracy'].iloc[-1]:9.4f} {m['accuracy'].max():9.4f}"
        )
    out["gap"] = acc["stream"] - acc["mv"]
    say(f"  {'gap (DS - MV)':<22} {out['gap']:+9.4f}")

    # Per-round dominance is the claim that survives noise better than a single mean.
    s = D.metrics("stream").sort_values("round")["accuracy"].to_numpy()[:last]
    v = D.metrics("mv").sort_values("round")["accuracy"].to_numpy()[:last]
    out["rounds_ahead"] = float((s > v).sum())
    out["worst_round_gap"] = float((s - v).min())
    say(f"  DS ahead in {int(out['rounds_ahead'])}/{last} rounds; worst round {out['worst_round_gap']:+.4f}")

    # --- error structure ------------------------------------------------------------------
    say("\nerror split at the last round")
    say(f"  {'arm':<22} {'errors':>8} {'top pair':>8} {'rest':>8}  dominant confusion")
    for arm in order:
        cm = D.confusion(arm, rounds[arm])
        a, b, both = top_pair(cm)
        total = int(cm.sum() - np.trace(cm))
        out[f"errors_{arm}"] = float(total)
        out[f"reachable_{arm}"] = float(total - both)
        say(
            f"  {D.LABELS[arm]:<22} {total:8d} {both:8d} {total - both:8d}  "
            f"{names[a]}/{names[b]}"
            + ("" if (a, b) == D.UNREACHABLE_PAIR else "   <- NOT the seed-2023 pair")
        )
        if both:
            say(f"  {'':<22} {'':>8} one-directional: {int(cm[a, b])} vs {int(cm[b, a])}")

    # A pair that collapses one way in both arms is structural, not an aggregator failure: the
    # ceiling it implies binds majority vote and Dawid-Skene equally.
    cm_mv = D.confusion("mv", rounds["mv"])
    a, b, both = top_pair(cm_mv)
    out["ceiling"] = 1.0 - both / float(cm_mv.sum())
    say(f"\n  ceiling implied by {names[a]}/{names[b]}: {out['ceiling']:.4f}")
    say(f"  headroom above MV: {out['ceiling'] - acc['mv']:+.4f}, of which DS took "
        f"{(acc['stream'] - acc['mv']) / max(out['ceiling'] - acc['mv'], 1e-9):.0%}")

    # --- per-class ------------------------------------------------------------------------
    say("\nper-class F1, last 20 rounds (|delta| > 0.02 only)")
    mv_f1, ds_f1 = D.per_class_last("mv"), D.per_class_last("stream")
    deltas = (ds_f1 - mv_f1).sort_values(ascending=False)
    for cls, d in deltas.items():
        if abs(d) > 0.02:
            label = names[int(cls)] if str(cls).isdigit() else str(cls)
            say(f"  {label:<18} {mv_f1[cls]:7.4f} -> {ds_f1[cls]:7.4f}  {d:+8.4f}")

    # --- pseudo-label quality -------------------------------------------------------------
    # The distillation target's own accuracy. It runs above test accuracy in both arms; what
    # matters is that the two arms' pseudo-labels diverge in the same direction as their models.
    say("\npseudo-label accuracy vs test accuracy (last 20 rounds)")
    if not have_labels:
        say(f"  skipped: {data} is not on this machine, so there is no sealed open-set truth to")
        say("  score the pseudo-labels against. Copy the partition over and rerun for this block.")
        return out
    for arm in order:
        if arm == "stream":
            pseudo = float(D.pseudo_label_series(arm)["ds"][-D.LAST:].mean())
        else:
            pseudo = float(D.vote_quality(arm)["accuracy"][-D.LAST:].mean())
        out[f"pseudo_{arm}"] = pseudo
        say(f"  {D.LABELS[arm]:<22} pseudo {pseudo:.4f}  test {acc[arm]:.4f}  "
            f"distillation loss {pseudo - acc[arm]:+.4f}")
    out["pseudo_gap"] = out["pseudo_stream"] - out["pseudo_mv"]
    say(f"  {'gap (DS - MV)':<22} {out['pseudo_gap']:+.4f}")

    return out


SELF_CHECK = {
    "acc_stream": 0.8714,
    "acc_mv": 0.8365,
    "gap": 0.0349,
    "reachable_stream": 455.0,
    "reachable_mv": 1119.0,
    "ceiling": 0.8989,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=REPO / "artifacts" / "data-2026")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--self-check", action="store_true",
                    help="run against the seed-2023 pair and assert the published numbers")
    args = ap.parse_args()

    if args.self_check:
        got = report(REPO / "artifacts" / "data", 2023, quiet=True, discover=False)
        bad = {k: (got[k], want) for k, want in SELF_CHECK.items()
               if abs(got[k] - want) > (0.5 if want > 1 else 5e-4)}
        if bad:
            for k, (g, w) in bad.items():
                print(f"MISMATCH {k}: got {g}, expected {w}")
            return 1
        print(f"self-check ok: {len(SELF_CHECK)} published seed-2023 values reproduced")
        return 0

    report(args.data, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
