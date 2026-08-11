"""Offline headroom check for the Dawid-Skene feasibility question.

Answers, from artifacts that already exist (no training, no code change): how wrong are the
majority-vote global labels, and how much of that error could ANY re-aggregator fix?

Inputs
  artifacts/data/audit/source_rows.parquet          sealed open-set labels (offline eval only)
  <run>/attempts/*/aggregation_audit/*.npz          per-round votes_per_class already recorded

Sealed labels are used here only for offline measurement. They never enter aggregation, training,
or fallback logic (DAWID_SKENE_FEASIBILITY_PLAN.md section 1).

    uv run python scripts/ds_headroom.py [run_dir] [--plot out.png]
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

NUM_CLASSES = 11
DEFAULT_RUN = "ssfl-s1-gpu50_baseline-469254ea52f11ea0"


def sealed_open_labels(data_root: Path) -> np.ndarray:
    """Reconstruct open-split ground truth in open/features.npy row order.

    ponytail: read from the existing audit parquet instead of writing a new labels file into
    artifacts/data -- a new file there changes dataset_manifest.json's checksum set, which changes
    every run_id and breaks comparability with runs already on disk.
    """
    rows = pd.read_parquet(data_root / "audit" / "source_rows.parquet")
    open_rows = rows[rows.split == "open"]
    index = open_rows.global_index.to_numpy().astype(int)
    labels = np.full(len(open_rows), -1, dtype=np.int64)
    labels[index] = open_rows.label.to_numpy().astype(int)
    assert (labels >= 0).all(), "open split has a row with no global_index"
    return labels


def round_stats(path: str, sealed: np.ndarray) -> dict:
    round_number = int(re.search(r"round_(\d+)", path).group(1))
    audit = np.load(path)
    votes = audit["votes_per_class"].astype(np.int64)
    labels = audit["global_labels"].astype(np.int64)
    valid = audit["valid_mask"].astype(bool)

    ordered = np.sort(votes, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    correct = (labels == sealed) & valid
    # A re-aggregator can only choose a class some client actually proposed for that item, so
    # "true class got at least one vote" is a hard ceiling on any vote-based aggregator.
    reachable = votes[np.arange(len(sealed)), sealed] > 0

    return {
        "round": round_number,
        "valid_rate": float(valid.mean()),
        "majority_acc": float(correct[valid].sum() / valid.sum()),
        "reachable_ceiling": float((valid & (reachable | correct)).sum() / valid.sum()),
        "tie_rate": float((valid & (margin == 0)).mean()),
        "margin_le2_rate": float((valid & (margin <= 2)).mean()),
        "errors_with_margin_le2": float(
            ((~correct) & valid & (margin <= 2)).sum() / max(1, int(((~correct) & valid).sum()))
        ),
        "mean_annotators": float(votes.sum(axis=1)[valid].mean()),
    }


def per_class_recall(path: str, sealed: np.ndarray) -> pd.DataFrame:
    audit = np.load(path)
    labels = audit["global_labels"].astype(np.int64)
    valid = audit["valid_mask"].astype(bool)
    out = []
    for class_id in range(NUM_CLASSES):
        mask = valid & (sealed == class_id)
        counts = np.bincount(labels[mask], minlength=NUM_CLASSES)
        absorbed = int(np.argsort(counts)[::-1][0])
        out.append(
            {
                "class": class_id,
                "n": int(mask.sum()),
                "recall": float((labels[mask] == class_id).mean()),
                "mapped_to": absorbed,
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", default=DEFAULT_RUN, type=Path)
    parser.add_argument("--data", default=Path("artifacts/data"), type=Path)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()

    sealed = sealed_open_labels(args.data)
    audits = sorted(
        glob.glob(str(args.run_dir / "attempts" / "*" / "aggregation_audit" / "*.npz")),
        key=lambda p: int(re.search(r"round_(\d+)", p).group(1)),
    )
    if not audits:
        raise SystemExit(f"no aggregation_audit/*.npz under {args.run_dir}")

    table = pd.DataFrame([round_stats(p, sealed) for p in audits])
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nmean over rounds:")
    print(table.drop(columns=["round"]).mean().to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nper-class recall at round {table['round'].iloc[-1]}:")
    print(per_class_recall(audits[-1], sealed).to_string(index=False))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        recall = per_class_recall(audits[-1], sealed)
        fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
        left.plot(table["round"], table["reachable_ceiling"], label="reachable ceiling")
        left.plot(table["round"], table["majority_acc"], label="majority vote")
        left.fill_between(
            table["round"], table["majority_acc"], table["reachable_ceiling"], alpha=0.15
        )
        left.set(xlabel="round", ylabel="open-set label accuracy", ylim=(0, 1))
        left.set_title("Aggregation headroom (scenario 1, 50 rounds)")
        left.legend(loc="lower right")
        colors = ["tab:red" if r < 0.5 else "tab:blue" for r in recall["recall"]]
        right.bar(recall["class"], recall["recall"], color=colors)
        right.set(xlabel="true class", ylabel="majority-label recall", xticks=range(NUM_CLASSES))
        right.set_title("Where the labels are lost (final round)")
        for _, row in recall[recall["recall"] < 0.5].iterrows():
            right.text(
                row["class"], row["recall"] + 0.03, f"-> {int(row['mapped_to'])}", ha="center"
            )
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
