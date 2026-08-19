"""Stage-7 per-round metric ledger for a finished SSFL run.

Section 7 of DAWID_SKENE_SONRAKI_ADIMLAR.md lists what every round of the coming 50-round
experiment has to record. Half of it the server already writes (``metrics.parquet``,
``per_class_metrics.parquet``, ``confusion_matrices.npz``); the other half is scored against the
sealed open-set labels, which the server must never see. This joins the two offline.

    uv run python scripts/round_metrics.py <run_dir> [--pair gafgyt.combo gafgyt.junk]

Inputs
  <run>/resolved_config.yaml's data_path, audit/source_rows.parquet   sealed open-set labels,
                                                       evaluation only; --data overrides
  <run>/attempts/*/aggregation_audit/*.npz             per-round votes and labels
  <run>/metrics.parquet, per_class_metrics.parquet     what the server recorded live

Nothing here feeds back into training: the sealed labels are read after the run is over, and the
script writes only to ``round_metrics.parquet`` next to the inputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.ds_headroom import sealed_open_labels

NUM_CLASSES = 11
DEFAULT_PAIR = ("gafgyt.combo", "gafgyt.junk")
# Which of the two is expected to be the weaker one, i.e. the class the plan proposes to average
# over rounds 41-50 as the headline metric. Decided in stage 4, before any run.
WEAKER_OF_PAIR = "gafgyt.junk"


def round_number(path: Path) -> int:
    return int(re.search(r"round_(\d+)", path.name).group(1))


def audit_paths(run_dir: Path) -> list[Path]:
    paths = list(run_dir.glob("attempts/*/aggregation_audit/*.npz"))
    if not paths:
        paths = list(run_dir.rglob("ssfl_aggregation_round_*.npz"))
    return sorted(paths, key=round_number)


def label_indices(data_root: Path, pair: tuple[str, str]) -> tuple[int, int]:
    label_map = json.loads((data_root / "label_map.json").read_text())
    missing = [name for name in pair if name not in label_map]
    if missing:
        raise SystemExit(f"unknown class name(s) {missing}; have {sorted(label_map)}")
    return int(label_map[pair[0]]), int(label_map[pair[1]])


def accuracy(labels: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Accuracy over ``mask`` only. Items the aggregator abstained on are not counted as wrong --
    they were never broadcast, so scoring them would mix two different failures together."""
    return float((labels[mask] == truth[mask]).mean()) if mask.any() else float("nan")


def pair_slice(labels: np.ndarray, truth: np.ndarray, mask: np.ndarray, first: int, second: int):
    """Per-class recall and macro F1 restricted to the two target classes.

    Precision counts predictions of the class among items whose truth is one of the pair, so the
    F1 here answers "can the aggregator tell these two apart", not "can it find them in the whole
    open set" -- the second question is what the global metrics already report.
    """
    out: dict[str, float] = {}
    both = mask & np.isin(truth, (first, second))
    f1s = []
    for name, class_id in (("a", first), ("b", second)):
        actual = mask & (truth == class_id)
        predicted = both & (labels == class_id)
        recall = float((labels[actual] == class_id).mean()) if actual.any() else float("nan")
        precision = (
            float((truth[predicted] == class_id).mean()) if predicted.any() else float("nan")
        )
        denominator = precision + recall
        f1 = 0.0 if not denominator or np.isnan(denominator) else 2 * precision * recall / denominator
        out[f"pair_{name}_recall"] = recall
        out[f"pair_{name}_precision"] = precision
        f1s.append(f1)
    out["pair_macro_f1"] = float(np.mean(f1s))
    out["pair_confusion_ab"] = int((mask & (truth == first) & (labels == second)).sum())
    out["pair_confusion_ba"] = int((mask & (truth == second) & (labels == first)).sum())
    return out


def round_row(path: Path, truth: np.ndarray, first: int, second: int) -> dict:
    audit = np.load(path)
    votes = audit["votes_per_class"].astype(np.int64)
    valid = audit["valid_mask"].astype(bool)
    broadcast = audit["global_labels"].astype(np.int64)

    ordered = np.sort(votes, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    tied = valid & (margin == 0)

    row = {
        "round": round_number(path),
        "valid_rate": float(valid.mean()),
        "broadcast_accuracy": accuracy(broadcast, truth, valid),
        "tie_count": int(tied.sum()),
        "tie_rate": float(tied.mean()),
        # The whole case for a re-aggregator is that it does better than a coin flip exactly here.
        "tie_accuracy": accuracy(broadcast, truth, tied),
        "margin_p10": float(np.percentile(margin[valid], 10)) if valid.any() else float("nan"),
        "margin_median": float(np.median(margin[valid])) if valid.any() else float("nan"),
        "margin_mean": float(margin[valid].mean()) if valid.any() else float("nan"),
    }
    # Present only once a fit ran: in shadow mode the broadcast labels are the majority ones, and
    # comparing the two columns is the entire point of a shadow run.
    for prefix, labels_key, mask_key in (
        ("majority", "majority_labels", "majority_valid_mask"),
        ("dawid_skene", "dawid_skene_labels", "dawid_skene_valid_mask"),
    ):
        if labels_key not in audit.files:
            continue
        labels = audit[labels_key].astype(np.int64)
        mask = audit[mask_key].astype(bool)
        row[f"{prefix}_accuracy"] = accuracy(labels, truth, mask)
        row[f"{prefix}_valid_rate"] = float(mask.mean())
        row[f"{prefix}_tie_accuracy"] = accuracy(labels, truth, tied & mask)
    row.update(pair_slice(broadcast, truth, valid, first, second))
    return row


def first_fallback(metrics: pd.DataFrame) -> dict:
    """The round the fit stopped being usable, and why. A run that falls back at round 6 and one
    that falls back at round 44 have the same mean ``ds_status`` and mean nothing alike."""
    if "ds_status" not in metrics:
        return {}
    failed = metrics[metrics["ds_status"].notna() & (metrics["ds_status"] != 0)]
    if failed.empty:
        return {"first_fallback_round": None, "first_fallback_status": None}
    row = failed.sort_values("round").iloc[0]
    return {
        "first_fallback_round": int(row["round"]),
        "first_fallback_status": int(row["ds_status"]),
    }


def server_side(run_dir: Path, first: int, second: int) -> pd.DataFrame:
    """Sealed-test columns the server already wrote, narrowed to the target pair."""
    metrics_path = run_dir / "metrics.parquet"
    if not metrics_path.exists():
        return pd.DataFrame(columns=["round"])
    metrics = pd.read_parquet(metrics_path)
    keep = [
        column
        for column in metrics.columns
        if column == "round" or column.startswith("ds_") or column in {"accuracy", "macro_f1"}
    ]
    table = metrics[keep].rename(columns={"accuracy": "test_accuracy"})
    per_class_path = run_dir / "per_class_metrics.parquet"
    if per_class_path.exists():
        per_class = pd.read_parquet(per_class_path)
        for name, class_id in (("a", first), ("b", second)):
            sliced = per_class[per_class["class"] == class_id][["round", "recall", "f1"]]
            table = table.merge(
                sliced.rename(
                    columns={"recall": f"test_pair_{name}_recall", "f1": f"test_pair_{name}_f1"}
                ),
                on="round",
                how="left",
            )
    return table


def data_root_for(run_dir: Path, override: Path | None) -> Path:
    """Where to read the sealed open-set labels from, defaulting to the root the run trained on.

    The controlled target-pair arms train against ``artifacts/data-balanced`` and
    ``artifacts/data-specialist`` rather than the default root. Scoring one of those against
    ``artifacts/data`` does not crash -- the open splits have the same shape -- it silently compares
    the broadcast labels to a different sampling of the dataset, which is the one failure this
    ledger exists to catch rather than commit.
    """
    if override is not None:
        return override
    config = run_dir / "resolved_config.yaml"
    if not config.exists():
        raise SystemExit(f"{config} is missing; pass --data explicitly")
    for line in config.read_text().splitlines():
        if line.startswith("data_path:"):
            return Path(line.split(":", 1)[1].strip())
    raise SystemExit(f"{config} has no data_path; pass --data explicitly")


def build(run_dir: Path, data_root: Path, pair: tuple[str, str]) -> tuple[pd.DataFrame, dict]:
    paths = audit_paths(run_dir)
    if not paths:
        raise SystemExit(f"no aggregation audit npz under {run_dir}")
    first, second = label_indices(data_root, pair)
    truth = sealed_open_labels(data_root)
    table = pd.DataFrame([round_row(path, truth, first, second) for path in paths])
    table = table.merge(server_side(run_dir, first, second), on="round", how="left")
    weaker = "a" if pair[0] == WEAKER_OF_PAIR else "b"
    summary = {
        "run": run_dir.name,
        "pair": list(pair),
        "weaker_class": WEAKER_OF_PAIR,
        "rounds_audited": len(table),
        **first_fallback(table),
    }
    tail = table[table["round"] > table["round"].max() - 10]
    summary["weaker_recall_last10_mean"] = float(tail[f"pair_{weaker}_recall"].mean())
    summary["broadcast_accuracy_last10_mean"] = float(tail["broadcast_accuracy"].mean())
    return table, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    # No default: it is resolved from the run itself, so a run trained on a non-default
    # prepared root cannot be scored against the wrong sealed labels by omission.
    parser.add_argument("--data", default=None, type=Path)
    parser.add_argument("--pair", nargs=2, default=list(DEFAULT_PAIR))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    table, summary = build(args.run_dir, data_root_for(args.run_dir, args.data), tuple(args.pair))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(json.dumps(summary, indent=2))
    out = args.out or args.run_dir / "round_metrics.parquet"
    table.to_parquet(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
