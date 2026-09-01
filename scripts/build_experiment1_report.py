#!/usr/bin/env python
"""Deney 1 / Scenario 3: MV-only vs DS-only vs Hybrid comparison tables and figures.

Reads the three finished 200-round runs under ``artifacts/runs`` and writes every table and
figure listed in DENEY_1_SCENARIO_3_DENEY_PLANI.md sections 13-15 into ``artifacts/experiment1_s3``.

The per-round audit derivations (pseudo-label accuracy, ties, client reliability) scan 600
``aggregation_audit/*.npz`` files, so they are cached in ``cache/``; delete that directory to
recompute. Raw annotation matrices stay where they are -- nothing here copies them into the
output tree (plan section 10: restricted diagnostics).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ssfl.protocols.dawid_skene import ABSTAIN, DawidSkeneSettings, fit_dawid_skene
from ssfl.strategies.ssfl import DS_STATUS_CODES

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "artifacts" / "data"
OUT = REPO / "artifacts" / "experiment1_s3"
FIGS = OUT / "figures"
CACHE = OUT / "cache"
ARMS = ["mv", "ds_only", "hybrid"]
LABELS = {"mv": "MV-only", "ds_only": "DS-only", "hybrid": "Hybrid"}
COLORS = {"mv": "#1f77b4", "ds_only": "#d62728", "hybrid": "#2ca02c"}
STATUS_NAME = {v: k for k, v in DS_STATUS_CODES.items()}
LAST = 10  # plan section 13: the primary metric is the mean over the final 10 rounds


def run_dir(arm: str) -> Path:
    hits = sorted(glob.glob(str(REPO / f"artifacts/runs/ssfl-s3-experiment1_s3_{arm}_200-*")))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one run dir for {arm}, found {hits}")
    return Path(hits[0])


def audit_files(arm: str) -> list[Path]:
    files = glob.glob(str(run_dir(arm) / "attempts/*/aggregation_audit/ssfl_aggregation_round_*.npz"))
    return sorted((Path(f) for f in files), key=lambda p: int(p.stem.rsplit("_", 1)[1]))


def open_truth() -> np.ndarray:
    rows = pd.read_parquet(DATA / "audit" / "source_rows.parquet")
    # ``global_index`` is the row's position in ``open/features.npy`` -- prepare_data sorts the
    # open rows by (device_id, label, position) and writes the features in that order. ``position``
    # alone is an index *within* a (device, label) group, so sorting on it scrambles the mapping.
    rows = rows[rows["split"] == "open"].sort_values("global_index")
    return rows["label"].to_numpy(dtype=np.int64)


def class_names() -> list[str]:
    mapping = json.loads((DATA / "label_map.json").read_text())
    return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


def assignments() -> pd.DataFrame:
    payload = json.loads((DATA / "scenarios" / "3.json").read_text())
    return pd.DataFrame(
        [
            {
                "row": i,
                "client_id": c["client_id"],
                "device_id": c["device_id"],
                "num_examples": c["num_examples"],
                "classes": sorted(int(k) for k in c["class_local_indices"]),
                "counts": {int(k): len(v) for k, v in c["class_local_indices"].items()},
            }
            # sorted by client_id: build_annotation_matrix sorts senders, so this is the row
            # order of every ``annotations`` matrix in the audit npz files.
            for i, c in enumerate(sorted(payload["clients"], key=lambda c: c["client_id"]))
        ]
    )


# --------------------------------------------------------------------------------------- caches


def audit_series(arm: str, truth: np.ndarray, num_classes: int) -> pd.DataFrame:
    """Per-round quantities that only the audit npz files can answer."""
    cached = CACHE / f"audit_series_{arm}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    records = []
    for path in audit_files(arm):
        z = np.load(path)
        rnd = int(path.stem.rsplit("_", 1)[1])
        labels, valid = z["global_labels"].astype(np.int64), z["valid_mask"]
        votes = z["votes_per_class"]
        top = votes.max(axis=1)
        tie = (votes == top[:, None]).sum(axis=1) > 1
        tie = tie & (top > 0)
        rec = {
            "round": rnd,
            "coverage": float(valid.mean()),
            "pseudo_accuracy": _acc(labels, truth, valid),
            "tie_count": int(tie.sum()),
            "tie_accuracy": _acc(labels, truth, tie & valid),
            "participation_mean": float(z["participating_counts"].mean()),
        }
        if "majority_labels" in z:
            mv_lab, mv_valid = z["majority_labels"].astype(np.int64), z["majority_valid_mask"]
            ds_lab, ds_valid = z["dawid_skene_labels"].astype(np.int64), z["dawid_skene_valid_mask"]
            both = mv_valid & ds_valid
            rec |= {
                "mv_candidate_accuracy": _acc(mv_lab, truth, mv_valid),
                "ds_candidate_accuracy": _acc(ds_lab, truth, ds_valid),
                # Disagreement measured on the samples both candidates called, which is the only
                # place the two are comparable.
                "candidate_disagreement": float((mv_lab[both] != ds_lab[both]).mean())
                if both.any()
                else np.nan,
                "ds_accuracy_on_disagreement": _acc(ds_lab, truth, both & (mv_lab != ds_lab)),
                "mv_accuracy_on_disagreement": _acc(mv_lab, truth, both & (mv_lab != ds_lab)),
            }
        records.append(rec)
    frame = pd.DataFrame(records).sort_values("round").reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cached)
    return frame


def _acc(labels: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return float((labels[mask] == truth[mask]).mean()) if mask.any() else float("nan")


def reliability(arm: str, truth: np.ndarray, num_classes: int) -> dict[str, np.ndarray]:
    """``correct``/``observed`` counts shaped (clients, rounds, classes).

    Reliability is a rate, and the denominators differ wildly per client and class in Scenario 3
    (Dirichlet alpha=0.1), so the counts are cached rather than the ratio -- a mean of ratios over
    rounds is not the same thing as the pooled rate, and the tables need both.
    """
    cached = CACHE / f"reliability_{arm}.npz"
    if cached.exists():
        z = np.load(cached)
        return {"correct": z["correct"], "observed": z["observed"], "rounds": z["rounds"]}
    files = audit_files(arm)
    per_class = [np.flatnonzero(truth == k) for k in range(num_classes)]
    num_clients = np.load(files[0])["annotations"].shape[0]
    correct = np.zeros((num_clients, len(files), num_classes), dtype=np.int32)
    observed = np.zeros_like(correct)
    rounds = np.array([int(p.stem.rsplit("_", 1)[1]) for p in files])
    for t, path in enumerate(files):
        ann = np.load(path)["annotations"]
        for k, cols in enumerate(per_class):
            block = ann[:, cols]
            observed[:, t, k] = (block != ABSTAIN).sum(axis=1)
            correct[:, t, k] = (block == k).sum(axis=1)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cached, correct=correct, observed=observed, rounds=rounds)
    return {"correct": correct, "observed": observed, "rounds": rounds}


def counterfactual(truth: np.ndarray, num_classes: int, stride: int = 10) -> pd.DataFrame:
    """Plan section 10: replay Dawid-Skene offline on the MV arm's own recorded annotations.

    The MV arm never ran the estimator, so this is the only way to ask whether DS would have hurt
    a run that was not already being steered by DS labels -- the DS arms' annotations are
    downstream of DS pseudo-labels and cannot answer it.
    """
    cached = CACHE / "counterfactual.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    settings = DawidSkeneSettings.from_config(
        _resolved_config("ds_only")
    )
    records = []
    for path in audit_files("mv"):
        rnd = int(path.stem.rsplit("_", 1)[1])
        if rnd % stride and rnd != 1:
            continue
        z = np.load(path)
        mv_lab, mv_valid = z["global_labels"].astype(np.int64), z["valid_mask"]
        fit = fit_dawid_skene(z["annotations"], num_classes, mv_lab, settings=settings)
        ds_lab = fit.candidate_labels if fit.candidate_labels is not None else fit.labels
        ds_valid = fit.candidate_valid_mask if fit.candidate_valid_mask is not None else fit.valid_mask
        both = mv_valid & ds_valid
        records.append(
            {
                "round": rnd,
                "status": fit.status,
                "iterations": fit.iterations,
                "converged": bool(fit.converged),
                "mv_accuracy": _acc(mv_lab, truth, mv_valid),
                "ds_replay_accuracy": _acc(ds_lab.astype(np.int64), truth, ds_valid),
                "disagreement": float((mv_lab[both] != ds_lab[both]).mean()) if both.any() else np.nan,
                "ds_accuracy_on_disagreement": _acc(
                    ds_lab.astype(np.int64), truth, both & (mv_lab != ds_lab)
                ),
                "mv_accuracy_on_disagreement": _acc(mv_lab, truth, both & (mv_lab != ds_lab)),
                "diagonal_fraction": fit.diagonal_fraction,
                "majority_agreement": fit.majority_agreement,
            }
        )
    frame = pd.DataFrame(records)
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cached)
    return frame


class _Config:
    """Minimal duck-type for ``DawidSkeneSettings.from_config`` -- it only reads attributes."""

    def __init__(self, mapping: dict):
        for key, value in mapping.items():
            setattr(self, key.replace("-", "_"), value)


def _resolved_config(arm: str) -> _Config:
    import yaml

    return _Config(yaml.safe_load((run_dir(arm) / "resolved_config.yaml").read_text()))


# --------------------------------------------------------------------------------------- tables


def write_table(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / f"{name}.csv", index=False)
    # Hand-rolled rather than ``to_markdown``: that pulls in ``tabulate``, a dependency this repo
    # does not have and does not need for a pipe-separated table.
    def cell(v):
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)

    header = list(frame.columns)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n")


def primary_table(metrics, audit) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        m, a = metrics[arm], audit[arm]
        tail = m.tail(LAST)
        rows.append(
            {
                "arm": LABELS[arm],
                "accuracy_last10": tail.accuracy.mean(),
                "macro_f1_last10": tail.macro_f1.mean(),
                "weighted_f1_last10": tail.weighted_f1.mean(),
                "accuracy_r200": m.accuracy.iloc[-1],
                "macro_f1_r200": m.macro_f1.iloc[-1],
                "curve_area_accuracy": m.accuracy.mean(),
                "curve_area_macro_f1": m.macro_f1.mean(),
                "pseudo_label_accuracy_last10": a.tail(LAST).pseudo_accuracy.mean(),
                "coverage_last10": a.tail(LAST).coverage.mean(),
                "best_accuracy_reference_only": m.accuracy.max(),
            }
        )
    return pd.DataFrame(rows)


def per_class_table(per_class, names) -> pd.DataFrame:
    frames = []
    for arm in ARMS:
        p = per_class[arm]
        tail = p[p["round"] > p["round"].max() - LAST]
        agg = tail.groupby("class")[["recall", "f1", "precision"]].mean()
        agg.columns = [f"{c}_{arm}" for c in agg.columns]
        frames.append(agg)
    out = pd.concat(frames, axis=1).reset_index()
    out.insert(1, "class_name", [names[c] for c in out["class"]])
    for metric in ("recall", "f1"):
        out[f"{metric}_ds_only_minus_mv"] = out[f"{metric}_ds_only"] - out[f"{metric}_mv"]
        out[f"{metric}_hybrid_minus_mv"] = out[f"{metric}_hybrid"] - out[f"{metric}_mv"]
    return out


def ds_table(metrics, audit) -> pd.DataFrame:
    rows = []
    for arm in ("ds_only", "hybrid"):
        m, a = metrics[arm], audit[arm]
        counts = m.ds_status.value_counts()
        rows.append(
            {
                "arm": LABELS[arm],
                "ds_applied_rounds": int(m.ds_applied.sum()),
                "fallback_rounds": int(len(m) - m.ds_applied.sum()),
                "converged_rounds": int(m.ds_converged.sum()),
                "iterations_mean": m.ds_iterations.mean(),
                "iterations_max": m.ds_iterations.max(),
                "alignment_identity_rounds": int(m.ds_alignment_identity.sum()),
                "alignment_score_mean": m.ds_alignment_score.mean(),
                "diagonal_fraction_mean": m.ds_diagonal_fraction.mean(),
                "reference_diagonal_fraction_mean": m.ds_reference_diagonal_fraction.mean(),
                "majority_agreement_mean": m.ds_majority_agreement.mean(),
                "disagreement_mean": m.ds_disagreement_rate.mean(),
                "ds_accuracy_on_disagreement": a.ds_accuracy_on_disagreement.mean(),
                "mv_accuracy_on_disagreement": a.mv_accuracy_on_disagreement.mean(),
                "excluded_clients_mean": m.ds_excluded_clients.mean(),
                "status_counts": json.dumps(
                    {STATUS_NAME.get(int(k), int(k)): int(v) for k, v in counts.items()}
                ),
            }
        )
    return pd.DataFrame(rows)


def client_table(rel, assign) -> pd.DataFrame:
    out = assign[["client_id", "device_id", "num_examples"]].copy()
    out["private_classes"] = assign["classes"].map(len)
    out["has_mirai"] = assign["classes"].map(lambda cs: any(c >= 6 for c in cs))
    for arm in ARMS:
        correct, observed = rel[arm]["correct"], rel[arm]["observed"]
        c, o = correct[:, -LAST:, :].sum(axis=(1, 2)), observed[:, -LAST:, :].sum(axis=(1, 2))
        out[f"reliability_{arm}"] = np.where(o > 0, c / np.maximum(o, 1), np.nan)
        out[f"coverage_{arm}"] = o / (LAST * 8900)
    return out


# -------------------------------------------------------------------------------------- figures


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=160)
    plt.close(fig)
    print("  wrote", name)


def _smooth(series: pd.Series, window: int = 10) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def fig_curves(metrics) -> None:
    for idx, (column, title) in enumerate(
        [("accuracy", "Test accuracy"), ("macro_f1", "Macro-F1")], start=1
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        for arm in ARMS:
            m = metrics[arm]
            ax.plot(m["round"], m[column], color=COLORS[arm], alpha=0.25, lw=1)
            ax.plot(
                m["round"],
                _smooth(m[column]),
                color=COLORS[arm],
                lw=2,
                label=f"{LABELS[arm]} (10-round mean)",
            )
        ax.set_xlabel("communication round")
        ax.set_ylabel(title)
        ax.set_title(f"{title} over 200 rounds -- Scenario 3 (raw + 10-round moving average)")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(fig, f"fig{idx:02d}_{column}_curves.png")


def fig_last10(metrics) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.35
    x = np.arange(len(ARMS))
    for offset, (column, label) in zip(
        (-width / 2, width / 2), [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")]
    ):
        vals = [metrics[a].tail(LAST)[column].mean() for a in ARMS]
        errs = [metrics[a].tail(LAST)[column].std() for a in ARMS]
        bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=4, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x, [LABELS[a] for a in ARMS])
    ax.set_ylabel("mean over rounds 191-200")
    ax.set_title("Primary and secondary metric (last 10 rounds, +/- 1 sd)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig03_last10_comparison.png")


def fig_per_class(table, names) -> None:
    for idx, metric in zip((4, 5), ("recall", "f1")):
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(names))
        width = 0.27
        for offset, arm in zip((-width, 0.0, width), ARMS):
            ax.bar(x + offset, table[f"{metric}_{arm}"], width, label=LABELS[arm], color=COLORS[arm])
        ax.set_xticks(x, names, rotation=45, ha="right")
        ax.set_ylabel(f"mean {metric} (rounds 191-200)")
        ax.set_title(f"Per-class {metric}, last 10 rounds")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        _save(fig, f"fig{idx:02d}_per_class_{metric}.png")


def fig_pseudo(audit) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for arm in ARMS:
        a = audit[arm]
        ax.plot(a["round"], a.pseudo_accuracy, color=COLORS[arm], alpha=0.25, lw=1)
        ax.plot(a["round"], _smooth(a.pseudo_accuracy), color=COLORS[arm], lw=2, label=LABELS[arm])
    ax.set_xlabel("communication round")
    ax.set_ylabel("accuracy of broadcast pseudo-labels")
    ax.set_title("Broadcast pseudo-label accuracy on the 8,900 open samples")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "fig06_pseudo_label_accuracy.png")


def fig_disagreement(metrics, audit) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for arm in ("ds_only", "hybrid"):
        a = audit[arm]
        axes[0].plot(a["round"], a.candidate_disagreement, color=COLORS[arm], lw=1.4, label=LABELS[arm])
        axes[1].plot(
            a["round"], _smooth(a.ds_accuracy_on_disagreement), color=COLORS[arm], lw=2,
            label=f"{LABELS[arm]}: DS label",
        )
        axes[1].plot(
            a["round"], _smooth(a.mv_accuracy_on_disagreement), color=COLORS[arm], lw=2,
            ls="--", label=f"{LABELS[arm]}: MV label",
        )
    axes[0].set_ylabel("fraction of samples where MV != DS")
    axes[0].set_title("MV-DS disagreement rate")
    axes[1].set_ylabel("accuracy on disagreeing samples")
    axes[1].set_xlabel("communication round")
    axes[1].set_title("Who is right where they disagree (10-round mean)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    _save(fig, "fig07_mv_ds_disagreement.png")


def fig_ties(audit) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for arm in ARMS:
        a = audit[arm]
        axes[0].plot(a["round"], a.tie_count, color=COLORS[arm], lw=1.2, label=LABELS[arm])
        axes[1].plot(a["round"], _smooth(a.tie_accuracy), color=COLORS[arm], lw=2, label=LABELS[arm])
    axes[0].set_ylabel("tied open samples")
    axes[0].set_title("Vote ties per round")
    axes[1].set_ylabel("accuracy on tied samples")
    axes[1].set_xlabel("communication round")
    axes[1].set_title("Pseudo-label accuracy on tied samples (10-round mean)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    _save(fig, "fig08_ties.png")


def fig_ds_iterations(metrics) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for arm in ("ds_only", "hybrid"):
        m = metrics[arm]
        ax.plot(m["round"], m.ds_iterations, color=COLORS[arm], lw=1.2, label=LABELS[arm])
    cap = _resolved_config("ds_only").dawid_skene_max_iterations
    ax.axhline(cap, color="k", ls=":", label=f"max_iterations = {cap}")
    ax.set_xlabel("communication round")
    ax.set_ylabel("EM iterations")
    ax.set_title("Dawid-Skene EM iterations per round")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "fig09_ds_iterations.png")


def fig_ds_timeline(metrics) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    present = sorted({int(v) for arm in ("ds_only", "hybrid") for v in metrics[arm].ds_status.unique()})
    for ax, arm in zip(axes, ("ds_only", "hybrid")):
        m = metrics[arm]
        for code in present:
            hit = m[m.ds_status == code]
            ax.scatter(
                hit["round"], [present.index(code)] * len(hit), s=12,
                label=f"{STATUS_NAME.get(code, code)} ({len(hit)})",
            )
        ax.set_yticks(range(len(present)), [STATUS_NAME.get(c, c) for c in present], fontsize=8)
        ax.set_title(f"{LABELS[arm]}: Dawid-Skene status per round")
        ax.grid(alpha=0.3, axis="x")
        ax.legend(fontsize=7, loc="center right")
    axes[1].set_xlabel("communication round")
    _save(fig, "fig10_ds_status_timeline.png")


def fig_hybrid_selection(metrics) -> None:
    m = metrics["hybrid"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    axes[0].fill_between(m["round"], 0, m.ds_applied, step="mid", color=COLORS["hybrid"], alpha=0.6)
    axes[0].set_yticks([0, 1], ["fell back to MV", "used DS"])
    axes[0].set_xlabel("communication round")
    axes[0].set_title(
        f"Hybrid label source per round -- DS in {int(m.ds_applied.sum())}/{len(m)} rounds"
    )
    axes[0].grid(alpha=0.3, axis="x")
    fallback = m[m.ds_applied == 0].ds_status.value_counts()
    if len(fallback):
        names = [STATUS_NAME.get(int(k), int(k)) for k in fallback.index]
        axes[1].barh(names, fallback.to_numpy(), color="#888888")
        axes[1].bar_label(axes[1].containers[0], padding=3)
        axes[1].set_xlabel("rounds")
    else:
        axes[1].text(0.5, 0.5, "no fallback rounds", ha="center", va="center")
        axes[1].set_axis_off()
    axes[1].set_title("Hybrid fallback reasons")
    _save(fig, "fig11_hybrid_selection_and_fallback.png")


def fig_private_counts(assign, names) -> None:
    grid = np.zeros((len(assign), len(names)))
    for i, counts in enumerate(assign["counts"]):
        for k, v in counts.items():
            grid[i, k] = v
    fig, ax = plt.subplots(figsize=(8, 12))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(0, len(assign), 4), assign["client_id"][::4], fontsize=6)
    ax.set_title("Scenario 3: private sample count per client x class")
    fig.colorbar(im, ax=ax, label="private samples")
    _save(fig, "fig12_client_class_private_counts.png")


def fig_reliability(rel, assign, names) -> None:
    for arm in ARMS:
        correct, observed, rounds = rel[arm]["correct"], rel[arm]["observed"], rel[arm]["rounds"]
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(observed > 0, correct / observed, np.nan)
        fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True, sharey=True)
        for k, ax in enumerate(axes.ravel()):
            if k >= len(names):
                ax.set_axis_off()
                continue
            im = ax.imshow(
                rate[:, :, k], aspect="auto", cmap="magma", vmin=0, vmax=1,
                extent=(rounds[0], rounds[-1], len(assign), 0),
            )
            ax.set_title(names[k], fontsize=10)
        fig.suptitle(
            f"{LABELS[arm]}: per-class client x round reliability "
            "(fraction of that class's open samples the client labelled correctly)"
        )
        fig.supxlabel("communication round")
        fig.supylabel("client (sorted by client_id)")
        fig.colorbar(im, ax=axes, label="reliability", shrink=0.7)
        fig.savefig(FIGS / f"fig13_reliability_{arm}.png", dpi=140)
        plt.close(fig)
        print("  wrote", f"fig13_reliability_{arm}.png")


def fig_classes_vs_reliability(clients) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in ARMS:
        ax.scatter(
            clients["private_classes"] + np.random.default_rng(0).normal(0, 0.06, len(clients)),
            clients[f"reliability_{arm}"],
            s=22, alpha=0.6, color=COLORS[arm], label=LABELS[arm],
        )
    ax.set_xlabel("number of classes present in the client's private data")
    ax.set_ylabel("pooled reliability, rounds 191-200")
    ax.set_title("Private class coverage vs. annotation reliability")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "fig14_private_classes_vs_reliability.png")


def fig_devices(rel, assign, names, devices=(3, 7)) -> None:
    mirai = [k for k, n in enumerate(names) if n.startswith("mirai")]
    fig, axes = plt.subplots(len(devices), 1, figsize=(10, 4 * len(devices)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, device in zip(axes, devices):
        rows = assign.index[assign["device_id"] == device].to_numpy()
        has = sorted({c for cs in assign.loc[rows, "classes"] for c in cs if c in mirai})
        for arm in ARMS:
            correct = rel[arm]["correct"][rows][:, :, mirai].sum(axis=(0, 2))
            observed = rel[arm]["observed"][rows][:, :, mirai].sum(axis=(0, 2))
            with np.errstate(invalid="ignore"):
                series = np.where(observed > 0, correct / np.maximum(observed, 1), np.nan)
            ax.plot(rel[arm]["rounds"], series, color=COLORS[arm], lw=1.6, label=LABELS[arm])
        present = ", ".join(names[c] for c in has) or "none"
        ax.set_title(
            f"Device {device} ({len(rows)} clients): reliability on Mirai classes "
            f"-- Mirai classes in their private data: {present}"
        )
        ax.set_ylabel("pooled reliability")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("communication round")
    _save(fig, "fig15_device_mirai_behaviour.png")


def fig_counterfactual(cf) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(cf["round"], cf.mv_accuracy, "o-", color=COLORS["mv"], label="MV labels (actually used)")
    axes[0].plot(cf["round"], cf.ds_replay_accuracy, "s-", color=COLORS["ds_only"], label="DS replay on the same annotations")
    axes[0].set_ylabel("pseudo-label accuracy")
    axes[0].set_title("Counterfactual: Dawid-Skene replayed offline on the MV arm's annotations")
    axes[1].plot(cf["round"], cf.mv_accuracy_on_disagreement, "o--", color=COLORS["mv"], label="MV right")
    axes[1].plot(cf["round"], cf.ds_accuracy_on_disagreement, "s--", color=COLORS["ds_only"], label="DS right")
    axes[1].set_ylabel("accuracy where they disagree")
    axes[1].set_xlabel("communication round")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    _save(fig, "fig16_counterfactual_ds_on_mv_run.png")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    truth = open_truth()
    names = class_names()
    num_classes = len(names)
    assign = assignments()
    metrics = {a: pd.read_parquet(run_dir(a) / "metrics.parquet") for a in ARMS}
    per_class = {a: pd.read_parquet(run_dir(a) / "per_class_metrics.parquet") for a in ARMS}
    audit = {a: audit_series(a, truth, num_classes) for a in ARMS}
    rel = {a: reliability(a, truth, num_classes) for a in ARMS}
    cf = counterfactual(truth, num_classes)

    primary = primary_table(metrics, audit)
    classes = per_class_table(per_class, names)
    clients = client_table(rel, assign)
    write_table(primary, "table01_primary")
    write_table(classes, "table02_per_class_last10")
    write_table(ds_table(metrics, audit), "table03_ds_diagnostics")
    write_table(cf, "table04_counterfactual_ds_on_mv")
    write_table(clients, "table05_client_reliability")

    fig_curves(metrics)
    fig_last10(metrics)
    fig_per_class(classes, names)
    fig_pseudo(audit)
    fig_disagreement(metrics, audit)
    fig_ties(audit)
    fig_ds_iterations(metrics)
    fig_ds_timeline(metrics)
    fig_hybrid_selection(metrics)
    fig_private_counts(assign, names)
    fig_reliability(rel, assign, names)
    fig_classes_vs_reliability(clients)
    fig_devices(rel, assign, names)
    fig_counterfactual(cf)

    json.dump(
        {"arms": {a: run_dir(a).name for a in ARMS}, "num_open": int(len(truth))},
        (OUT / "runs.json").open("w"),
        indent=2,
    )
    print("\n" + primary.to_string(index=False))


if __name__ == "__main__":
    main()
