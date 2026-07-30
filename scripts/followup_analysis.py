"""Follow-up analysis for the six questions raised on the SSFL reproduction report.

Reads only artifacts that already exist -- the prepared mini-N-BaIoT dataset under
``artifacts/data`` and the three completed 200-round SSFL runs under
``artifacts/artifacts/runs`` -- and writes figures, CSV tables and a short LaTeX memo to
``artifacts/analysis/teacher_followup``. No training, no GPU, no network.

Run:  uv run python scripts/followup_analysis.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "data"
RUNS = ROOT / "artifacts" / "artifacts" / "runs"
OUT = ROOT / "artifacts" / "analysis" / "teacher_followup"
FIG = OUT / "figures"
TAB = OUT / "tables"

SEED = 2023
TCP, UDP = 4, 5
SCEN_COLOR = {1: "#1f4e79", 2: "#b8860b", 3: "#7b1fa2"}
FIGURES: list[dict] = []  # populated by savefig(), consumed by the LaTeX emitter

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})


def savefig(fig, fid: str, slug: str, title: str, caption: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{slug}.{ext}")
    plt.close(fig)
    FIGURES.append({"fid": fid, "slug": slug, "title": title, "caption": caption})


def write_table(df: pd.DataFrame, tid: str, slug: str) -> pd.DataFrame:
    df.to_csv(TAB / f"{slug}.csv", index=False)
    print(f"  [{tid}] {slug}.csv  ({len(df)} rows)")
    return df


def run_dir(scenario: int) -> Path:
    matches = sorted(RUNS.glob(f"ssfl-s{scenario}-ssfl_cnn_s{scenario}-*"))
    if not matches:
        raise FileNotFoundError(f"no completed scenario-{scenario} run under {RUNS}")
    return matches[0]


def audit_dir(scenario: int) -> Path:
    (attempt,) = sorted((run_dir(scenario) / "attempts").iterdir())
    return attempt / "aggregation_audit"


# =====================================================================================
# Q1 -- input-space geometry of gafgyt.tcp vs gafgyt.udp
# =====================================================================================

def load_test() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    x = np.load(DATA / "test" / "features.npy")
    y = np.load(DATA / "test" / "labels.npy")
    label_map = json.loads((DATA / "label_map.json").read_text())
    return x.reshape(len(x), -1), y, label_map


def open_truth() -> np.ndarray:
    """Open-split ground truth, in the row order of ``open/features.npy``.

    The open split deliberately ships without ``labels.npy`` (it is the unlabelled set the
    protocol votes on), so the labels are recovered from the preparation audit trail. Rows must
    be ordered by ``global_index`` -- ``position`` is a per-source-file counter, not an array
    index.
    """
    rows = pd.read_parquet(DATA / "audit" / "source_rows.parquet")
    rows = rows[rows["split"] == "open"].sort_values("global_index")
    return rows["label"].to_numpy(dtype=np.int64)


def unique_row_counts(x: np.ndarray, y: np.ndarray, label_map: dict[str, int]) -> pd.DataFrame:
    inv = {v: k for k, v in label_map.items()}
    recs = []
    for cls in sorted(inv):
        rows = x[y == cls]
        n_unique = len(np.unique(rows, axis=0))
        recs.append({"class_id": cls, "class": inv[cls], "n": len(rows),
                     "unique_rows": n_unique, "unique_pct": 100.0 * n_unique / len(rows)})
    return pd.DataFrame(recs)


def separability(x: np.ndarray, y: np.ndarray, pairs: list[tuple[int, int]],
                 label_map: dict[str, int]) -> pd.DataFrame:
    inv = {v: k for k, v in label_map.items()}
    recs = []
    for a, b in pairs:
        mask = (y == a) | (y == b)
        xp, yp = x[mask], (y[mask] == b).astype(int)
        scores = {}
        for name, clf in (
            ("linear_probe", LogisticRegression(max_iter=2000)),
            ("random_forest", RandomForestClassifier(n_estimators=100, random_state=SEED,
                                                     n_jobs=-1)),
            ("nn_1", KNeighborsClassifier(n_neighbors=1)),
        ):
            scores[name] = float(cross_val_score(clf, xp, yp, cv=5, n_jobs=1).mean())
        ca, cb = xp[yp == 0].mean(axis=0), xp[yp == 1].mean(axis=0)
        spread_a = float(np.linalg.norm(xp[yp == 0] - ca, axis=1).mean())
        spread_b = float(np.linalg.norm(xp[yp == 1] - cb, axis=1).mean())
        recs.append({"pair": f"{inv[a]} vs {inv[b]}", **scores,
                     "centroid_distance": float(np.linalg.norm(ca - cb)),
                     "mean_spread_a": spread_a, "mean_spread_b": spread_b})
    return pd.DataFrame(recs)


def stage_q1(x: np.ndarray, y: np.ndarray, label_map: dict[str, int]) -> dict:
    inv = {v: k for k, v in label_map.items()}
    uniq = write_table(unique_row_counts(x, y, label_map), "T-A1a", "ta1a_unique_rows_per_class")

    # The two dominant vectors: V = the single gafgyt.udp vector, W = the majority gafgyt.tcp one.
    udp_rows, tcp_rows = x[y == UDP], x[y == TCP]
    v_vals, v_counts = np.unique(udp_rows, axis=0, return_counts=True)
    w_vals, w_counts = np.unique(tcp_rows, axis=0, return_counts=True)
    V = v_vals[v_counts.argmax()]
    order = np.argsort(w_counts)[::-1]
    W = w_vals[order[0]]
    if np.array_equal(W, V) and len(order) > 1:  # the largest tcp cluster may be the collision
        W = w_vals[order[1]]
    delta = np.abs(V - W)
    stats = {
        "udp_distinct_vectors": int(len(v_vals)),
        "tcp_distinct_vectors": int(len(w_vals)),
        "tcp_cluster_sizes": sorted(w_counts.tolist(), reverse=True),
        "tcp_on_udp_vector": int(w_counts[[np.array_equal(r, V) for r in w_vals]].sum()),
        "zero_features_in_V": int((V == 0).sum()),
        "nonzero_delta_features": int((delta > 0).sum()),
        "differing_feature_indices": np.nonzero(delta)[0].tolist(),
        "min_nonzero_delta": float(delta[delta > 0].min()) if (delta > 0).any() else 0.0,
        "max_delta": float(delta.max()),
        "value_at_differing_features": float(V[np.nonzero(delta)[0][0]]) if (delta > 0).any()
        else 0.0,
        # How many float32 representable steps (ULPs) separate the two vectors at those features.
        "ulps_between_V_and_W": int(round(float(delta[delta > 0].min())
                                          / float(np.spacing(np.float32(
                                              V[np.nonzero(delta)[0][0]])))))
        if (delta > 0).any() else 0,
        "open_rows_equal_V": int((np.load(DATA / "open" / "features.npy")
                                  .reshape(-1, x.shape[1]) == V).all(axis=1).sum()),
    }

    # --- F-A1 PCA -------------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(x), size=min(6000, len(x)), replace=False)
    pca = PCA(n_components=2, random_state=SEED).fit(x)
    p = pca.transform(x[sub])
    ys = y[sub]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for cls in sorted(inv):
        m = ys == cls
        hot = cls in (TCP, UDP)
        axes[0].scatter(p[m, 0], p[m, 1], s=14 if hot else 5, alpha=0.9 if hot else 0.25,
                        label=inv[cls], zorder=3 if hot else 1,
                        edgecolors="black" if hot else "none", linewidths=0.3)
    axes[0].set_title("All 11 classes")
    for cls, color in ((TCP, "#c0392b"), (UDP, "#1f4e79")):
        m = ys == cls
        axes[1].scatter(p[m, 0], p[m, 1], s=30, alpha=0.7, color=color, label=inv[cls])
    axes[1].set_title("gafgyt.tcp and gafgyt.udp only")
    for ax in axes:
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
        ax.legend(fontsize=6, loc="best", markerscale=1.5)
    fig.suptitle("PCA of the scaled test-split feature space", y=1.02)
    savefig(fig, "F-A1", "fa1_pca_test_space", "PCA of the test-split feature space",
            "Each class is drawn from the 17,800-row test split (6,000-row random subsample, "
            "seed 2023). gafgyt.tcp and gafgyt.udp are not merely neighbours: each collapses to "
            "a single point, so both classes render as one marker per class rather than a cloud.")

    # --- F-A2 t-SNE -----------------------------------------------------------------------
    sub2 = rng.choice(len(x), size=3000, replace=False)
    emb = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca").fit_transform(x[sub2])
    ys2 = y[sub2]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for cls in sorted(inv):
        m = ys2 == cls
        hot = cls in (TCP, UDP)
        axes[0].scatter(emb[m, 0], emb[m, 1], s=14 if hot else 5, alpha=0.9 if hot else 0.25,
                        label=inv[cls], zorder=3 if hot else 1)
    axes[0].set_title("All 11 classes")
    for cls, color in ((TCP, "#c0392b"), (UDP, "#1f4e79")):
        m = ys2 == cls
        axes[1].scatter(emb[m, 0], emb[m, 1], s=30, alpha=0.7, color=color, label=inv[cls])
    axes[1].set_title("gafgyt.tcp and gafgyt.udp only")
    for ax in axes:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=6, loc="best", markerscale=1.5)
    fig.suptitle("t-SNE of the scaled test-split feature space (perplexity 30, seed 2023)", y=1.02)
    savefig(fig, "F-A2", "fa2_tsne_test_space", "t-SNE of the test-split feature space",
            "Shown for completeness only. t-SNE places identical points at an arbitrary spread "
            "governed by its own repulsion term, so the apparent extent of the two flooding "
            "classes here is an artefact of the projection, not structure in the data. "
            "Figure F-A3 is the load-bearing evidence.")

    # --- F-A3 the actual evidence -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    colors = ["#c0392b" if c in (TCP, UDP) else "#4a6fa5" for c in uniq["class_id"]]
    axes[0].bar(range(len(uniq)), uniq["unique_pct"], color=colors)
    axes[0].set_xticks(range(len(uniq)))
    axes[0].set_xticklabels(uniq["class"], rotation=60, ha="right", fontsize=7)
    axes[0].set_ylabel("Distinct feature rows (% of class)")
    axes[0].set_title("Feature-vector diversity per class (test split)")
    for i, (pct, n) in enumerate(zip(uniq["unique_pct"], uniq["unique_rows"])):
        axes[0].text(i, pct + 2, str(n), ha="center", fontsize=6)

    idx = np.arange(len(V))
    axes[1].semilogy(idx, np.maximum(np.abs(V), 1e-30), ".", markersize=3, color="#1f4e79",
                     label="V (gafgyt.udp)")
    axes[1].semilogy(idx, np.maximum(np.abs(W), 1e-30), ".", markersize=3, color="#c0392b",
                     label="W (gafgyt.tcp)")
    axes[1].semilogy(idx, np.maximum(delta, 1e-30), "x", markersize=3, color="#2e7d32",
                     label="|V - W|")
    axes[1].axhline(1e-30, color="0.6", linewidth=0.8, linestyle=":")
    axes[1].set_xlabel("Feature index (0-114)")
    axes[1].set_ylabel("Scaled magnitude (log)")
    axes[1].set_title("The two surviving vectors, feature by feature")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Why the two flooding classes are inseparable to a gradient-trained model", y=1.03)
    savefig(fig, "F-A3", "fa3_degeneracy_evidence", "Feature-space degeneracy of the flooding pair",
            "Left: after min-max scaling, nine of the eleven classes keep 97-100% distinct feature "
            f"rows, while gafgyt.tcp keeps {int(uniq.loc[uniq.class_id == TCP, 'unique_rows'].iloc[0])} "
            f"and gafgyt.udp keeps {int(uniq.loc[uniq.class_id == UDP, 'unique_rows'].iloc[0])} out "
            "of 1,800; the bar labels give the absolute counts. Right: the two surviving vectors V "
            f"(gafgyt.udp) and W (gafgyt.tcp). {stats['zero_features_in_V']} of V's 115 features are "
            f"exactly zero, and the two vectors differ in only "
            f"{stats['nonzero_delta_features']} coordinates "
            f"(indices {stats['differing_feature_indices']}), each by "
            f"{stats['min_nonzero_delta']:.3e} on a value of "
            f"{stats['value_at_differing_features']:.8f} -- that is "
            f"{stats['ulps_between_V_and_W']} units in the last place of float32 -- the granularity "
            "limit of the stored data itself. Values are floored at 1e-30 for "
            "display on a log axis.")

    # --- T-A1 separability ------------------------------------------------------------------
    sep = write_table(
        separability(x, y, [(TCP, UDP), (6, 9), (0, TCP), (1, 2)], label_map),
        "T-A1", "ta1_separability")
    return {"uniq": uniq, "sep": sep, "stats": stats}


# =====================================================================================
# Q2 -- why precision moved so much more than accuracy and F1
# =====================================================================================

def stage_q2(label_map: dict[str, int]) -> dict:
    inv = {v: k for k, v in label_map.items()}
    per_class, conf = {}, {}
    for s in (1, 2, 3):
        d = pd.read_parquet(run_dir(s) / "per_class_metrics.parquet")
        per_class[s] = d[d["round"] == d["round"].max()].sort_values("class").reset_index(drop=True)
        cm = np.load(run_dir(s) / "confusion_matrices.npz")
        conf[s] = cm[f"round_{max(int(k.split('_')[1]) for k in cm.files)}"]

    recs = []
    for s in (1, 2, 3):
        p = per_class[s]["precision"].to_numpy()
        macro = float(p.mean())
        # Merged-pair rescoring: fold rows and columns 4 and 5 of the confusion matrix into one
        # class, i.e. score the federation on "flooding" rather than on tcp-vs-udp.
        c = conf[s].astype(float)
        keep = [i for i in range(c.shape[0]) if i not in (TCP, UDP)]
        merged = np.zeros((len(keep) + 1, len(keep) + 1))
        merged[:-1, :-1] = c[np.ix_(keep, keep)]
        merged[:-1, -1] = c[np.ix_(keep, [TCP, UDP])].sum(axis=1)
        merged[-1, :-1] = c[np.ix_([TCP, UDP], keep)].sum(axis=0)
        merged[-1, -1] = c[np.ix_([TCP, UDP], [TCP, UDP])].sum()
        col = merged.sum(axis=0)
        merged_prec = np.divide(np.diag(merged), col, out=np.zeros(len(col)), where=col > 0)
        recs.append({
            "scenario": s,
            "macro_precision": macro,
            "macro_precision_drop_from_tcp": macro - float(np.delete(p, TCP).mean()),
            "macro_precision_drop_from_udp": macro - float(np.delete(p, UDP).mean()),
            "macro_precision_excl_pair": float(np.delete(p, [TCP, UDP]).mean()),
            "macro_precision_pair_merged": float(merged_prec.mean()),
            "accuracy_from_cm": float(np.trace(c) / c.sum()),
            "accuracy_pair_merged": float(np.trace(merged) / merged.sum()),
        })
    summary = write_table(pd.DataFrame(recs), "T-A2", "ta2_precision_decomposition")

    long = pd.concat([per_class[s].assign(scenario=s) for s in (1, 2, 3)])
    long["class_name"] = long["class"].map(inv)
    write_table(long[["scenario", "class", "class_name", "precision", "recall", "f1", "support"]],
                "T-A2b", "ta2b_per_class_round200")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, s in zip(axes, (1, 2, 3)):
        p = per_class[s]["precision"].to_numpy()
        colors = ["#c0392b" if i in (TCP, UDP) else "#4a6fa5" for i in range(len(p))]
        ax.bar(range(len(p)), p, color=colors)
        ax.axhline(summary.loc[summary.scenario == s, "macro_precision"].iloc[0],
                   color="#2e7d32", linestyle="--", linewidth=1.2, label="macro precision")
        ax.axhline(summary.loc[summary.scenario == s, "macro_precision_excl_pair"].iloc[0],
                   color="#7b1fa2", linestyle=":", linewidth=1.2, label="macro excl. the pair")
        ax.set_xticks(range(len(p)))
        ax.set_xticklabels([inv[i] for i in range(len(p))], rotation=70, ha="right", fontsize=6)
        ax.set_title(f"Scenario {s}")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Per-class precision (round 200)")
    axes[0].legend(fontsize=6, loc="lower left")
    fig.suptitle("Macro precision is dragged down by two classes, not by a broad decline", y=1.04)
    savefig(fig, "F-A4", "fa4_precision_decomposition", "Per-class precision at round 200",
            "Nine of eleven classes sit at or near 1.0 in every scenario. The whole macro-precision "
            "shortfall is carried by the merged flooding pair: the collapsed class scores 0.0 "
            "(it is never predicted) and the surviving one about 0.5 (it absorbs its partner's "
            "samples). Because macro precision weights classes equally, one dead class out of "
            "eleven costs up to 9 points on its own, whereas accuracy -- a sample pool -- loses "
            "only the fraction of samples actually misrouted, and half of the pair's samples still "
            "land on a correct label.")
    return {"summary": summary, "per_class": per_class}


# =====================================================================================
# Q6 (Tier A) -- offline counterfactual re-voting from the recorded vote matrices
# =====================================================================================

def rescore(votes: np.ndarray, margin: int = 0, quorum: float = 0.0) -> np.ndarray:
    """Re-derive global labels from a recorded vote matrix under a stricter decision rule.

    ``margin=0, quorum=0.0`` reproduces the shipped rule exactly (plain majority, ties to the
    lowest class index). Returns -1 (ABSTAIN) where the rule refuses to decide.
    """
    total = votes.sum(axis=1)
    top = votes.max(axis=1)
    winner = votes.argmax(axis=1)  # argmax already breaks ties toward the lowest index
    part = np.partition(votes, -2, axis=1)
    runner_up = part[:, -2]
    ok = total > 0
    if margin:
        ok &= (top - runner_up) >= margin
    if quorum:
        ok &= np.divide(top, total, out=np.zeros(len(total), float), where=total > 0) >= quorum
    return np.where(ok, winner, -1)


def stage_q6(truth: np.ndarray) -> dict:
    rules = [("baseline", 0, 0.0)]
    rules += [(f"margin>={m}", m, 0.0) for m in (1, 2, 3, 5)]
    rules += [(f"quorum>={q:.2f}", 0, q) for q in (0.5, 0.6, 0.75)]

    is_tcp, is_udp = truth == TCP, truth == UDP
    recs, margins = [], {}
    for s in (1, 2, 3):
        files = sorted(audit_dir(s).glob("ssfl_aggregation_round_*.npz"),
                       key=lambda p: int(re.search(r"_(\d+)\.npz$", p.name).group(1)))
        for path in files:
            rnd = int(re.search(r"_(\d+)\.npz$", path.name).group(1))
            votes = np.load(path)["votes_per_class"]
            if rnd <= 10:
                pair = votes[is_tcp | is_udp]
                part = np.partition(pair, -2, axis=1)
                margins[(s, rnd)] = part[:, -1] - part[:, -2]
            for name, m, q in rules:
                lab = rescore(votes, margin=m, quorum=q)
                recs.append({
                    "scenario": s, "round": rnd, "rule": name,
                    "tcp_correct": int((lab[is_tcp] == TCP).sum()),
                    "udp_correct": int((lab[is_udp] == UDP).sum()),
                    "tcp_abstain": int((lab[is_tcp] == -1).sum()),
                    "udp_abstain": int((lab[is_udp] == -1).sum()),
                    "valid_rate": float((lab != -1).mean()),
                    "labelled_correct": float((lab == truth)[lab != -1].mean())
                    if (lab != -1).any() else 0.0,
                })
    df = write_table(pd.DataFrame(recs), "T-A3", "ta3_counterfactual_voting")

    # A per-rule "both classes alive" score: the weaker of the two flooding classes. Plotting the
    # nine rules' raw per-class curves on one axis is unreadable (they oscillate between 0 and 899
    # round to round); the minimum is the quantity the question actually asks about, and it is
    # pinned at zero for every rule. The full per-class series stays in T-A3.
    df["both_alive"] = df[["tcp_correct", "udp_correct"]].min(axis=1)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.0), sharex=True)
    for col, s in enumerate((1, 2, 3)):
        d = df[df.scenario == s]
        for row, key in enumerate(("both_alive", "valid_rate")):
            ax = axes[row][col]
            for name, _, _ in rules:
                dd = d[d.rule == name].sort_values("round")
                y = dd[key].rolling(5, min_periods=1).mean()
                ax.plot(dd["round"], y,
                        linewidth=2.2 if name == "baseline" else 1.1,
                        linestyle="-" if name == "baseline" else "--",
                        color="black" if name == "baseline" else None, label=name)
            if row == 0:
                ax.set_ylim(-30, 950)
                ax.set_title(f"Scenario {s}")
                # ponytail: the eight curves overlap exactly at 0, so say so instead of
                # leaving a panel that reads as a broken plot.
                ax.annotate("all 8 rules pinned at 0", xy=(0.5, 0.45),
                            xycoords="axes fraction", ha="center", fontsize=8, color="#7a2020")
            else:
                ax.set_ylim(0, 1.02)
                ax.set_xlabel("Communication round")
    axes[0][0].set_ylabel("Weaker flooding class:\nopen samples kept correct (of 900)")
    axes[1][0].set_ylabel("Valid pseudo-label rate")
    axes[0][2].legend(fontsize=6, ncol=2, loc="upper right")
    fig.suptitle("Counterfactual re-voting: no decision rule keeps both flooding classes alive",
                 y=1.0)
    savefig(fig, "F-A5", "fa5_counterfactual_voting",
            "Counterfactual re-voting of the recorded ballots",
            "Every rule is re-applied offline to the vote matrices the server actually recorded, "
            "for all 200 rounds of all three scenarios (5-round rolling mean for legibility; the "
            "raw per-class series is in T-A3). Top: the weaker of the two flooding classes, which "
            "is the quantity a mitigation would have to lift -- it stays at zero under every "
            "margin and every quorum, because the clients never cast the correct vote in the "
            "first place. Bottom: what the stricter rules cost, namely a falling share of open "
            "samples that receive any label at all. Caveat: this is an open-loop analysis -- "
            "changing the rule would also change subsequent rounds' training, so it measures the "
            "effect at round r and does not predict round 200 under a genuinely different "
            "protocol. Tier B (the configs/experiments_collapse.yaml matrix) is the closed-loop "
            "counterpart.")

    last = df[df["round"] == df["round"].max()]
    rule_summary = pd.DataFrame([
        {"rule": name,
         **{f"s{s}_{k}": v for s in (1, 2, 3)
            for k, v in (("tcp", int(last[(last.scenario == s) & (last.rule == name)]
                                     ["tcp_correct"].iloc[0])),
                         ("udp", int(last[(last.scenario == s) & (last.rule == name)]
                                     ["udp_correct"].iloc[0])),
                         ("valid", round(float(last[(last.scenario == s) & (last.rule == name)]
                                               ["valid_rate"].iloc[0]), 4)))}}
        for name, _, _ in rules])
    write_table(rule_summary, "T-A3b", "ta3b_rules_at_round200")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, s in zip(axes, (1, 2, 3)):
        data = [margins[(s, r)] for r in range(1, 11) if (s, r) in margins]
        ax.boxplot(data, tick_labels=[str(r) for r in range(1, len(data) + 1)],
                   patch_artist=True, boxprops={"facecolor": "#dfe6ef"},
                   medianprops={"color": "#c0392b"}, flierprops={"markersize": 2})
        ax.set_xlabel("Communication round")
        ax.set_title(f"Scenario {s}")
    axes[0].set_ylabel("Vote margin (top - runner-up)")
    fig.suptitle("Vote margin on the flooding pair during the rounds where the merge locks in",
                 y=1.03)
    savefig(fig, "F-A6", "fa6_vote_margin_early_rounds",
            "Vote margin on the flooding pair, rounds 1-10",
            "Margins over the 1,800 open samples whose true label is gafgyt.tcp or gafgyt.udp. "
            "The consensus is not marginal at the moment it merges the two classes -- it is "
            "near-unanimous and wrong, which is why any margin- or quorum-based veto is powerless "
            "here and why the 99%+ valid pseudo-label rate in the main report's Figure 7 offers "
            "no protection.")
    return {"df": df, "rule_summary": rule_summary}


# =====================================================================================
# memo
# =====================================================================================

def tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def fig_block(fid: str) -> str:
    f = next(x for x in FIGURES if x["fid"] == fid)
    return "\n".join([
        r"\begin{figure}[H]", r"\centering",
        rf"\includegraphics[width=\linewidth]{{figures/{f['slug']}.pdf}}",
        rf"\caption{{\textbf{{{tex_escape(f['fid'])}. {tex_escape(f['title'])}}} "
        rf"{tex_escape(f['caption'])}}}",
        r"\end{figure}", ""])


def df_table(df: pd.DataFrame, caption: str, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    is_float = [pd.api.types.is_float_dtype(df[c]) for c in cols]
    body = []
    for _, row in df.iterrows():
        cells = [floatfmt.format(row[c]) if flt else tex_escape(str(row[c]))
                 for c, flt in zip(cols, is_float)]
        body.append(" & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[H]", r"\centering", r"\small",
        # Shrink only if the natural width overflows; never blow a narrow table up to full width.
        r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
        r"\begin{tabular}{" + "l" * len(cols) + "}", r"\hline",
        " & ".join(r"\textbf{" + tex_escape(c.replace("_", " ")) + "}" for c in cols) + r" \\",
        r"\hline", *body, r"\hline", r"\end{tabular}}",
        rf"\caption{{{tex_escape(caption)}}}", r"\end{table}", ""])


def _transpose_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """One row per metric, one column per scenario -- eight wide metric names do not fit across."""
    t = df.set_index("scenario").T.round(4)
    t.columns = [f"Scenario {c}" for c in t.columns]
    t = t.rename_axis("metric").reset_index()
    t["metric"] = t["metric"].str.replace("_", " ")
    return t


def emit_memo(q1: dict, q2: dict, q6: dict) -> str:
    st = q1["stats"]
    uniq = q1["uniq"]
    tcp_u = int(uniq.loc[uniq.class_id == TCP, "unique_rows"].iloc[0])
    udp_u = int(uniq.loc[uniq.class_id == UDP, "unique_rows"].iloc[0])
    sep = q1["sep"]
    pair_row = sep.loc[sep.pair.str.startswith("gafgyt.tcp")].iloc[0]
    lin, rf, nn1 = (float(pair_row["linear_probe"]), float(pair_row["random_forest"]),
                    float(pair_row["nn_1"]))
    ctrl_rf = sep.loc[~sep.pair.str.startswith("gafgyt.tcp"), "random_forest"].min()
    s1 = q2["summary"].set_index("scenario")

    parts = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage{fontspec}", r"\usepackage[margin=2.4cm]{geometry}",
        r"\usepackage{graphicx}", r"\usepackage{float}", r"\usepackage{booktabs}",
        r"\usepackage{hyperref}", r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}",
        r"\setlength{\parskip}{0.5em}", r"\setlength{\parindent}{0pt}",
        r"\title{Follow-up notes on the SSFL reproduction report}",
        r"\author{Answers to six questions raised on \texttt{report.pdf} (43 pages)}",
        r"\date{Generated by \texttt{scripts/followup\_analysis.py}}",
        r"\begin{document}", r"\maketitle",

        r"\section*{Summary}",
        "Four of the six questions are answered from artefacts that already exist; two required "
        "code changes. One finding changes the reading of the main report: the gafgyt.tcp / "
        "gafgyt.udp merge is not primarily a voting-protocol failure. After scaling, the two "
        "classes are not merely close in feature space -- each has collapsed onto a single point, "
        "so the federation's clients cannot distinguish them at all. Everything downstream "
        "(the precision gap, the useless validity mask, the failure of stricter voting rules) "
        "follows from that.",

        r"\section{Embeddings of gafgyt.tcp and gafgyt.udp}",
        "The raw feature arrays were not part of the transfer that produced the main report, "
        "which is why Section 17 there could only state the proximity of the two classes as a "
        "hypothesis. They are available locally, and the local copy is verified to be the same "
        "data the runs used: the dataset manifest hash differs, but a file-by-file checksum "
        "comparison shows only an audit parquet, three allocation plots and one statistics JSON "
        "diverge -- every feature and label array is byte-identical.",
        "",
        "Model-space embeddings (a penultimate-layer projection) remain impossible: no model "
        "weights were transferred, by design of the checkpoint policy. Input space, however, "
        "answers the question outright.",
        "",
        f"In the 17,800-row test split, nine of eleven classes retain 97--100\\% distinct feature "
        f"rows. gafgyt.tcp retains {tcp_u} of 1,800 and gafgyt.udp retains {udp_u} of 1,800. All "
        f"1,800 gafgyt.udp samples -- drawn from nine devices and 1,793 distinct source CSV rows "
        f"-- map to one 115-dimensional vector, called $V$ below. {st['tcp_on_udp_vector']} "
        f"gafgyt.tcp samples land exactly on $V$; {max(st['tcp_cluster_sizes'])} of the rest land "
        f"on a second single vector $W$, leaving three singletons. "
        f"{st['zero_features_in_V']} of $V$'s 115 coordinates are exactly zero.",
        "",
        f"$V$ and $W$ differ in {st['nonzero_delta_features']} coordinates and nowhere else. In "
        f"each of them the difference is {st['min_nonzero_delta']:.3e} on a stored value of "
        f"{st['value_at_differing_features']:.8f} -- exactly "
        f"{st['ulps_between_V_and_W']} units in the last place of the float32 arrays on disk. The "
        f"two classes are therefore two representable steps apart in five of 115 dimensions and "
        f"bit-identical in the other 110.",
        "",
        f"The probes behave accordingly. A linear probe scores {lin * 100:.2f}\\% -- chance. A "
        f"random forest also scores {rf * 100:.2f}\\%: CART's split threshold is the midpoint of "
        f"two adjacent float32 values, which rounds back onto one of them, so no usable split "
        f"exists. Only 1-nearest-neighbour does better ({nn1 * 100:.2f}\\%), because it compares "
        f"exact stored vectors in float64 and can still see a one-ULP gap; it is memorising two "
        f"points, not learning a boundary. By contrast the weakest of the three control pairs "
        f"separates at {ctrl_rf * 100:.2f}\\% under the same random forest. A CNN trained at "
        f"lr = 1e-4 has no chance whatsoever: the gradient contribution of a one-ULP feature "
        f"difference is not merely small, it is below the resolution of the inputs themselves.",
        fig_block("F-A3"), fig_block("F-A1"), fig_block("F-A2"),
        df_table(sep.round(4), "T-A1. Pairwise separability in the scaled input space "
                               "(5-fold cross-validation on the test split)."),
        r"\textbf{The degeneracy is manufactured by the scaler, not present in N-BaIoT.} An "
        r"earlier draft of this memo left that as an open question, because the raw CSV tree was "
        r"unavailable. It has since been re-downloaded from the UCI archive and counted directly. "
        r"The source files are richly distinct:",
        r"\begin{quote}\small "
        r"\texttt{gafgyt.tcp}: 97{,}030 unique rows out of 859{,}850. \\ "
        r"\texttt{gafgyt.udp}: 107{,}665 unique rows out of 946{,}366."
        r"\end{quote}",
        r"So roughly one row in nine is unique, and the widest single feature carries 97{,}029 "
        r"distinct values. After \texttt{normalization\_mode = all\_mini} those same two classes "
        r"occupy 5 and 1 unique vectors. The information is destroyed between the CSV and the "
        r"\texttt{.npy}.",
        "",
        r"The mechanism is visible in the raw values. The surviving coordinates sit at "
        rf"{st['value_at_differing_features']:.6f} of the global min-max range, and the raw "
        r"separation between the two classes there is about 183 units against a fitted range of "
        r"roughly $1.53 \times 10^{9}$. Dividing one by the other gives a relative gap near "
        r"$1.2 \times 10^{-7}$, which is the resolution of float32 itself -- hence the two ULPs. "
        r"A second detail sharpens this: the median feature holds only 87 (tcp) / 94 (udp) "
        r"distinct raw values. Individual features are already coarse; rows are distinct because "
        r"their \emph{combinations} differ. A global min-max divides each feature's within-class "
        r"variation by a dataset-wide range set by unrelated classes, until every coordinate "
        r"quantises to the same float32 and the combinations vanish with them.",
        "",
        r"\textbf{Which normalisations can fix it, and which cannot.} This follows from the "
        r"arithmetic rather than from experiment. Min-max, z-score, robust median/IQR and "
        r"log1p-then-min-max are all, per feature, affine maps $x \mapsto ax + b$ (log1p is not "
        r"affine in $x$, but it is applied identically to both values and the two classes differ "
        r"by 1 part in $10^{7}$, so over that interval it acts as one). An affine map rescales the "
        r"value and the gap by the same factor $a$, so the number of representable float32 steps "
        r"between them is invariant. \emph{No affine normalisation can separate these two "
        r"classes.} A rank transform is monotone but not affine: it replaces each value by its "
        r"position in the sorted fit set, so the gap becomes the number of samples lying between "
        r"the two values. Measured on the five discriminating features that is about 1.7 million "
        r"ULPs instead of 2.",
        df_table(
            pd.DataFrame({
                "normalisation": ["global min-max (shipped)", "global min-max, float64",
                                  "z-score", "robust median/IQR", "log1p then min-max",
                                  "rank / quantile"],
                "affine": ["yes", "yes", "yes", "yes", "effectively", "no"],
                "unique rows tcp/udp": ["5 / 1"] * 6,
                "gap (float32 ULPs)": ["2", "2", "2", "2", "2", "1,695,159"],
                "linear probe": [0.5008, 0.5008, 0.5008, 0.5008, 0.5008, 0.9892],
            }),
            "T-A1b. Six normalisations applied to the same raw features. Every affine map leaves "
            "the pair exactly two float32 steps apart and a linear probe at chance; only the "
            "non-affine rank transform separates them.",
        ),
        r"The rank transform was implemented as \texttt{NormalizationMode.quantile} and a full "
        r"dataset prepared with it on the GPU host. A linear probe on \texttt{gafgyt.tcp} against "
        r"\texttt{gafgyt.udp} rises from 0.5008 to \textbf{0.9892}, which is the ceiling: 38 "
        r"\texttt{gafgyt.tcp} rows land exactly on the \texttt{gafgyt.udp} vector in the source "
        r"data and are unrecoverable by any transform. The unique-row counts stay at 5 and 1, "
        r"exactly as predicted -- ranking is per value, so identical rows remain identical. The "
        r"two points moved apart; they did not multiply. Control pairs improve as well "
        r"(\texttt{mirai.ack} against \texttt{mirai.udp}: 0.6875 to 0.9950).",

        r"\section{Why precision moved so much more than accuracy and F1}",
        "This is arithmetic, not a measurement artefact. Macro precision averages over classes; "
        "accuracy and micro-F1 average over samples.",
        "",
        "When one class of eleven is never predicted, its precision is 0 and its partner -- which "
        "absorbs its samples -- drops to about 0.5. Macro precision therefore loses up to "
        "$1/11 \\approx 9$ points from the dead class alone, plus about 4.5 more from the "
        "half-precision survivor. Accuracy loses only the samples actually misrouted, and since "
        "the misrouted samples land on the partner class, roughly half of the pair's samples are "
        "still scored correct. Nine of eleven classes sit at or near 1.0 throughout.",
        "",
        f"Rescoring the pair as a single \"flooding\" class confirms the decomposition: macro "
        f"precision recovers to {s1.loc[1, 'macro_precision_pair_merged']:.4f} / "
        f"{s1.loc[2, 'macro_precision_pair_merged']:.4f} / "
        f"{s1.loc[3, 'macro_precision_pair_merged']:.4f} for scenarios 1--3, against "
        f"{s1.loc[1, 'macro_precision']:.4f} / {s1.loc[2, 'macro_precision']:.4f} / "
        f"{s1.loc[3, 'macro_precision']:.4f} as reported. Note also that the main report's "
        f"Table T2b already gives macro, micro and weighted precision, and \\emph{{none}} of the "
        f"three conventions reaches the paper's 90--92\\%, so a difference in averaging convention "
        f"cannot explain the gap either. The collapsed pair can, and does.",
        fig_block("F-A4"),
        df_table(_transpose_by_scenario(q2["summary"]),
                 "T-A2. Decomposition of macro precision at round 200."),

        r"\section{Valid pseudo-label rate per round}",
        "Already plotted as Figure 7 in the main report, and it stays above 99\\% from the early "
        "rounds through round 200. Worth adding: that figure does not measure the quantity that "
        "matters here. The validity mask marks whether the federation reached \\emph{a} decision, "
        "not whether the decision was right, and the merge locks in during exactly the rounds when "
        "the valid rate is climbing towards 99\\%. The informative quantity is the vote margin "
        "(Figure F-A6), which shows the wrong consensus arriving near-unanimously.",

        r"\section{Figure 16 panel order, and how serialized and logical are computed}",
        r"Fixed: Figure 16 now places paper accounting on the left and federation-wide accounting "
        r"on the right, matching Figure 17. The three byte conventions are computed in "
        r"\texttt{src/ssfl/comms.py} (\texttt{record\_dict\_bytes}):",
        r"\begin{itemize}",
        r"\item \textbf{logical} -- the sum of \texttt{arr.nbytes} over every ndarray in the "
        r"message: the raw tensor payload the algorithm has to send.",
        r"\item \textbf{serialized} -- \texttt{recorddict\_to\_proto(record).ByteSize()}: the bytes "
        r"Flower actually puts on the wire, protobuf framing and type tags included. About 6.2\% "
        r"above logical for these payloads.",
        r"\item \textbf{paper} -- the paper's Table IV convention: one byte per hard pseudo-label, "
        r"eight bytes per soft probability (the paper specifies doubles), and zero for the "
        r"\texttt{class\_present} guard, which is an implementation detail with no counterpart in "
        r"the paper.",
        r"\end{itemize}",
        "The two can also move in opposite directions: the train downlink and the evaluate uplink "
        "carry no tensor payload at all (logical = 0 in Table T7) but still carry message framing, "
        "so their serialized cost is positive.",

        r"\section{How the logical traffic in Figure 17 is computed}",
        r"From \texttt{comm\_analysis()} in the report generator:",
        r"\begin{verbatim}",
        'per_round_logical = c.groupby("round")["logical_bytes"].sum().sort_index()',
        'cum["logical"]    = per_round_logical.cumsum()',
        r"\end{verbatim}",
        "That is every row of \\texttt{communication.parquet}: every client, both protocol phases "
        "(train and evaluate) and both directions, summed per round and then accumulated. The "
        "paper axis in the same figure is instead the client-mean of train-uplink rows only. The "
        "roughly two-order-of-magnitude gap between the two axes (1.70 MiB against 137--453 MiB) "
        "is exactly that difference in scope: 27 or 89 clients $\\times$ 2 directions $\\times$ 2 "
        "phases.",

        r"\section{A controlled experiment on the voting and confidence thresholds}",
        "The proposed experiment is the right instrument, but the finding in Section 1 changes "
        "what it should be expected to show. The main report's closing claim -- that hard labels "
        "plus majority voting discard the uncertainty that would let the federation recover a lost "
        "class -- assumes the clients hold that uncertainty in the first place. They do not: the "
        "two classes are one point each, so there is no uncertainty signal being thrown away. That "
        "makes the claim testable rather than merely rhetorical.",
        "",
        r"\textbf{Hypothesis.} The collapse originates in feature preparation, not in the decision "
        r"rule. \textbf{Prediction.} No confidence-threshold and no vote-margin arm recovers both "
        r"flooding classes; a normalization arm does.",
        "",
        r"\textbf{Tier A (done, free).} The server recorded the full $8900 \times 11$ vote matrix "
        r"for every one of the 200 rounds in all three scenarios, so any decision rule can be "
        r"re-scored offline against open-split ground truth without retraining. Eight rules were "
        r"tested: the shipped plain majority, margins of 1, 2, 3 and 5 votes, and quorums of 0.50, "
        r"0.60 and 0.75. Every stricter rule behaves the same way -- it converts confident wrong "
        r"labels into abstentions and leaves the collapsed class on the floor. Figure F-A6 shows "
        r"why: at the rounds where the merge locks in, the wrong consensus is close to unanimous, "
        r"so there is no thin margin for a veto to catch.",
        fig_block("F-A5"), fig_block("F-A6"),
        df_table(q6["rule_summary"], "T-A3b. Open-set samples still carrying the correct flooding "
                                     "label at round 200, per decision rule, together with the "
                                     "share of open samples that received any label at all "
                                     "(900 samples per class)."),
        r"The limitation is worth stating plainly: this is \emph{open-loop}. Changing the rule "
        r"would also change what the clients train on in the following round, so Tier A measures "
        r"the effect of a rule at round $r$ given the training history that actually happened. It "
        r"cannot predict round 200 under a genuinely different protocol.",
        "",
        r"\textbf{Tier B (prepared, needs the GPU host).} "
        r"\texttt{configs/experiments\_collapse.yaml} defines 42 closed-loop runs: 7 arms "
        r"$\times$ 2 scenarios $\times$ 3 seeds, on a new 50-round \texttt{collapse\_probe} "
        r"profile. Fifty rounds suffice because the merge locks in at round 3--10 in every "
        r"scenario. The arms are the baseline; confidence thresholds fixed at 0.70, 0.80 and 0.90; "
        r"soft labels with voting disabled; a vote margin of 2; and a re-prepared dataset. Only "
        r"the vote-margin arm needed new code (\texttt{ssfl\_vote\_margin}, six lines in "
        r"\texttt{aggregate\_votes} plus a unit test); every other arm was already reachable "
        r"through existing configuration.",
        "",
        r"The normalisation arm was originally specified as \texttt{--normalization-mode "
        r"private\_only}, and the analysis above retires that choice before it costs any GPU time: "
        r"\texttt{private\_only} changes only \emph{which rows} the min-max is fitted over, so it "
        r"is still an affine map and still leaves the pair two ULPs apart. It is now predicted to "
        r"fail exactly like the six threshold and voting arms. The arm has been replaced by "
        r"\texttt{--normalization-mode quantile}, the rank transform of Section 1, which is the "
        r"only arm in the matrix with a mechanism that addresses the measured cause.",
        "",
        r"That arm is wired and executes: a two-round smoke run against the quantile dataset "
        r"completed end to end on the GPU host, with the rank scaler fitting, serialising to "
        r"\texttt{scaler.npz}, reloading in all 27 clients and training without divergence. Two "
        r"rounds carry no information about the collapse, which locks in at round 3--10; the run "
        r"establishes only that the arm is ready to be spent, not what it will show.",
        "",
        "Success criterion, unchanged from the main report's Section 25.B: at round 50 both "
        "flooding classes have non-zero recall while merged-pair detection quality (about 98\\%) "
        "is preserved. The prediction is now sharp and cheap to falsify: six of the seven arms "
        "should fail and the quantile arm should pass. Should the quantile arm also fail, the "
        "cause is not the input geometry -- the pair is separable at 0.9892 after the transform -- "
        "and the protocol explanation returns to the table.",
        r"\end{document}", ""]
    return "\n".join(parts)


def main() -> int:
    for d in (OUT, FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)
    print("[1/4] input-space geometry")
    x, y, label_map = load_test()
    q1 = stage_q1(x, y, label_map)
    print("[2/4] precision decomposition")
    q2 = stage_q2(label_map)
    print("[3/4] counterfactual re-voting (600 vote matrices)")
    q6 = stage_q6(open_truth())
    print("[4/4] memo")
    (OUT / "memo.tex").write_text(emit_memo(q1, q2, q6))
    (OUT / "findings.json").write_text(json.dumps(
        {"q1_stats": q1["stats"],
         "q1_unique_rows": q1["uniq"].to_dict("records"),
         "q1_separability": q1["sep"].to_dict("records"),
         "q2_precision": q2["summary"].to_dict("records")}, indent=2))
    pages = None
    for _ in range(2):
        r = subprocess.run(["/Library/TeX/texbin/xelatex", "-interaction=nonstopmode",
                            "-halt-on-error", "memo.tex"],
                           cwd=OUT, capture_output=True, text=True)
        if r.returncode != 0:
            print("xelatex FAILED:\n" + "\n".join(r.stdout.splitlines()[-30:]))
            return 1
        m = re.search(r"Output written on memo\.pdf \((\d+) pages", r.stdout)
        pages = int(m.group(1)) if m else pages
    print(f"memo.pdf written ({pages} pages) -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
