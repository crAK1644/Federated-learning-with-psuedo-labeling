"""Shared derivations for the streaming Dawid-Skene report (rapor/ds_streaming_raporu.tex).

Three of the four arms were archived off disk and live only on the ``runs-archive`` branch, so
every read goes through :func:`blob`, which falls back to ``git show`` when the file is not in the
working tree. Nothing is restored wholesale: the report needs about 16 MB of the archive's 3.1 GB.

Sealed open-set labels are used here only for offline measurement. They never enter aggregation,
training, or fallback logic (DAWID_SKENE_FEASIBILITY_PLAN.md section 1).
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from ds_headroom import sealed_open_labels  # noqa: E402

ARCHIVE_REF = "runs-archive"
RUNS = REPO / "artifacts" / "runs"
DATA = REPO / "artifacts" / "data"
OUT = REPO / "rapor"
FIGS = OUT / "figures"
# Derivations are cached by name only, so the cache directory is what keeps two partitions apart.
# It is rebound by :func:`retarget` alongside DATA -- never make one of them partition-scoped
# without the other, or a seed-2026 pseudo-label series will silently answer from the 2023 cache.
CACHE = OUT / "cache"

NUM_CLASSES = 11
NUM_OPEN = 8900
ABSTAIN = -1
LAST = 20  # primary metric = mean over the final 20 rounds

# run dir + attempt id per arm, for the seed-2023 partition (manifest a97311fe...). Only the
# streaming arm is still on disk; the rest come from ``runs-archive``. For any other partition call
# :func:`retarget`, which finds the run directories instead of hardcoding them -- their names end in
# a hash over config + manifest + commit + seed, so they cannot be written down before the run.
ARMS = {
    "stream": ("ssfl-s3-ds_streaming_rtx3090_200-811eb2e459f92099", "3126416-ade0647270ea"),
    "mv": ("ssfl-s3-experiment1_s3_mv_200-79e618b3d32e06ee", "2037537-f9f496bb63c5"),
    "ds_only": ("ssfl-s3-experiment1_s3_ds_only_200-b09be0a7fd6fcfe2", "3394735-1e75ad8a1832"),
    "hybrid": ("ssfl-s3-experiment1_s3_hybrid_200-1f17d410103f5438", "295457-e57b5a4c7295"),
}

# blue = MV (the control everything is measured against), orange = the new streaming arm,
# red = batch DS-only, green = hybrid. Same hexes as the Deney 1 report so the two PDFs read alike.
COLORS = {"mv": "#1f77b4", "stream": "#ff7f0e", "ds_only": "#d62728", "hybrid": "#2ca02c"}
LABELS = {
    "mv": "Majority Vote",
    "stream": "Akan DS (online-EM)",
    "ds_only": "Toplu DS (batch)",
    "hybrid": "Hybrid",
}
ORDER = ["stream", "mv", "ds_only", "hybrid"]

# the structurally unresolvable pair: no client votes correctly on both sides, so no vote-based
# aggregator can separate them (DENEY_1_SCENARIO_3_DENEY_PLANI.md, reachability analysis).
UNREACHABLE_PAIR = (4, 5)  # gafgyt.tcp, gafgyt.udp

# arm key -> the ``profile`` value the run recorded, i.e. the matrix entry name. Stable across
# partitions by design: the seed-2026 matrix reuses the seed-2023 entry names, because these are the
# same two arms on new data rather than new arms.
PROFILES = {
    "mv": "experiment1_s3_mv_200",
    "stream": "ds_streaming_rtx3090_200",
    "ds_only": "experiment1_s3_ds_only_200",
    "hybrid": "experiment1_s3_hybrid_200",
}


def _find_run(profile: str, seed: int, data_name: str) -> tuple[str, str]:
    """Locate one run directory by what it recorded, not by its name.

    A run directory is ``ssfl-s<scenario>-<profile>-<run_id>`` where run_id hashes config + dataset
    manifest + git commit + seed, so it is not predictable before the run finishes. Matching on the
    resolved config instead means a rerun (new commit, new hash, same experiment) still resolves.

    ``seed`` alone is not a sufficient key: the same seed on two partitions would match twice, which
    is exactly the confusion this whole exercise exists to avoid. The partition directory name
    disambiguates, and an ambiguous match is an error rather than a silent first-hit.
    """
    import yaml

    hits = []
    for cfg_path in sorted(RUNS.glob(f"ssfl-s*-{profile}-*/resolved_config.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text())
        if cfg.get("profile") != profile or cfg.get("seed") != seed:
            continue
        if Path(str(cfg.get("data_path", ""))).name != data_name:
            continue
        hits.append(cfg_path.parent)
    if not hits:
        raise SystemExit(
            f"no run found for profile={profile!r} seed={seed} data={data_name!r} under {RUNS}. "
            "Has it finished, and is this the machine it ran on?"
        )
    if len(hits) > 1:
        listed = ", ".join(h.name for h in hits)
        raise SystemExit(f"ambiguous: {len(hits)} runs match {profile!r} seed={seed}: {listed}")
    run = hits[0]
    pointer = run / "current_attempt.json"
    if not pointer.is_file():
        raise SystemExit(f"{run.name} has no current_attempt.json; cannot locate its audit files")
    return run.name, json.loads(pointer.read_text())["attempt_id"]


def retarget(
    data: Path, seed: int, arms: tuple[str, ...] = ("stream", "mv"), *, require_labels: bool = True
) -> bool:
    """Point every derivation in this module at a different partition's runs.

    The module's defaults describe the seed-2023 pair. This rebinds DATA (which is where the sealed
    open-set labels come from -- scoring seed-2026 pseudo-labels against the 2023 truth would
    produce plausible-looking nonsense), CACHE (so the two partitions' cached derivations cannot
    overwrite each other), ARMS, and ORDER.

    ``arms`` defaults to the two-arm replicate. The seed-2023 matrix also ran ds_only and hybrid;
    later partitions do not, so asking for them raises rather than plotting an absent arm.

    Returns whether the partition itself is on disk. Runs are small enough to copy off the training
    box on their own; the partition is not, so a caller that only reads run outputs (metrics,
    confusion matrices) can pass ``require_labels=False`` and skip the derivations that need the
    sealed labels rather than failing outright.
    """
    global DATA, CACHE, ARMS, ORDER

    data = Path(data)
    have_labels = (data / "dataset_manifest.json").is_file()
    if require_labels and not have_labels:
        raise SystemExit(f"{data} is not a prepared dataset directory")
    unknown = set(arms) - set(PROFILES)
    if unknown:
        raise SystemExit(f"unknown arm(s): {sorted(unknown)}")

    DATA = data
    CACHE = OUT / f"cache-{data.name}"
    ARMS = {arm: _find_run(PROFILES[arm], seed, data.name) for arm in arms}
    ORDER = [arm for arm in ORDER if arm in ARMS]
    return have_labels


def blob(arm: str, relpath: str) -> bytes:
    """File contents for one arm, from the working tree if present, else from ``runs-archive``."""
    run = ARMS[arm][0]
    local = RUNS / run / relpath
    if local.is_file():
        return local.read_bytes()
    done = subprocess.run(
        ["git", "show", f"{ARCHIVE_REF}:artifacts/runs/{run}/{relpath}"],
        cwd=REPO,
        capture_output=True,
    )
    if done.returncode != 0:
        raise SystemExit(f"{arm}: cannot read {relpath}\n{done.stderr.decode()[:400]}")
    return done.stdout


def audit(arm: str, rnd: int) -> dict[str, np.ndarray]:
    """One round's aggregation audit. ``np.load`` needs a seekable handle, so buffer it."""
    attempt = ARMS[arm][1]
    raw = blob(arm, f"attempts/{attempt}/aggregation_audit/ssfl_aggregation_round_{rnd}.npz")
    with np.load(io.BytesIO(raw)) as npz:
        return {k: npz[k] for k in npz.files}


def metrics(arm: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(blob(arm, "metrics.parquet")))


def per_class(arm: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(blob(arm, "per_class_metrics.parquet")))


def communication(arm: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(blob(arm, "communication.parquet")))


def confusion(arm: str, rnd: int) -> np.ndarray:
    with np.load(io.BytesIO(blob(arm, "confusion_matrices.npz"))) as npz:
        return npz[f"round_{rnd}"]


def num_rounds(arm: str) -> int:
    """Rounds this arm actually completed. Cheaper than trusting a literal that a rerun invalidates."""
    return int(metrics(arm)["round"].max())


def summary(arm: str) -> dict:
    return json.loads(blob(arm, "summary.json"))


def config(arm: str) -> dict:
    import yaml

    return yaml.safe_load(blob(arm, "resolved_config.yaml").decode())


def class_names() -> list[str]:
    # Every run copies its partition's label map into dataset_manifest.json, so class names stay
    # readable when only the runs were copied off the training box and the partition itself was not.
    path = DATA / "label_map.json"
    mapping = (
        json.loads(path.read_text())
        if path.is_file()
        else json.loads(blob(ORDER[0], "dataset_manifest.json"))["label_map"]
    )
    return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


def truth() -> np.ndarray:
    return sealed_open_labels(DATA)


def cached(name: str, build):
    """Derive once, reuse. The audit scans are the only expensive part of the whole report."""
    path = CACHE / f"{name}.npz"
    if path.is_file():
        with np.load(path) as npz:
            return {k: npz[k] for k in npz.files}
    CACHE.mkdir(parents=True, exist_ok=True)
    result = build()
    np.savez_compressed(path, **result)
    return result


def pseudo_label_series(arm: str = "stream") -> dict[str, np.ndarray]:
    """Per round: how good were the DS labels, and how good would Majority Vote have been?

    Both are scored on the same items -- the intersection of the two valid masks -- so the delta is
    the aggregator and not a difference in which open samples each one was willing to label. This
    is exactly the rule ``ds_streaming_replay.replay`` uses offline, so the two are comparable.
    """

    def build():
        sealed = truth()
        last = num_rounds(arm)
        ds_acc, mv_acc, scored_n = [], [], []
        for rnd in range(1, last + 1):
            a = audit(arm, rnd)
            ds_lab, ds_mask = a["dawid_skene_labels"], a["dawid_skene_valid_mask"].astype(bool)
            mv_lab, mv_mask = a["majority_labels"], a["majority_valid_mask"].astype(bool)
            scored = ds_mask & mv_mask & (mv_lab != ABSTAIN)
            ds_acc.append(float((ds_lab[scored] == sealed[scored]).mean()))
            mv_acc.append(float((mv_lab[scored] == sealed[scored]).mean()))
            scored_n.append(int(scored.sum()))
        return {
            "round": np.arange(1, last + 1),
            "ds": np.array(ds_acc),
            "mv": np.array(mv_acc),
            "scored": np.array(scored_n),
        }

    return cached(f"pseudo_{arm}", build)


def vote_quality(arm: str) -> dict[str, np.ndarray]:
    """Majority Vote applied to *that arm's own* votes -- a yardstick for the client ensemble.

    Holding the aggregator fixed at Majority Vote turns the comparison into a question about the
    votes themselves: did this arm's clients become better annotators?
    """

    def build():
        sealed = truth()
        last = num_rounds(arm)
        acc, valid = [], []
        for rnd in range(1, last + 1):
            a = audit(arm, rnd)
            if "majority_labels" in a:  # a DS arm records the counterfactual majority itself
                lab, mask = a["majority_labels"], a["majority_valid_mask"].astype(bool)
            else:  # the MV arm's broadcast IS the majority result
                lab, mask = a["global_labels"], a["valid_mask"].astype(bool)
            acc.append(float((lab[mask] == sealed[mask]).mean()))
            valid.append(float(mask.mean()))
        return {"round": np.arange(1, last + 1), "accuracy": np.array(acc), "valid": np.array(valid)}

    return cached(f"votes_{arm}", build)


def offline_replay(arm: str = "mv") -> dict[str, np.ndarray]:
    """Replay the production streaming estimator over the MV arm's recorded votes.

    This is the pre-registration prediction: the aggregator is swapped but the training trajectory
    is held fixed, because the votes are the ones the MV arm actually produced. The gap between
    this and the closed-loop measurement is the report's main finding.
    """

    def build():
        from ssfl.protocols.dawid_skene import DawidSkeneSettings, fit_dawid_skene

        sealed = truth()
        settings = DawidSkeneSettings(
            confusion_prior="diagonal",
            confusion_prior_diagonal=0.95,
            confusion_pseudocount=20.0,
            state_decay=0.9,
            state_init_rounds=150.0,
            class_alignment=True,
        )
        last = num_rounds(arm)
        state = None
        ds_acc, mv_acc = [], []
        for rnd in range(1, last + 1):
            a = audit(arm, rnd)
            votes = a["annotations"]
            mv_lab = a.get("majority_labels", a["global_labels"]).astype(np.int64)
            mv_mask = a.get("majority_valid_mask", a["valid_mask"]).astype(bool)
            senders = tuple(f"c{j:03d}" for j in range(votes.shape[0]))
            fit = fit_dawid_skene(
                votes,
                num_classes=NUM_CLASSES,
                majority_labels=mv_lab,
                settings=settings,
                senders=senders,
                state=state,
            )
            if fit.state is not None:
                state = fit.state
            labels = fit.candidate_labels if fit.numerically_valid else mv_lab
            mask = fit.candidate_valid_mask if fit.numerically_valid else mv_mask
            scored = mask & mv_mask & (mv_lab != ABSTAIN)
            ds_acc.append(float((labels[scored] == sealed[scored]).mean()))
            mv_acc.append(float((mv_lab[scored] == sealed[scored]).mean()))
        return {"round": np.arange(1, last + 1), "ds": np.array(ds_acc), "mv": np.array(mv_acc)}

    return cached(f"replay_{arm}", build)


def last_mean(frame: pd.DataFrame, column: str, last: int = LAST) -> float:
    return float(frame.sort_values("round")[column].tail(last).mean())


def per_class_last(arm: str, last: int = LAST) -> pd.Series:
    frame = per_class(arm)
    cut = frame["round"].max() - last
    return frame[frame["round"] > cut].groupby("class")["f1"].mean()


def error_split(arm: str, rnd: int = 200) -> dict[str, int]:
    """Split the test-set errors into the ones no vote-based aggregator can fix, and the rest."""
    cm = confusion(arm, rnd)
    a, b = UNREACHABLE_PAIR
    total = int(cm.sum() - np.trace(cm))
    unreachable = int(cm[a, b] + cm[b, a])
    return {
        "total": total,
        "unreachable": unreachable,
        "reachable": total - unreachable,
        "samples": int(cm.sum()),
    }
