"""Stage 4: pick the target class pair on real N-BaIoT data, with evidence.

The follow-up plan is explicit that "most confused pair" is the wrong selection rule. A pair no
model can separate is not a positive control for Dawid-Skene -- if the features do not carry the
distinction, better label aggregation cannot invent it. What the experiment needs is a pair that
is *hard but learnable*: linear models struggle, non-linear models succeed. A pair that defeats
every model is still useful, as the negative control.

So this measures, for each candidate pair:

* how a linear probe, a random forest and a 1-nearest-neighbour classifier confuse the two, both
  inside the full 11-class problem and head-to-head;
* how many samples exist per class in each split;
* how many scenario-1 clients actually hold each class -- Dawid-Skene needs several clients
  annotating a class before it can estimate anything about it;
* what the first round of a real federated run actually did to the pair, taken from the recorded
  shadow run's aggregation audit.

Everything is fit on the private split and evaluated on the sealed test split. Nothing here runs
during a federated run; the open-set ground truth used at the end is read from the data-prep audit
trail, which the server never sees.

    uv run python scripts/probe_class_pairs.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "artifacts" / "data"
OUTPUT = REPO_ROOT / "artifacts" / "validation" / "class_pair_probe.json"
SHADOW_RUN = REPO_ROOT / "artifacts" / "runs" / "ssfl-s1-ds50_shadow-45eb87b9e6757088"
SEED = 20_260_820

# Named in the plan, on the strength of earlier informal probes. This script exists to check them.
CANDIDATES = {
    "positive": ("gafgyt.combo", "gafgyt.junk"),
    "negative": ("gafgyt.tcp", "gafgyt.udp"),
}

# Pre-registered before looking at any output. A pair is only a usable positive control if a
# non-linear model can actually tell the classes apart while a linear one cannot -- that gap is
# the room in which a better label aggregator can show a difference at all.
CRITERIA = {
    "positive_min_nonlinear_accuracy": 0.95,
    "positive_max_linear_accuracy": 0.90,
    "positive_min_clients_per_class": 3,
    "positive_min_open_support": 200,
    "negative_max_nonlinear_accuracy": 0.80,
}

PROBES = {
    "linear": lambda: LogisticRegression(max_iter=2000, random_state=SEED),
    "forest": lambda: RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=SEED, min_samples_leaf=2
    ),
    "one_nn": lambda: KNeighborsClassifier(n_neighbors=1, n_jobs=-1),
}
NONLINEAR_PROBE = "forest"


# --- data ---------------------------------------------------------------------------------------


def load_label_map() -> dict[str, int]:
    return json.loads((DATA_ROOT / "label_map.json").read_text())


def load_private_pool() -> tuple[np.ndarray, np.ndarray]:
    """Every private sample, pooled across clients. Class separability is a property of the
    features, not of the partition, so the non-IID split is irrelevant here and is measured
    separately by ``client_class_coverage``."""
    features, labels = [], []
    for path in sorted((DATA_ROOT / "private").glob("*.npz")):
        label = int(path.stem.split("_")[1])
        npz = np.load(path)
        features.append(npz["features_flat"])
        labels.append(np.full(len(npz["features_flat"]), label, dtype=np.int64))
    return np.concatenate(features), np.concatenate(labels)


def load_test() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.load(DATA_ROOT / "test" / "features.npy"),
        np.load(DATA_ROOT / "test" / "labels.npy").astype(np.int64),
    )


def load_open_truth() -> np.ndarray:
    """Open-set ground truth, recovered from the data-prep audit trail.

    This is evaluation-only data. The server never loads it and the estimator never sees it; it is
    here so a pseudo-label confusion matrix can be scored offline.
    """
    rows = pd.read_parquet(DATA_ROOT / "audit" / "source_rows.parquet")
    rows = rows[rows["split"] == "open"]
    index = rows["global_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(index), np.arange(len(index))):
        raise SystemExit("open-set audit rows are not a permutation of the open indices")
    truth = np.empty(len(index), dtype=np.int64)
    truth[index] = rows["label"].to_numpy(dtype=np.int64)
    return truth


def feature_degeneracy(features: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    """How many genuinely distinct feature rows each class still has after preprocessing.

    A class whose 1,800 samples collapse to a handful of distinct rows is not a hard class, it is
    a destroyed one: no aggregator, no model and no amount of federation can separate what the
    stored features no longer distinguish. This is the difference between "these two attacks look
    alike" and "the pipeline threw the difference away", and only the first is interesting.
    """
    result = {}
    for index in range(num_classes):
        rows = features[labels == index]
        if not len(rows):
            continue
        result[index] = {
            "samples": int(len(rows)),
            "distinct_rows": int(len({row.tobytes() for row in rows})),
            "mean_feature_std": float(rows.std(axis=0).mean()),
        }
    return result


# --- pair statistics ----------------------------------------------------------------------------


def pair_confusion_mass(confusion: np.ndarray) -> dict[tuple[int, int], float]:
    """Symmetric confusion between every class pair, as a fraction of the two classes' support.

    Raw off-diagonal counts favour whichever class happens to be larger, so each pair is
    normalised by the support it could have been drawn from.
    """
    support = confusion.sum(axis=1)
    masses = {}
    for a, b in itertools.combinations(range(confusion.shape[0]), 2):
        total = support[a] + support[b]
        masses[(a, b)] = float(confusion[a, b] + confusion[b, a]) / total if total else 0.0
    return masses


def head_to_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    pair: tuple[int, int],
) -> dict[str, float]:
    """Each probe refit on just the two classes. A pair can look separable inside the 11-class
    problem only because some third class absorbed the errors, so the binary fit is the one that
    answers whether the features distinguish these two."""
    train = np.isin(y_train, pair)
    test = np.isin(y_test, pair)
    scores = {}
    for name, build in PROBES.items():
        model = build().fit(x_train[train], y_train[train])
        scores[name] = float((model.predict(x_test[test]) == y_test[test]).mean())
    return scores


def client_class_coverage(scenario: int = 1) -> dict[int, int]:
    """Clients holding at least one private sample of each class, in the given scenario."""
    payload = json.loads((DATA_ROOT / "scenarios" / f"{scenario}.json").read_text())
    coverage: dict[int, int] = {}
    for client in payload["clients"]:
        for label, indices in client["class_local_indices"].items():
            if indices:
                coverage[int(label)] = coverage.get(int(label), 0) + 1
    return coverage


def early_round_confusion(truth: np.ndarray, num_classes: int) -> dict:
    """What the first recorded federated round actually produced on the open set.

    Reads the shadow run's aggregation audit rather than re-running anything: the point is what
    the pseudo-labels looked like when the models were still bad, which is exactly when a better
    aggregator would have to earn its keep.
    """
    audits = sorted(SHADOW_RUN.glob("attempts/*/aggregation_audit/ssfl_aggregation_round_*.npz"))
    if not audits:
        return {}
    rounds = {}
    for path in audits:
        server_round = int(path.stem.rsplit("_", 1)[1])
        payload = np.load(path)
        labels = payload["majority_labels"].astype(np.int64)
        valid = payload["majority_valid_mask"]
        rounds[server_round] = {
            "valid_rate": float(valid.mean()),
            "accuracy": float((labels[valid] == truth[valid]).mean()) if valid.any() else 0.0,
            "confusion": confusion_matrix(
                truth[valid], labels[valid], labels=list(range(num_classes))
            ).tolist(),
        }
    return dict(sorted(rounds.items()))


# --- report -------------------------------------------------------------------------------------


def verdict(pair_row: dict, role: str, coverage: dict[int, int], open_support: dict[int, int]):
    """Pre-registered criteria, applied without re-reading the numbers first."""
    a, b = pair_row["labels"]
    checks = []
    if role == "positive":
        checks.append(
            (
                "non-linear separable",
                pair_row["head_to_head"][NONLINEAR_PROBE]
                >= CRITERIA["positive_min_nonlinear_accuracy"],
            )
        )
        checks.append(
            (
                "linear struggles",
                pair_row["head_to_head"]["linear"] <= CRITERIA["positive_max_linear_accuracy"],
            )
        )
        checks.append(
            (
                "client coverage",
                min(coverage.get(a, 0), coverage.get(b, 0))
                >= CRITERIA["positive_min_clients_per_class"],
            )
        )
        checks.append(
            (
                "open-set support",
                min(open_support.get(a, 0), open_support.get(b, 0))
                >= CRITERIA["positive_min_open_support"],
            )
        )
    else:
        checks.append(
            (
                "not learnable",
                pair_row["head_to_head"][NONLINEAR_PROBE]
                <= CRITERIA["negative_max_nonlinear_accuracy"],
            )
        )
    return checks


def run_probe(top_pairs: int = 6) -> dict:
    label_map = load_label_map()
    names = {index: name for name, index in label_map.items()}
    num_classes = len(label_map)

    x_train, y_train = load_private_pool()
    x_test, y_test = load_test()
    truth = load_open_truth()

    support = {
        "private": {int(c): int(n) for c, n in zip(*np.unique(y_train, return_counts=True))},
        "test": {int(c): int(n) for c, n in zip(*np.unique(y_test, return_counts=True))},
        "open": {int(c): int(n) for c, n in zip(*np.unique(truth, return_counts=True))},
    }
    coverage = client_class_coverage()
    degeneracy = feature_degeneracy(x_test, y_test, num_classes)

    multiclass = {}
    masses = {}
    for name, build in PROBES.items():
        model = build().fit(x_train, y_train)
        predicted = model.predict(x_test)
        matrix = confusion_matrix(y_test, predicted, labels=list(range(num_classes)))
        multiclass[name] = {
            "accuracy": float((predicted == y_test).mean()),
            "per_class_recall": (np.diag(matrix) / matrix.sum(axis=1)).tolist(),
            "confusion": matrix.tolist(),
        }
        masses[name] = pair_confusion_mass(matrix)

    ranked = sorted(
        masses[NONLINEAR_PROBE], key=lambda pair: masses[NONLINEAR_PROBE][pair], reverse=True
    )
    named = [tuple(sorted(label_map[n] for n in pair)) for pair in CANDIDATES.values()]
    selected = list(dict.fromkeys(ranked[:top_pairs] + named))

    pairs = []
    for pair in selected:
        pairs.append(
            {
                "labels": list(pair),
                "names": [names[pair[0]], names[pair[1]]],
                "confusion_mass": {name: masses[name][pair] for name in PROBES},
                "head_to_head": head_to_head(x_train, y_train, x_test, y_test, pair),
                "role": next(
                    (
                        role
                        for role, candidate in CANDIDATES.items()
                        if tuple(sorted(label_map[n] for n in candidate)) == pair
                    ),
                    None,
                ),
            }
        )

    for row in pairs:
        if row["role"]:
            row["checks"] = verdict(row, row["role"], coverage, support["open"])

    return {
        "seed": SEED,
        "label_map": label_map,
        "criteria": CRITERIA,
        "support": support,
        "client_coverage": coverage,
        "feature_degeneracy": degeneracy,
        "multiclass": multiclass,
        "pairs": pairs,
        "early_rounds": early_round_confusion(truth, num_classes),
    }


def render(report: dict) -> str:
    names = {index: name for name, index in report["label_map"].items()}
    lines = ["", "multiclass probes on the sealed test split", ""]
    for name, result in report["multiclass"].items():
        lines.append(f"   {name:<8} accuracy {result['accuracy']:.4f}")
    lines += ["", "distinct feature rows per class in the sealed test split", ""]
    for index, stats in report["feature_degeneracy"].items():
        index = int(index)
        flag = "  <- collapsed" if stats["distinct_rows"] < 0.5 * stats["samples"] else ""
        lines.append(
            f"   {names[index]:<16} {stats['distinct_rows']:>5} distinct of {stats['samples']:>5}"
            f"   mean feature std {stats['mean_feature_std']:.6f}{flag}"
        )
    lines += ["", "most-confused pairs (random forest), and each pair refit head-to-head", ""]
    lines.append(
        f"   {'pair':<28}{'mass':>8}{'linear':>9}{'forest':>9}{'1-nn':>8}   role"
    )
    for row in sorted(report["pairs"], key=lambda r: -r["confusion_mass"]["forest"]):
        pair = f"{row['names'][0]} / {row['names'][1]}"
        lines.append(
            f"   {pair:<28}{row['confusion_mass']['forest']:>8.4f}"
            f"{row['head_to_head']['linear']:>9.4f}"
            f"{row['head_to_head']['forest']:>9.4f}"
            f"{row['head_to_head']['one_nn']:>8.4f}   {row['role'] or ''}"
        )
    lines += ["", "candidates against the pre-registered criteria", ""]
    for row in report["pairs"]:
        if not row.get("checks"):
            continue
        lines.append(f"   {row['role']}: {row['names'][0]} / {row['names'][1]}")
        for label, passed in row["checks"]:
            lines.append(f"      {'PASS' if passed else 'FAIL'}  {label}")
        for index in row["labels"]:
            lines.append(
                f"      {names[index]:<16} clients {report['client_coverage'].get(index, 0):>3}"
                f"   private {report['support']['private'].get(index, 0):>6}"
                f"   open {report['support']['open'].get(index, 0):>5}"
                f"   test {report['support']['test'].get(index, 0):>6}"
            )
    if report["early_rounds"]:
        lines += ["", "recorded shadow run, majority pseudo-labels on the open set", ""]
        for server_round, result in report["early_rounds"].items():
            lines.append(
                f"   round {server_round:>3}   accuracy {result['accuracy']:.4f}"
                f"   valid rate {result['valid_rate']:.4f}"
            )
        for row in report["pairs"]:
            if not row["role"]:
                continue
            a, b = row["labels"]
            lines.append("")
            lines.append(f"   {row['names'][0]} / {row['names'][1]} as pseudo-labelled:")
            for server_round, result in report["early_rounds"].items():
                matrix = np.array(result["confusion"])
                lines.append(
                    f"      round {server_round:>3}"
                    f"   {names[a]} -> {names[b]}: {matrix[a, b]:>4}"
                    f"   {names[b]} -> {names[a]}: {matrix[b, a]:>4}"
                    f"   correct: {matrix[a, a]:>4} / {matrix[b, b]:>4}"
                )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-pairs", type=int, default=6)
    args = parser.parse_args()

    if not DATA_ROOT.exists():
        raise SystemExit(f"no prepared data at {DATA_ROOT}; run ssfl.data.prepare_data first")

    report = run_probe(top_pairs=args.top_pairs)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(render(report))
    print(f"written to {OUTPUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
