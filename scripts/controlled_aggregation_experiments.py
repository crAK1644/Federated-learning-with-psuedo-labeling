"""The three controlled aggregation experiments from the Dawid-Skene follow-up plan, stage 3.

These run entirely outside federated learning: no Flower, no model training, no N-BaIoT. Their job
is to answer "does the aggregator behave the way the theory says it should?" on data where the
answer is known by construction, so that a later result on real data can be attributed to the data
rather than to the estimator.

    uv run python scripts/controlled_aggregation_experiments.py

**1. Easy convergence.** Seven reliable clients, ~90% accurate, a thousand shared items.
Dawid-Skene should converge in a handful of EM steps and land near the ceiling. If it cannot do
this, nothing further is worth running.

**2. Systematic error sweep.** Start from all-reliable and swap in clients carrying the *same*
systematic error one at a time. Majority vote should be strong when the pool is clean and should
degrade as the shared bias grows, because a vote cannot distinguish "three clients agree because
they are right" from "three clients agree because they are wrong in the same way". Dawid-Skene
should hold on longer, because a per-client confusion matrix can.

The sweep deliberately continues past the point where the biased clients become a majority of the
pool, but asserts nothing there. The bias used here is *class-conditional*: the biased clients stay
accurate on three of the four classes and fail only on the fourth. A confusion matrix represents
that directly, so Dawid-Skene keeps working even when the biased clients outnumber the rest -- the
measured curve shows this, and it is a property of this error shape, not a general guarantee. A
client that were wrong everywhere, or a bias shared by nearly all clients, is a different regime
and is not evidence from this experiment.

**3. Artificial ties.** Anchor items establish who is reliable; target items are then constructed
as exact 2-2 splits between a reliable pair and an unreliable pair. Majority vote has no signal to
break those ties with -- this repository breaks them by lowest class index, which is an arbitrary
rule, not a decision -- while Dawid-Skene has already learned which pair to believe.

A negative control runs the same tie items with the anchors removed. The problem is then genuinely
unidentifiable, and the estimator must *not* look like it solved it. An aggregator that scores well
here is exploiting the class indices, not the data.

Every expectation is declared in ``EXPECTATIONS`` before the run, and the script exits non-zero if
one of them is missed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ssfl.protocols.dawid_skene import ABSTAIN, DawidSkeneSettings, fit_dawid_skene

NUM_CLASSES = 4
SEED = 20_260_819

#: Written down before the run. Each entry is checked in ``_check`` and reported as pass/fail.
EXPECTATIONS = {
    # 1. Easy convergence.
    "easy_min_ds_accuracy": 0.99,
    "easy_max_median_iterations": 20,
    "easy_min_fit_pass_rate": 1.0,
    # 2. Systematic sweep, over the identifiable part of the curve only (biased clients in the
    #    minority). Majority is allowed to win on a clean pool; it is not allowed to keep winning.
    "sweep_max_ds_deficit_when_clean": 0.01,
    "sweep_min_ds_advantage_at_minority_bias": 0.02,
    # 3. Artificial ties. Chance on a 2-2 split between two distinct classes is 0.5.
    "tie_min_ds_accuracy": 0.90,
    "tie_max_majority_accuracy": 0.60,
    # 3b. Negative control: without anchors the tie is unidentifiable and must stay that way.
    "unidentifiable_max_ds_accuracy": 0.60,
}


def estimator_settings() -> DawidSkeneSettings:
    """The estimator without the deployment gates.

    ``permutation_min_*`` decide whether a fit may be broadcast to clients. They are a safety
    policy, and leaving them on would mean these experiments measure the policy rather than the
    estimator -- the tie experiment in particular constructs cases where the fit is *supposed* to
    disagree with majority, which is exactly what the agreement gate rejects.
    """
    return DawidSkeneSettings(
        max_iterations=500,
        min_clients=3,
        permutation_min_diagonal_ratio=0.0,
        permutation_min_majority_agreement=0.0,
    )


# --- shared helpers ---------------------------------------------------------------------------


def uniform_confusion(diagonal: float) -> np.ndarray:
    """``M[c, k] = P(says k | truth c)``: correct with ``diagonal``, errors spread evenly."""
    off = (1.0 - diagonal) / (NUM_CLASSES - 1)
    matrix = np.full((NUM_CLASSES, NUM_CLASSES), off, dtype=np.float64)
    np.fill_diagonal(matrix, diagonal)
    return matrix


def systematic_confusion(diagonal: float, source: int, target: int, rate: float) -> np.ndarray:
    """Reliable everywhere except on ``source``, where the client mostly reports ``target``."""
    matrix = uniform_confusion(diagonal)
    row = np.full(NUM_CLASSES, (1.0 - rate) / (NUM_CLASSES - 1))
    row[target] = rate
    matrix[source] = row
    return matrix


def simulate(rng: np.random.Generator, truth: np.ndarray, confusions: np.ndarray) -> np.ndarray:
    """Draw one ``(J, N)`` annotation matrix from per-client confusion matrices."""
    annotations = np.empty((len(confusions), len(truth)), dtype=np.int8)
    for j, confusion in enumerate(confusions):
        cumulative = confusion.cumsum(axis=1)
        draws = rng.random(len(truth))
        annotations[j] = (draws[:, None] > cumulative[truth]).sum(axis=1)
    return annotations


def majority_lowest_index(annotations: np.ndarray) -> np.ndarray:
    """Production's rule: per-item majority, ties to the lowest class index."""
    votes = _votes(annotations)
    return np.where(votes.sum(axis=1) > 0, votes.argmax(axis=1), ABSTAIN).astype(np.int64)


def majority_random_tie(annotations: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Majority with ties broken uniformly -- the honest chance baseline on a constructed tie.

    Production's lowest-index rule is not a chance baseline: on a 2-2 split it is right whenever
    the true class happens to carry the lower index, which is a property of the label map rather
    than of the aggregator.
    """
    votes = _votes(annotations)
    labels = np.full(annotations.shape[1], ABSTAIN, dtype=np.int64)
    for item, row in enumerate(votes):
        if row.sum() == 0:
            continue
        winners = np.flatnonzero(row == row.max())
        labels[item] = int(rng.choice(winners))
    return labels


def _votes(annotations: np.ndarray) -> np.ndarray:
    votes = np.zeros((annotations.shape[1], NUM_CLASSES), dtype=np.int64)
    for row in annotations:
        observed = np.nonzero(row != ABSTAIN)[0]
        np.add.at(votes, (observed, row[observed].astype(np.int64)), 1)
    return votes


def _accuracy(labels: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> float:
    mask = np.ones(len(truth), dtype=bool) if mask is None else mask
    return float((labels[mask] == truth[mask]).mean()) if mask.any() else 0.0


# --- 1. easy convergence ----------------------------------------------------------------------


def experiment_easy_convergence(
    *, num_clients: int = 7, num_items: int = 1_000, diagonal: float = 0.90, replicates: int = 20
) -> dict:
    settings = estimator_settings()
    confusions = np.stack([uniform_confusion(diagonal)] * num_clients)
    ds_scores, majority_scores, iterations, passed = [], [], [], 0

    for replicate in np.random.SeedSequence(SEED).spawn(replicates):
        rng = np.random.default_rng(replicate)
        truth = rng.integers(0, NUM_CLASSES, size=num_items)
        annotations = simulate(rng, truth, confusions)
        majority = majority_lowest_index(annotations)
        majority_scores.append(_accuracy(majority, truth))

        fit = fit_dawid_skene(annotations, NUM_CLASSES, majority, settings)
        if fit.ok:
            passed += 1
            ds_scores.append(_accuracy(fit.labels, truth))
            iterations.append(fit.iterations)

    return {
        "num_clients": num_clients,
        "num_items": num_items,
        "client_accuracy": diagonal,
        "replicates": replicates,
        "ds_accuracy": float(np.mean(ds_scores)) if ds_scores else 0.0,
        "majority_accuracy": float(np.mean(majority_scores)),
        "median_iterations": float(np.median(iterations)) if iterations else 0.0,
        "max_iterations_seen": int(np.max(iterations)) if iterations else 0,
        "fit_pass_rate": passed / replicates,
    }


# --- 2. systematic error sweep ----------------------------------------------------------------


def experiment_systematic_sweep(
    *,
    num_clients: int = 7,
    num_items: int = 1_000,
    reliable_diagonal: float = 0.90,
    biased_diagonal: float = 0.90,
    source: int = 2,
    target: int = 1,
    rate: float = 0.85,
    replicates: int = 20,
) -> list[dict]:
    """Swap reliable clients for identically-biased ones, one at a time.

    The biased clients share *one* error direction rather than each having their own. Independent
    biases average out in a vote and would make this experiment easy; a shared bias is the case
    that actually defeats majority, and the one a federated pool with correlated data produces.
    """
    settings = estimator_settings()
    rows = []
    for biased in range(num_clients - 1):
        confusions = np.stack(
            [uniform_confusion(reliable_diagonal)] * (num_clients - biased)
            + [systematic_confusion(biased_diagonal, source, target, rate)] * biased
        )
        ds_scores, majority_scores, source_recall, passed = [], [], [], 0
        for replicate in np.random.SeedSequence(SEED + biased).spawn(replicates):
            rng = np.random.default_rng(replicate)
            truth = rng.integers(0, NUM_CLASSES, size=num_items)
            annotations = simulate(rng, truth, confusions)
            majority = majority_lowest_index(annotations)
            majority_scores.append(_accuracy(majority, truth))

            fit = fit_dawid_skene(annotations, NUM_CLASSES, majority, settings)
            if fit.ok:
                passed += 1
                ds_scores.append(_accuracy(fit.labels, truth))
                # The attacked class on its own: the aggregate number hides a class that has been
                # wiped out when the other three are unaffected.
                source_recall.append(_accuracy(fit.labels, truth, truth == source))

        rows.append(
            {
                "biased_clients": biased,
                "reliable_clients": num_clients - biased,
                "biased_are_majority": biased * 2 > num_clients,
                "ds_accuracy": float(np.mean(ds_scores)) if ds_scores else 0.0,
                "majority_accuracy": float(np.mean(majority_scores)),
                "ds_recall_on_attacked_class": (
                    float(np.mean(source_recall)) if source_recall else 0.0
                ),
                "fit_pass_rate": passed / replicates,
            }
        )
    for row in rows:
        row["ds_advantage"] = row["ds_accuracy"] - row["majority_accuracy"]
    return rows


# --- 3. artificial ties -----------------------------------------------------------------------


def build_tie_case(
    rng: np.random.Generator, *, num_anchor: int, num_tie: int, reliable: float, poor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anchors that identify the reliable pair, then target items split exactly 2-2.

    Clients 0-1 are the reliable pair, clients 2-3 the unreliable one. On a target item the
    reliable pair reports the truth and the unreliable pair reports a fixed *relabelling* of it, so
    the vote is tied by construction and only per-client reliability can break it.

    A relabelling fixed within the replicate, not a random wrong class per item, because the
    negative control depends on it. If the unreliable pair scattered its answers, its emissions
    would be measurably less consistent with any latent class than the reliable pair's, and the
    estimator could tell the two apart from the tie items alone -- there would be nothing for the
    anchors to contribute and nothing for the control to prove. Against a fixed relabelling the two
    hypotheses are exact permutations of each other: with no anchors the likelihood cannot separate
    them, which is the point.

    The shift itself is drawn per replicate rather than pinned, because the lowest-index tie rule
    is sensitive to it: with a shift of +1 the true class carries the lower index three times in
    four and the rule looks like it is deciding something. Averaging over shifts removes that
    artefact from the reported number, and the spread is reported alongside it.

    The unreliable pair is therefore deliberately *more* consistent on target items than its anchor
    accuracy implies. That is a property of the construction, and it is why the anchors, not the
    targets, are what identify the raters.
    """
    truth = rng.integers(0, NUM_CLASSES, size=num_anchor + num_tie)
    tie_mask = np.zeros(len(truth), dtype=bool)
    tie_mask[num_anchor:] = True

    confusions = np.stack([uniform_confusion(reliable)] * 2 + [uniform_confusion(poor)] * 2)
    annotations = simulate(rng, truth, confusions)

    target_truth = truth[tie_mask]
    shifted = (target_truth + int(rng.integers(1, NUM_CLASSES))) % NUM_CLASSES
    annotations[0, tie_mask] = target_truth.astype(np.int8)
    annotations[1, tie_mask] = target_truth.astype(np.int8)
    annotations[2, tie_mask] = shifted.astype(np.int8)
    annotations[3, tie_mask] = shifted.astype(np.int8)
    return truth, annotations, tie_mask


def experiment_artificial_ties(
    *,
    num_anchor: int = 600,
    num_tie: int = 200,
    reliable: float = 0.95,
    poor: float = 0.45,
    replicates: int = 20,
) -> dict:
    settings = estimator_settings()
    results: dict[str, dict] = {}

    for label, anchors in (("with_anchors", num_anchor), ("unidentifiable", 0)):
        ds_tie, majority_tie, majority_random, passed = [], [], [], 0
        for replicate in np.random.SeedSequence(SEED + 100).spawn(replicates):
            rng = np.random.default_rng(replicate)
            truth, annotations, tie_mask = build_tie_case(
                rng, num_anchor=anchors, num_tie=num_tie, reliable=reliable, poor=poor
            )
            majority = majority_lowest_index(annotations)
            majority_tie.append(_accuracy(majority, truth, tie_mask))
            majority_random.append(
                _accuracy(majority_random_tie(annotations, rng), truth, tie_mask)
            )

            fit = fit_dawid_skene(annotations, NUM_CLASSES, majority, settings)
            if fit.ok:
                passed += 1
                ds_tie.append(_accuracy(fit.labels, truth, tie_mask))

        results[label] = {
            "anchor_items": anchors,
            "tie_items": num_tie,
            "ds_tie_accuracy": float(np.mean(ds_tie)) if ds_tie else 0.0,
            "majority_tie_accuracy_lowest_index": float(np.mean(majority_tie)),
            # The spread across replicates, i.e. across the relabelling the tie is built from.
            # A rule that decides on signal would not have one.
            "majority_tie_accuracy_lowest_index_min": float(np.min(majority_tie)),
            "majority_tie_accuracy_lowest_index_max": float(np.max(majority_tie)),
            "majority_tie_accuracy_random": float(np.mean(majority_random)),
            "fit_pass_rate": passed / replicates,
        }
    return results


# --- orchestration ----------------------------------------------------------------------------


def _check(easy: dict, sweep: list[dict], ties: dict) -> list[str]:
    failures = []
    if easy["ds_accuracy"] < EXPECTATIONS["easy_min_ds_accuracy"]:
        failures.append(f"easy: ds accuracy {easy['ds_accuracy']:.4f}")
    if easy["median_iterations"] > EXPECTATIONS["easy_max_median_iterations"]:
        failures.append(f"easy: median iterations {easy['median_iterations']:.0f}")
    if easy["fit_pass_rate"] < EXPECTATIONS["easy_min_fit_pass_rate"]:
        failures.append(f"easy: fit pass rate {easy['fit_pass_rate']:.2f}")

    clean = sweep[0]
    if -clean["ds_advantage"] > EXPECTATIONS["sweep_max_ds_deficit_when_clean"]:
        failures.append(f"sweep: ds deficit on a clean pool {-clean['ds_advantage']:.4f}")

    identifiable = [row for row in sweep if not row["biased_are_majority"]]
    best = max(row["ds_advantage"] for row in identifiable)
    if best < EXPECTATIONS["sweep_min_ds_advantage_at_minority_bias"]:
        failures.append(f"sweep: best ds advantage under minority bias {best:.4f}")

    anchored = ties["with_anchors"]
    if anchored["ds_tie_accuracy"] < EXPECTATIONS["tie_min_ds_accuracy"]:
        failures.append(f"ties: ds tie accuracy {anchored['ds_tie_accuracy']:.4f}")
    if anchored["majority_tie_accuracy_lowest_index"] > EXPECTATIONS["tie_max_majority_accuracy"]:
        failures.append(
            f"ties: majority tie accuracy {anchored['majority_tie_accuracy_lowest_index']:.4f}"
        )
    blind = ties["unidentifiable"]["ds_tie_accuracy"]
    if blind > EXPECTATIONS["unidentifiable_max_ds_accuracy"]:
        failures.append(f"ties: ds solved an unidentifiable case, {blind:.4f}")
    return failures


def run_experiments(*, replicates: int = 20) -> dict:
    easy = experiment_easy_convergence(replicates=replicates)
    sweep = experiment_systematic_sweep(replicates=replicates)
    ties = experiment_artificial_ties(replicates=replicates)
    failures = _check(easy, sweep, ties)
    return {
        "seed": SEED,
        "num_classes": NUM_CLASSES,
        "replicates": replicates,
        "expectations": EXPECTATIONS,
        "easy_convergence": easy,
        "systematic_sweep": sweep,
        "artificial_ties": ties,
        "failures": failures,
        "passed": not failures,
    }


def plot_sweep(sweep: list[dict], path: Path) -> None:
    """The one figure the plan asks for: both accuracy curves against the number of biased clients."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    biased = [row["biased_clients"] for row in sweep]
    figure, axes = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    axes.plot(biased, [row["majority_accuracy"] for row in sweep], marker="o", label="majority vote")
    axes.plot(biased, [row["ds_accuracy"] for row in sweep], marker="s", label="Dawid-Skene")
    axes.plot(
        biased,
        [row["ds_recall_on_attacked_class"] for row in sweep],
        marker="^",
        linestyle="--",
        label="Dawid-Skene, attacked class only",
    )
    breaking = [row["biased_clients"] for row in sweep if row["biased_are_majority"]]
    if breaking:
        axes.axvspan(
            min(breaking) - 0.5, max(biased) + 0.5, color="0.85", zorder=0, label="biased majority"
        )
    axes.set_xlabel("clients carrying the shared systematic error")
    axes.set_ylabel("accuracy")
    axes.set_title("Aggregation under a shared systematic error")
    axes.set_ylim(0.0, 1.02)
    axes.set_xlim(min(biased) - 0.5, max(biased) + 0.5)
    axes.grid(alpha=0.25)
    axes.legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _print(report: dict) -> None:
    easy = report["easy_convergence"]
    print("1. easy convergence")
    print(
        f"   {easy['num_clients']} clients at {easy['client_accuracy']:.2f}, "
        f"{easy['num_items']} items, {easy['replicates']} replicates"
    )
    print(
        f"   dawid-skene {easy['ds_accuracy']:.4f}   majority {easy['majority_accuracy']:.4f}   "
        f"median EM steps {easy['median_iterations']:.0f} (max {easy['max_iterations_seen']})   "
        f"fit pass rate {easy['fit_pass_rate']:.2f}"
    )

    print("\n2. systematic error sweep")
    header = f"   {'biased':>7}{'majority':>11}{'dawid-skene':>13}{'advantage':>11}{'attacked class':>16}"
    print(header)
    print("   " + "-" * (len(header) - 3))
    for row in report["systematic_sweep"]:
        flag = "  <- biased majority" if row["biased_are_majority"] else ""
        print(
            f"   {row['biased_clients']:>7}{row['majority_accuracy']:>11.4f}"
            f"{row['ds_accuracy']:>13.4f}{row['ds_advantage']:>+11.4f}"
            f"{row['ds_recall_on_attacked_class']:>16.4f}{flag}"
        )

    print("\n3. artificial ties (accuracy on the tied items only)")
    header = (
        f"   {'case':<16}{'anchors':>9}{'lowest index':>14}{'(spread)':>16}"
        f"{'random tie':>12}{'dawid-skene':>13}"
    )
    print(header)
    print("   " + "-" * (len(header) - 3))
    for label, row in report["artificial_ties"].items():
        spread = (
            f"{row['majority_tie_accuracy_lowest_index_min']:.2f}"
            f"-{row['majority_tie_accuracy_lowest_index_max']:.2f}"
        )
        print(
            f"   {label:<16}{row['anchor_items']:>9}"
            f"{row['majority_tie_accuracy_lowest_index']:>14.4f}{spread:>16}"
            f"{row['majority_tie_accuracy_random']:>12.4f}{row['ds_tie_accuracy']:>13.4f}"
        )

    print()
    if report["passed"]:
        print("PASS: every pre-registered expectation held.")
    else:
        for failure in report["failures"]:
            print(f"FAIL: {failure}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Controlled non-federated aggregation experiments")
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation"))
    args = parser.parse_args()

    report = run_experiments(replicates=args.replicates)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "controlled_aggregation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    figure = args.output / "systematic_sweep.png"
    plot_sweep(report["systematic_sweep"], figure)

    _print(report)
    print(f"\nreport: {args.output / 'controlled_aggregation.json'}\nfigure: {figure}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
