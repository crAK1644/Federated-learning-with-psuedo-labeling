"""Validate this repository's Dawid-Skene estimator against an independent implementation.

``scripts/reproduce_rater_evaluation.py`` reproduces the *behaviour* the ``rater`` presentation
reports (majority beats the full model with uniformly reliable raters; the full model wins against
systematic spammers). That is a qualitative check of one estimator against itself. This script does
the other half: it hands **the same fixed annotation matrix** to this repository's MAP-EM estimator
and to `crowd-kit <https://github.com/Toloka/crowd-kit>`_'s ``DawidSkene``, then compares the two.

    uv run --with crowd-kit python scripts/validate_dawid_skene_reference.py

crowd-kit is deliberately *not* a project dependency: it pulls transformers/tokenizers/nltk for
features this comparison never touches, and it is needed for offline validation only. Run it with
``uv run --with`` as above (the test that wraps this script skips when it is absent).
Note that the wrapping test needs ``uv run --with crowd-kit python -m pytest``: a bare
``pytest`` runs the base interpreter from its shebang and never sees the overlay.

Reference choice: the follow-up plan names ``rater``. ``rater`` is R + Stan (Bayesian, MCMC or
penalised optimisation); standing up an R toolchain to compare against a deterministic MAP-EM
estimator buys a second source of difference, not a cleaner comparison. crowd-kit is an
independent Python implementation of the same 1979 model, deterministic, and pip-installable.
Add ``rater`` on top only if the Bayesian posterior itself is what needs checking.

What is compared, and why each item is on the list:

* **Aggregate label accuracy**, against the stored latent truth and against majority vote. The
  headline number; both implementations must land within tolerance of each other.
* **Posterior class probabilities**, per item. Argmax agreement hides an estimator that is right
  for the wrong reason; the distributions have to line up too.
* **Confusion matrix orientation.** crowd-kit's ``errors_`` is indexed ``(worker, observed_label)``
  with the *true* class in the columns -- the transpose of this repository's
  ``M_j[c, k] = P(client j says k | truth is c)``. Getting this backwards is a silent bug that
  still produces plausible accuracy, so the check is explicit.
* **Systematic-error direction.** The planted biased client must have its off-diagonal mass in the
  same cell in both fits.
* **Missing annotations and all-abstain items.** ``ABSTAIN`` is missing data in this repository,
  never a class; crowd-kit represents it by absence of a row. An item nobody annotated must not
  move the fit for the items that were annotated.
* **Determinism and client-order invariance**, which float64 EM does not get for free.

Exact floating-point equality is not expected and not required: this repository fits a *MAP* model
with Dirichlet pseudocounts, crowd-kit fits the unregularised MLE. The comparison arm therefore
runs with the pseudocounts driven to ``NEAR_MLE_PSEUDOCOUNT`` so the two objectives nearly
coincide, and every check is a written-down tolerance (see ``TOLERANCES``) rather than an equality.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np

from ssfl.protocols.dawid_skene import ABSTAIN, DawidSkeneSettings, fit_dawid_skene

# --- The fixed case ---------------------------------------------------------------------------
# Small, hand-specified and readable on purpose: every client's true confusion matrix is written
# down below, so a disagreement between the two implementations can be attributed to a client.

NUM_CLASSES = 4
NUM_ITEMS = 400
SEED = 20_260_818

#: ``(name, kind)`` per client. ``kind`` selects the true confusion matrix built in ``_confusion``.
CLIENT_SPEC: tuple[tuple[str, str], ...] = (
    ("client-0", "reliable"),
    ("client-1", "reliable"),
    ("client-2", "reliable"),
    ("client-3", "reliable"),
    ("client-4", "noisy"),
    ("client-5", "systematic"),
    ("client-6", "sparse"),
)

#: Diagonal probability per client kind; the remainder is spread uniformly over the other classes,
#: except for "systematic", which is handled separately in ``_confusion``.
DIAGONAL = {"reliable": 0.85, "noisy": 0.55, "sparse": 0.85}

#: The systematic client reports class ``SYSTEMATIC_TO`` whenever the truth is ``SYSTEMATIC_FROM``.
SYSTEMATIC_FROM, SYSTEMATIC_TO, SYSTEMATIC_RATE = 2, 1, 0.80

#: Fraction of items the sparse client annotates at all (the rest are ``ABSTAIN``).
SPARSE_COVERAGE = 0.30

#: Fraction of the remaining items dropped at random in the "missing annotations" variant.
MISSING_RATE = 0.20

#: Number of never-annotated items appended in the "all abstain" variant.
NUM_ALL_ABSTAIN_ITEMS = 40

#: Pseudocounts for the comparison arm. crowd-kit fits the unregularised MLE, so the MAP
#: pseudocounts are driven towards zero rather than left at their production values.
NEAR_MLE_PSEUDOCOUNT = 1e-9

TOLERANCES = {
    # Fraction of items on which the two implementations must agree, after permutation alignment.
    "min_label_agreement": 0.98,
    # |our accuracy - reference accuracy| against the stored latent truth.
    "max_accuracy_gap": 0.02,
    "max_mean_posterior_diff": 0.05,
    "max_max_posterior_diff": 0.30,
    "max_mean_confusion_diff": 0.05,
    # The fit must be worth running at all: both implementations must beat majority vote on a
    # case that contains a systematically biased client.
    "min_gain_over_majority": 0.0,
}


@dataclass(frozen=True)
class Case:
    """One annotation matrix plus everything needed to score a fit on it."""

    name: str
    annotations: np.ndarray  # (J, N) int8, ABSTAIN = missing
    truth: np.ndarray  # (N,) int64 latent classes
    true_confusion: np.ndarray  # (J, C, C), row c = P(says k | truth c)
    senders: tuple[str, ...]


def _confusion(kind: str) -> np.ndarray:
    """True confusion matrix for one client kind, ``M[c, k] = P(says k | truth is c)``."""
    if kind == "systematic":
        matrix = np.eye(NUM_CLASSES, dtype=np.float64) * 0.90
        off = (1.0 - 0.90) / (NUM_CLASSES - 1)
        matrix[matrix == 0.0] = off
        # Overwrite one row: on SYSTEMATIC_FROM this client mostly says SYSTEMATIC_TO. A majority
        # vote cannot tell this apart from noise; a per-client confusion matrix can.
        row = np.full(NUM_CLASSES, (1.0 - SYSTEMATIC_RATE) / (NUM_CLASSES - 1))
        row[SYSTEMATIC_TO] = SYSTEMATIC_RATE
        matrix[SYSTEMATIC_FROM] = row
        return matrix
    diagonal = DIAGONAL[kind]
    off = (1.0 - diagonal) / (NUM_CLASSES - 1)
    matrix = np.full((NUM_CLASSES, NUM_CLASSES), off, dtype=np.float64)
    np.fill_diagonal(matrix, diagonal)
    return matrix


def build_case(rng: np.random.Generator) -> Case:
    """The base case: every client annotates every item except the sparse one."""
    truth = rng.integers(0, NUM_CLASSES, size=NUM_ITEMS)
    true_confusion = np.stack([_confusion(kind) for _, kind in CLIENT_SPEC])
    annotations = np.empty((len(CLIENT_SPEC), NUM_ITEMS), dtype=np.int8)
    for j, (_, kind) in enumerate(CLIENT_SPEC):
        draws = np.array(
            [rng.choice(NUM_CLASSES, p=true_confusion[j, c]) for c in truth], dtype=np.int8
        )
        if kind == "sparse":
            keep = rng.random(NUM_ITEMS) < SPARSE_COVERAGE
            draws = np.where(keep, draws, np.int8(ABSTAIN))
        annotations[j] = draws
    senders = tuple(name for name, _ in CLIENT_SPEC)
    return Case("base", annotations, truth.astype(np.int64), true_confusion, senders)


def with_missing(case: Case, rng: np.random.Generator) -> Case:
    """Drop ``MISSING_RATE`` of the observed cells: sparser, but every item keeps some coverage."""
    annotations = case.annotations.copy()
    drop = rng.random(annotations.shape) < MISSING_RATE
    # Never strip an item down to zero annotations -- that is the separate all-abstain variant.
    drop[:, (annotations != ABSTAIN).sum(axis=0) - drop.sum(axis=0) < 2] = False
    annotations[drop] = ABSTAIN
    return Case("missing", annotations, case.truth, case.true_confusion, case.senders)


def with_all_abstain_items(case: Case) -> Case:
    """Append items nobody annotated. The fit for the original items must not move."""
    padding = np.full((case.annotations.shape[0], NUM_ALL_ABSTAIN_ITEMS), ABSTAIN, dtype=np.int8)
    annotations = np.concatenate([case.annotations, padding], axis=1)
    truth = np.concatenate([case.truth, np.full(NUM_ALL_ABSTAIN_ITEMS, ABSTAIN, dtype=np.int64)])
    return Case("all_abstain", annotations, truth, case.true_confusion, case.senders)


# --- The two implementations ------------------------------------------------------------------


def majority_labels(annotations: np.ndarray) -> np.ndarray:
    """Deterministic majority vote, lowest class index on a tie -- production's rule."""
    votes = np.zeros((annotations.shape[1], NUM_CLASSES), dtype=np.int64)
    for row in annotations:
        observed = np.nonzero(row != ABSTAIN)[0]
        np.add.at(votes, (observed, row[observed].astype(np.int64)), 1)
    return np.where(votes.sum(axis=1) > 0, votes.argmax(axis=1), ABSTAIN).astype(np.int64)


def comparison_settings() -> DawidSkeneSettings:
    """Match crowd-kit as closely as the two models allow.

    Pseudocounts to ~0 (crowd-kit is unregularised MLE) and both permutation gates disabled: the
    gates are a *deployment* safeguard against broadcasting a relabelled fit, and applying them
    here would compare a guarded estimator against an unguarded one. Alignment is handled
    explicitly by ``align_classes`` instead.
    """
    return DawidSkeneSettings(
        max_iterations=500,
        tolerance=1e-10,
        initialization_pseudocount=NEAR_MLE_PSEUDOCOUNT,
        confusion_pseudocount=NEAR_MLE_PSEUDOCOUNT,
        class_prior_pseudocount=NEAR_MLE_PSEUDOCOUNT,
        permutation_min_diagonal_ratio=0.0,
        permutation_min_majority_agreement=0.0,
    )


def run_ours(case: Case) -> dict:
    fit = fit_dawid_skene(
        case.annotations,
        NUM_CLASSES,
        majority_labels(case.annotations),
        settings=comparison_settings(),
        senders=case.senders,
    )
    if not fit.ok:
        raise RuntimeError(f"{case.name}: our estimator refused the fit: {fit.status}")
    return {
        "labels": fit.labels,
        "posterior": fit.posterior,
        "confusion": fit.confusion,
        "prior": fit.prior,
        "iterations": fit.iterations,
        "converged": fit.converged,
    }


def run_reference(case: Case) -> dict:
    """Same matrix through crowd-kit, returned in *this repository's* orientation."""
    import pandas as pd
    from crowdkit.aggregation import DawidSkene

    rows, cols = np.nonzero(case.annotations != ABSTAIN)
    frame = pd.DataFrame(
        {
            "task": cols,
            "worker": [case.senders[j] for j in rows],
            "label": case.annotations[rows, cols].astype(np.int64),
        }
    )
    model = DawidSkene(n_iter=500, tol=1e-10)
    model.fit(frame)

    num_items = case.annotations.shape[1]
    posterior = np.full((num_items, NUM_CLASSES), np.nan)
    probas = model.probas_.reindex(columns=range(NUM_CLASSES)).fillna(0.0)
    posterior[probas.index.to_numpy()] = probas.to_numpy()

    # errors_ is (worker, observed_label) x true_class = P(says observed | truth). Transposing it
    # into M[c, k] = P(says k | truth c) is the orientation check this comparison exists for.
    confusion = np.zeros((len(case.senders), NUM_CLASSES, NUM_CLASSES))
    errors = model.errors_
    for j, sender in enumerate(case.senders):
        for observed_label in range(NUM_CLASSES):
            key = (sender, observed_label)
            if key in errors.index:
                confusion[j, :, observed_label] = errors.loc[key].reindex(
                    range(NUM_CLASSES)
                ).fillna(0.0)

    labels = np.full(num_items, ABSTAIN, dtype=np.int64)
    observed_items = np.nonzero((case.annotations != ABSTAIN).any(axis=0))[0]
    labels[observed_items] = posterior[observed_items].argmax(axis=1)
    prior = model.priors_.reindex(range(NUM_CLASSES)).fillna(0.0).to_numpy()
    return {
        "labels": labels,
        "posterior": posterior,
        "confusion": confusion,
        "prior": prior,
        "iterations": None,
        "converged": True,
    }


# --- Alignment and comparison -----------------------------------------------------------------


def align_classes(ours: dict, reference: dict, mask: np.ndarray) -> tuple[int, ...]:
    """Permutation ``p`` of the reference's classes maximising agreement with ours.

    The Dawid-Skene likelihood is invariant to relabelling the hidden classes, so two correct
    implementations may converge to the same solution under different names. ``NUM_CLASSES`` is
    small and fixed, so this brute-forces the permutations rather than pulling in an assignment
    solver.
    """
    ours_labels = ours["labels"][mask]
    reference_labels = reference["labels"][mask]
    best, best_score = tuple(range(NUM_CLASSES)), -1.0
    for candidate in permutations(range(NUM_CLASSES)):
        mapped = np.asarray(candidate, dtype=np.int64)[reference_labels]
        score = float((mapped == ours_labels).mean())
        if score > best_score:
            best, best_score = candidate, score
    return best


def _apply(reference: dict, permutation: tuple[int, ...]) -> dict:
    """Rewrite the reference fit into our class naming."""
    order = np.asarray(permutation, dtype=np.int64)
    inverse = np.argsort(order)
    return {
        "labels": np.where(reference["labels"] == ABSTAIN, ABSTAIN, order[reference["labels"]]),
        # Column k of the posterior is reference class k, which is our class order[k].
        "posterior": reference["posterior"][:, inverse],
        # Both axes of the confusion matrix are class axes: rows are the truth, columns the
        # emitted label. Permuting only one of them is the classic way to get this wrong.
        "confusion": reference["confusion"][:, inverse, :][:, :, inverse],
        "prior": reference["prior"][inverse],
        "iterations": reference["iterations"],
        "converged": reference["converged"],
    }


def compare(case: Case, ours: dict, reference: dict) -> dict:
    """One row of the comparison table, plus the checks that row has to pass."""
    scored = case.truth != ABSTAIN
    observed = (case.annotations != ABSTAIN).any(axis=0)
    mask = scored & observed

    permutation = align_classes(ours, reference, mask)
    aligned = _apply(reference, permutation)

    majority = majority_labels(case.annotations)
    posterior_diff = np.abs(ours["posterior"][mask] - aligned["posterior"][mask])
    confusion_diff = np.abs(ours["confusion"] - aligned["confusion"])

    # Direction of the planted systematic error: the off-diagonal cell carrying the most mass in
    # the SYSTEMATIC_FROM row must be the same cell in both fits.
    systematic = [j for j, (_, kind) in enumerate(CLIENT_SPEC) if kind == "systematic"][0]
    # Our fit drops clients below ``min_client_annotations`` and returns confusion rows for the
    # survivors only; every client in this case clears that bar, so the row index is the client.
    assert ours["confusion"].shape[0] == len(case.senders)
    row_ours = ours["confusion"][systematic][SYSTEMATIC_FROM].copy()
    row_reference = aligned["confusion"][systematic][SYSTEMATIC_FROM].copy()
    row_ours[SYSTEMATIC_FROM] = row_reference[SYSTEMATIC_FROM] = -np.inf

    row = {
        "case": case.name,
        "items_scored": int(mask.sum()),
        "class_permutation": list(permutation),
        "permutation_is_identity": permutation == tuple(range(NUM_CLASSES)),
        "our_accuracy": float((ours["labels"][mask] == case.truth[mask]).mean()),
        "reference_accuracy": float((aligned["labels"][mask] == case.truth[mask]).mean()),
        "majority_accuracy": float((majority[mask] == case.truth[mask]).mean()),
        "label_agreement": float((ours["labels"][mask] == aligned["labels"][mask]).mean()),
        "mean_posterior_diff": float(posterior_diff.mean()),
        "max_posterior_diff": float(posterior_diff.max()),
        "mean_confusion_diff": float(confusion_diff.mean()),
        "max_confusion_diff": float(confusion_diff.max()),
        "our_systematic_error_target": int(row_ours.argmax()),
        "reference_systematic_error_target": int(row_reference.argmax()),
        "our_iterations": ours["iterations"],
    }
    row["accuracy_gap"] = abs(row["our_accuracy"] - row["reference_accuracy"])
    row["failures"] = _check(row)
    return row


def _check(row: dict) -> list[str]:
    failures = []
    if row["label_agreement"] < TOLERANCES["min_label_agreement"]:
        failures.append(f"label agreement {row['label_agreement']:.4f}")
    if row["accuracy_gap"] > TOLERANCES["max_accuracy_gap"]:
        failures.append(f"accuracy gap {row['accuracy_gap']:.4f}")
    if row["mean_posterior_diff"] > TOLERANCES["max_mean_posterior_diff"]:
        failures.append(f"mean posterior diff {row['mean_posterior_diff']:.4f}")
    if row["max_posterior_diff"] > TOLERANCES["max_max_posterior_diff"]:
        failures.append(f"max posterior diff {row['max_posterior_diff']:.4f}")
    if row["mean_confusion_diff"] > TOLERANCES["max_mean_confusion_diff"]:
        failures.append(f"mean confusion diff {row['mean_confusion_diff']:.4f}")
    if row["our_systematic_error_target"] != row["reference_systematic_error_target"]:
        failures.append(
            "systematic error direction "
            f"{row['our_systematic_error_target']} vs {row['reference_systematic_error_target']}"
        )
    gain = min(row["our_accuracy"], row["reference_accuracy"]) - row["majority_accuracy"]
    if gain < TOLERANCES["min_gain_over_majority"]:
        failures.append(f"gain over majority {gain:.4f}")
    return failures


def invariance_checks(case: Case, rng: np.random.Generator) -> dict:
    """Properties of our estimator alone: rerunning and reordering clients must change nothing."""
    first = run_ours(case)
    repeat = run_ours(case)

    order = rng.permutation(len(case.senders))
    shuffled = Case(
        case.name,
        case.annotations[order],
        case.truth,
        case.true_confusion[order],
        tuple(case.senders[j] for j in order),
    )
    reordered = run_ours(shuffled)

    padded = run_ours(with_all_abstain_items(case))
    kept = slice(0, case.annotations.shape[1])
    return {
        "deterministic": bool(np.array_equal(first["labels"], repeat["labels"])),
        "client_order_invariant": bool(np.array_equal(first["labels"], reordered["labels"])),
        "all_abstain_items_do_not_move_the_fit": bool(
            np.array_equal(first["labels"], padded["labels"][kept])
        ),
        "all_abstain_items_are_invalid": bool(
            (padded["labels"][case.annotations.shape[1] :] == ABSTAIN).all()
        ),
    }


def run_validation() -> dict:
    rng = np.random.default_rng(SEED)
    base = build_case(rng)
    cases = [base, with_missing(base, rng), with_all_abstain_items(base)]
    rows = [compare(case, run_ours(case), run_reference(case)) for case in cases]
    invariance = invariance_checks(base, np.random.default_rng(SEED + 1))
    failures = [f"{row['case']}: {failure}" for row in rows for failure in row["failures"]]
    failures += [f"invariance: {name}" for name, passed in invariance.items() if not passed]
    return {
        "seed": SEED,
        "num_classes": NUM_CLASSES,
        "num_items": NUM_ITEMS,
        "clients": [{"name": name, "kind": kind} for name, kind in CLIENT_SPEC],
        "tolerances": TOLERANCES,
        "rows": rows,
        "invariance": invariance,
        "failures": failures,
        "passed": not failures,
    }


def _print(report: dict) -> None:
    header = (
        f"{'case':<12}{'ours':>8}{'crowdkit':>10}{'majority':>10}"
        f"{'agree':>8}{'post d':>9}{'conf d':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in report["rows"]:
        print(
            f"{row['case']:<12}{row['our_accuracy']:>8.4f}{row['reference_accuracy']:>10.4f}"
            f"{row['majority_accuracy']:>10.4f}{row['label_agreement']:>8.4f}"
            f"{row['mean_posterior_diff']:>9.4f}{row['mean_confusion_diff']:>9.4f}"
        )
    print()
    for row in report["rows"]:
        print(
            f"{row['case']:<12}permutation={row['class_permutation']} "
            f"identity={row['permutation_is_identity']} iterations={row['our_iterations']} "
            f"systematic {SYSTEMATIC_FROM}->{row['our_systematic_error_target']} "
            f"(crowd-kit {SYSTEMATIC_FROM}->{row['reference_systematic_error_target']})"
        )
    print()
    for name, passed in report["invariance"].items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print()
    if report["passed"]:
        print("PASS: both implementations agree within the declared tolerances.")
    else:
        for failure in report["failures"]:
            print(f"FAIL: {failure}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/validation/dawid_skene_reference.json"),
        help="where to write the machine-readable report",
    )
    args = parser.parse_args()

    report = run_validation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    _print(report)
    print(f"\nreport: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
