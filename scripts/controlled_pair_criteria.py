"""Pre-registered success criteria for the controlled target-pair experiment (stage 8).

WRITTEN BEFORE THE RUN. That is the whole point of the file: the thresholds in ``CRITERIA`` are
committed while no arm has been executed, so nothing here can be tuned after the fact to make a
result look like a win. Read the git history if you doubt it -- this file must predate every run
directory it is ever pointed at.

Usage, after ``run_suite.py`` finishes the matrix and ``round_metrics.py`` has been run per arm:

    uv run python -m scripts.controlled_pair_criteria --runs artifacts/runs

It exits non-zero if any criterion fails, and prints one line per criterion either way.

Three groups, in the order they are checked. VALIDITY comes first and is not about Dawid-Skene at
all: it asks whether the experiment ran at all. A failed validity gate voids the comparison rather
than deciding it -- if the specialist distribution never produced a tie, or shadow mode changed a
broadcast byte, then neither a positive nor a negative primary result means anything. PRIMARY is
the pre-registered definition of "Dawid-Skene helped here". GUARDRAIL is what a win is not allowed
to cost.

Every threshold clears the run-to-run noise floor measured in REPRODUCIBILITY.md #34 (about half a
point on test accuracy at this scale), because a criterion inside the noise band is not a criterion.
The single-seed pilot cannot establish an effect on its own; these gates are what decides whether
five more seeds are worth the GPU time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

#: Arm directory names, from configs/experiments_controlled_pair.yaml.
ARMS = (
    "controlled_specialist_majority",
    "controlled_specialist_shadow",
    "controlled_specialist_active",
    "controlled_balanced_majority",
    "controlled_balanced_shadow",
    "controlled_balanced_active",
)

#: Rounds the primary comparison is read over. Early rounds are the model finding its feet, and a
#: pseudo-label aggregator has nothing to aggregate until clients disagree in a stable way.
SETTLED_FROM = 10

CRITERIA = {
    # --- validity: did the experiment happen? --------------------------------------------------
    # An arm that abstains on most of the open set is not producing a comparable label set.
    "validity_min_valid_rate": 0.95,
    # Shadow mode must change no broadcast byte (config.py HardAggregation docstring). Its
    # broadcast labels and its majority labels are the same array, so this gap is exactly zero.
    "validity_max_shadow_broadcast_drift": 0.0,
    # The specialist distribution has to actually put clients into conflict on the target pair.
    # No ties, no disagreement for an aggregator to resolve, no experiment.
    "validity_min_specialist_tie_rate": 0.02,
    # The active arm must mostly be running its own estimator. An arm that falls back to majority
    # in half its rounds is measuring the fallback policy, not Dawid-Skene.
    "validity_min_active_ok_fraction": 0.80,
    # --- primary: did it help? -----------------------------------------------------------------
    # Tie items are where the two aggregators can differ at all, and where stage 3 measured a large
    # synthetic advantage. Shadow arm, specialist distribution, mean over settled rounds.
    "primary_min_ds_tie_advantage": 0.05,
    # The pair is the reason this experiment exists: recall on the weaker of the two classes,
    # sealed test set, active vs majority, mean over the last ten rounds.
    "primary_min_weak_class_recall_gain": 0.02,
    # --- guardrail: what it may not cost -------------------------------------------------------
    # Pair recall bought with global accuracy is not a win.
    "guardrail_max_accuracy_regression": 0.005,
    # Nor is it a win if it comes from labelling less of the open set.
    "guardrail_max_valid_rate_regression": 0.01,
}

#: What the pilot decides, written down with the thresholds so the answer is not chosen later.
DECISION = """\
Build the stage-6 hybrid only if primary_min_ds_tie_advantage passes AND every validity gate
passes. A tie advantage is the one measurement that a reliability weight could plausibly convert
into a broadcast gain, because ties are the only items where client weights change the argmax.
If the tie advantage fails, the hybrid is not built at any lambda -- weighting cannot rescue an
estimator that is worse than the arbitrary tie rule it would replace, and REPRODUCIBILITY.md #33
and #37 already say so on the paper scenarios.
Run five more seeds only if the primary criteria pass on this pilot.\
"""


def load_arm(runs_root: Path, name: str) -> pd.DataFrame:
    """One arm's per-round ledger, produced by scripts/round_metrics.py."""
    matches = sorted(runs_root.glob(f"**/{name}*/round_metrics.parquet"))
    if not matches:
        raise FileNotFoundError(f"no round_metrics.parquet for arm {name!r} under {runs_root}")
    return pd.read_parquet(matches[-1])


def settled(table: pd.DataFrame, last: int = 0) -> pd.DataFrame:
    """Rounds the comparison is read over: from ``SETTLED_FROM``, optionally the final ``last``."""
    window = table[table["round"] >= SETTLED_FROM]
    return window.tail(last) if last else window


def _mean(table: pd.DataFrame, column: str) -> float:
    """NaN rather than a KeyError for a column an arm legitimately does not have."""
    return float(table[column].mean()) if column in table else float("nan")


def evaluate(arms: dict[str, pd.DataFrame]) -> list[dict]:
    """Check every criterion. A NaN measurement fails: absent evidence is not evidence."""
    specialist_shadow = settled(arms["controlled_specialist_shadow"])
    specialist_active = settled(arms["controlled_specialist_active"])
    specialist_majority = settled(arms["controlled_specialist_majority"])
    final_active = settled(arms["controlled_specialist_active"], last=10)
    final_majority = settled(arms["controlled_specialist_majority"], last=10)

    checks = [
        (
            "validity_min_valid_rate",
            min(_mean(settled(table), "valid_rate") for table in arms.values()),
            "min",
        ),
        (
            "validity_max_shadow_broadcast_drift",
            abs(
                _mean(specialist_shadow, "broadcast_accuracy")
                - _mean(specialist_shadow, "majority_accuracy")
            ),
            "max",
        ),
        ("validity_min_specialist_tie_rate", _mean(specialist_shadow, "tie_rate"), "min"),
        (
            "validity_min_active_ok_fraction",
            float((specialist_active["ds_status"] == 0).mean())
            if "ds_status" in specialist_active
            else float("nan"),
            "min",
        ),
        (
            "primary_min_ds_tie_advantage",
            _mean(specialist_shadow, "dawid_skene_tie_accuracy")
            - _mean(specialist_shadow, "majority_tie_accuracy"),
            "min",
        ),
        (
            "primary_min_weak_class_recall_gain",
            _mean(final_active, "test_pair_b_recall") - _mean(final_majority, "test_pair_b_recall"),
            "min",
        ),
        (
            "guardrail_max_accuracy_regression",
            _mean(final_majority, "test_accuracy") - _mean(final_active, "test_accuracy"),
            "max",
        ),
        (
            "guardrail_max_valid_rate_regression",
            _mean(specialist_majority, "valid_rate") - _mean(specialist_active, "valid_rate"),
            "max",
        ),
    ]

    results = []
    for name, measured, direction in checks:
        threshold = CRITERIA[name]
        passed = measured >= threshold if direction == "min" else measured <= threshold
        results.append(
            {
                "criterion": name,
                "measured": measured,
                "threshold": threshold,
                "direction": direction,
                "passed": bool(passed),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()

    arms = {name: load_arm(args.runs, name) for name in ARMS}
    results = evaluate(arms)
    for row in results:
        mark = "PASS" if row["passed"] else "FAIL"
        comparator = ">=" if row["direction"] == "min" else "<="
        print(
            f"{mark} {row['criterion']}: {row['measured']:.4f} {comparator} {row['threshold']:.4f}"
        )
    print()
    print(DECISION)
    sys.exit(0 if all(row["passed"] for row in results) else 1)


if __name__ == "__main__":
    main()
