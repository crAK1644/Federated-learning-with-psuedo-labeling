"""Guard for the stage-8 pre-registered criteria.

The criteria themselves are checked against real runs, which do not exist yet. What can be checked
now is the thing that would quietly void them: an evaluator that passes everything.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.controlled_pair_criteria import ARMS, CRITERIA, evaluate, load_arm


def arm_table(**overrides) -> pd.DataFrame:
    """A 50-round ledger that clears every criterion, before overrides are applied."""
    rounds = np.arange(1, 51)
    base = {
        "round": rounds,
        "valid_rate": np.full(50, 0.99),
        "broadcast_accuracy": np.full(50, 0.90),
        "majority_accuracy": np.full(50, 0.90),
        "tie_rate": np.full(50, 0.10),
        "majority_tie_accuracy": np.full(50, 0.40),
        "dawid_skene_tie_accuracy": np.full(50, 0.60),
        "ds_status": np.zeros(50),
        "test_accuracy": np.full(50, 0.90),
        "test_pair_b_recall": np.full(50, 0.50),
    }
    return pd.DataFrame({**base, **overrides})


#: The active arms are the ones a criterion asks to be *better*, so their defaults carry the win.
CLEAN = {name: {} for name in ARMS}
CLEAN["controlled_specialist_active"] = {"test_pair_b_recall": np.full(50, 0.55)}


def arms(**per_arm) -> dict[str, pd.DataFrame]:
    return {name: per_arm.get(name, arm_table(**CLEAN[name])) for name in ARMS}


def failures(results) -> set[str]:
    return {row["criterion"] for row in results if not row["passed"]}


def test_a_clean_experiment_passes_everything():
    assert failures(evaluate(arms())) == set()


@pytest.mark.parametrize(
    ("arm", "column", "value", "criterion"),
    [
        # Shadow mode broadcasting something other than the majority labels voids the comparison.
        (
            "controlled_specialist_shadow",
            "broadcast_accuracy",
            0.88,
            "validity_max_shadow_broadcast_drift",
        ),
        # No ties means no items where the aggregators can differ.
        ("controlled_specialist_shadow", "tie_rate", 0.001, "validity_min_specialist_tie_rate"),
        # An arm that keeps falling back is measuring the fallback.
        ("controlled_specialist_active", "ds_status", 3.0, "validity_min_active_ok_fraction"),
        # The primary hypothesis: Dawid-Skene beating majority where they can disagree.
        (
            "controlled_specialist_shadow",
            "dawid_skene_tie_accuracy",
            0.41,
            "primary_min_ds_tie_advantage",
        ),
        # The pair is why the experiment exists: recall on its weaker class has to move.
        (
            "controlled_specialist_active",
            "test_pair_b_recall",
            0.51,
            "primary_min_weak_class_recall_gain",
        ),
        # Pair recall bought with global accuracy is not a win.
        (
            "controlled_specialist_active",
            "test_accuracy",
            0.80,
            "guardrail_max_accuracy_regression",
        ),
        # Nor is one bought by labelling less of the open set.
        ("controlled_specialist_active", "valid_rate", 0.90, "guardrail_max_valid_rate_regression"),
    ],
)
def test_each_criterion_catches_its_own_failure(arm, column, value, criterion):
    broken = arm_table(**{column: np.full(50, value)})
    assert criterion in failures(evaluate(arms(**{arm: broken})))


def test_a_missing_measurement_fails_rather_than_passes():
    """An arm whose ledger never got the shadow columns must not be scored as a success."""
    blank = arm_table().drop(columns=["dawid_skene_tie_accuracy"])

    results = evaluate(arms(controlled_specialist_shadow=blank))

    row = next(r for r in results if r["criterion"] == "primary_min_ds_tie_advantage")
    assert np.isnan(row["measured"])
    assert not row["passed"]


def test_every_declared_criterion_is_actually_checked():
    """A threshold nobody reads is not pre-registration, it is decoration."""
    assert {row["criterion"] for row in evaluate(arms())} == set(CRITERIA)


def _fake_run(runs: Path, arm: str, seed: int) -> Path:
    """A run directory named the way run_suite actually names them."""
    run_dir = runs / f"ssfl-s4-{arm}-{seed:016x}"
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text(f"profile: {arm}\nscenario: 4\nseed: {seed}\n")
    arm_table().to_parquet(run_dir / "round_metrics.parquet")
    return run_dir


def test_an_arm_is_found_under_the_run_id_that_run_suite_actually_produces(tmp_path):
    """Run directories are <algorithm>-s<scenario>-<profile>-<hash>, so the arm name is in the
    middle. A pattern anchored at the start of the directory name matches nothing at all."""
    _fake_run(tmp_path, "controlled_specialist_majority", 2023)

    assert len(load_arm(tmp_path, "controlled_specialist_majority")) == 50


def test_replicate_seeds_refuse_to_be_averaged_by_accident(tmp_path):
    for seed in (2023, 2024):
        _fake_run(tmp_path, "controlled_specialist_majority", seed)

    with pytest.raises(SystemExit, match="pass --seed"):
        load_arm(tmp_path, "controlled_specialist_majority")

    assert len(load_arm(tmp_path, "controlled_specialist_majority", seed=2024)) == 50

    with pytest.raises(FileNotFoundError, match="at seed 2025"):
        load_arm(tmp_path, "controlled_specialist_majority", seed=2025)
