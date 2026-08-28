"""``dawid_skene_only``: the arm with no majority path.

Experiment 1 (DENEY_1_SCENARIO_3_DENEY_PLANI.md) compares majority, Dawid-Skene-only and the
hybrid on one partition. The comparison is only meaningful if the DS-only arm genuinely never
broadcasts a majority label -- including on the rounds the hybrid arm would have fallen back on --
and if a numerically broken fit stops the run instead of quietly becoming the hybrid arm.
"""

# ruff: noqa: F811 -- the `replies` fixture is imported, then named again as a test parameter.
import numpy as np
import pytest

from ssfl.config import HardAggregation
from ssfl.strategies.ssfl import DS_STATUS_CODES

from tests.unit.test_strategy_dawid_skene import (  # noqa: F401  (fixture import)
    NUM_CLIENTS,
    _broadcast,
    _strategy,
    replies,
)

MAJORITY_ARM = HardAggregation.majority


def _ds_only(**kwargs):
    return _strategy(HardAggregation.dawid_skene_only, class_alignment=True, **kwargs)


def test_ds_only_broadcasts_the_fit_and_never_the_majority_labels(replies):
    majority_arrays, _ = _strategy(HardAggregation.majority).aggregate_train(1, replies)
    arrays, metrics = _ds_only().aggregate_train(1, replies)

    assert metrics["ds_applied"] == 1
    assert not np.array_equal(_broadcast(majority_arrays)[0], _broadcast(arrays)[0])


def test_ds_only_broadcasts_a_non_converged_fit_where_the_hybrid_falls_back(replies):
    """The single most important divergence between the two arms, and the reason both statuses
    exist: same estimator, same round, opposite decision."""
    arrays, _ = _strategy(MAJORITY_ARM).aggregate_train(1, replies)
    majority_labels = _broadcast(arrays)[0]
    cap = dict(max_iterations=1, min_iterations=1)

    hybrid_arrays, hybrid = _strategy(HardAggregation.dawid_skene, **cap).aggregate_train(
        1, replies
    )
    only_arrays, only = _ds_only(**cap).aggregate_train(1, replies)

    assert hybrid["ds_status"] == DS_STATUS_CODES["not_converged"]
    assert hybrid["ds_applied"] == 0
    np.testing.assert_array_equal(_broadcast(hybrid_arrays)[0], majority_labels)

    assert only["ds_status"] == DS_STATUS_CODES["not_converged_but_used"]
    assert only["ds_applied"] == 1
    assert not np.array_equal(_broadcast(only_arrays)[0], majority_labels)


def test_ds_only_broadcasts_a_fit_the_majority_anchored_gate_rejected(replies):
    """A gate scored against majority cannot decide anything in an arm that has no majority path.
    It still sets the status, so the round stays auditable."""
    gate = dict(permutation_min_majority_agreement=1.0)
    arrays, _ = _strategy(MAJORITY_ARM).aggregate_train(1, replies)
    majority_labels = _broadcast(arrays)[0]

    _, hybrid = _strategy(HardAggregation.dawid_skene, **gate).aggregate_train(1, replies)
    arrays, only = _ds_only(**gate).aggregate_train(1, replies)

    assert hybrid["ds_applied"] == 0
    assert only["ds_status"] == DS_STATUS_CODES["permutation_check_agreement"]
    assert only["ds_applied"] == 1
    assert not np.array_equal(_broadcast(arrays)[0], majority_labels)


def test_ds_only_stops_the_run_when_the_fit_produced_no_posterior(replies):
    """Not a fallback and not a skipped round: the comparison is void, so the run must fail."""
    strategy = _ds_only(min_clients=NUM_CLIENTS + 1)
    with pytest.raises(RuntimeError, match="no majority fallback"):
        strategy.aggregate_train(1, replies)


def test_ds_only_and_hybrid_are_handed_the_same_candidate(replies, tmp_path):
    """Section 4.3 of the plan: the two arms must differ in policy only. The audit npz carries the
    candidate rather than the accepted fit, so the two files are directly comparable."""
    hybrid_dir = tmp_path / "hybrid"
    only_dir = tmp_path / "only"
    settings = dict(class_alignment=True)
    _strategy(HardAggregation.dawid_skene, tmp_path=hybrid_dir, **settings).aggregate_train(
        1, replies
    )
    _strategy(HardAggregation.dawid_skene_only, tmp_path=only_dir, **settings).aggregate_train(
        1, replies
    )

    hybrid = np.load(hybrid_dir / "ssfl_aggregation_round_1.npz")
    only = np.load(only_dir / "ssfl_aggregation_round_1.npz")
    np.testing.assert_array_equal(hybrid["dawid_skene_labels"], only["dawid_skene_labels"])
    np.testing.assert_array_equal(hybrid["dawid_skene_valid_mask"], only["dawid_skene_valid_mask"])
    np.testing.assert_array_equal(hybrid["dawid_skene_alignment"], only["dawid_skene_alignment"])


def test_matching_valid_mask_is_a_run_stopper_not_a_metric(replies):
    """Section 6: different masks mean the arms are being scored on different sample sets."""
    strategy = _ds_only(posterior_threshold=0.99)
    strategy.require_matching_valid_mask = True
    with pytest.raises(RuntimeError, match="valid masks differ"):
        strategy.aggregate_train(1, replies)

    permissive = _ds_only()
    permissive.require_matching_valid_mask = True
    _, metrics = permissive.aggregate_train(1, replies)
    assert metrics["ds_valid_mask_match"] == 1


def test_ds_valid_rate_describes_the_labels_that_were_broadcast(replies):
    """Guards a silent reporting hole: `ds_valid_rate`/`ds_disagreement_rate` used to read the
    *accepted* fit, which stays empty on every non-ok path -- so the one arm whose labels they
    describe reported 0.0 coverage on exactly the rounds worth reading."""
    cap = dict(max_iterations=1, min_iterations=1)
    arrays, metrics = _ds_only(**cap).aggregate_train(1, replies)
    _, broadcast_mask = _broadcast(arrays)

    assert metrics["ds_status"] == DS_STATUS_CODES["not_converged_but_used"]
    assert metrics["ds_valid_rate"] == pytest.approx(float(broadcast_mask.mean()))
    assert metrics["ds_valid_rate"] > 0.0
