"""The stage-7 ledger scores labels against sealed open-set truth, so the arithmetic has to be
right on cases where a plausible mistake would flatter the result."""

import numpy as np
import pandas as pd
import pytest

from scripts.round_metrics import accuracy, first_fallback, pair_slice, round_row

ABSTAIN = -1


def test_abstained_items_are_excluded_rather_than_counted_wrong():
    """Two different failures. An aggregator that abstains on everything it would have got wrong
    must not be scored the same as one that broadcasts those items wrongly."""
    truth = np.array([0, 1, 2, 3])
    labels = np.array([0, 1, ABSTAIN, ABSTAIN])
    valid = np.array([True, True, False, False])

    assert accuracy(labels, truth, valid) == 1.0
    assert accuracy(labels, truth, np.ones(4, dtype=bool)) == 0.5


def test_accuracy_on_an_empty_mask_is_not_a_number():
    """A round with no tie items has no tie accuracy; 0.0 there would drag every average down."""
    empty = np.zeros(3, dtype=bool)

    assert np.isnan(accuracy(np.zeros(3, dtype=int), np.zeros(3, dtype=int), empty))


def test_the_pair_slice_sees_a_one_way_collapse_that_global_accuracy_hides():
    """Every item of class 5 labelled 4, everything else perfect: the shape stage 4 found on real
    data. Global accuracy stays high, and the pair metrics are what has to react."""
    truth = np.array([4] * 10 + [5] * 10 + [0] * 80)
    labels = np.array([4] * 20 + [0] * 80)
    valid = np.ones(100, dtype=bool)

    out = pair_slice(labels, truth, valid, 4, 5)

    assert accuracy(labels, truth, valid) == 0.9
    assert out["pair_a_recall"] == 1.0
    assert out["pair_b_recall"] == 0.0
    assert out["pair_a_precision"] == 0.5
    assert out["pair_confusion_ba"] == 10
    assert out["pair_confusion_ab"] == 0
    # Recall alone would read 0.5 and look ordinary; F1 has to price in the halved precision.
    assert out["pair_macro_f1"] == pytest.approx(1 / 3)


def test_tie_items_are_the_ones_with_no_vote_margin(tmp_path):
    votes = np.array([[3, 0, 0], [2, 2, 0], [1, 1, 1]], dtype=np.int64)
    truth = np.array([0, 1, 2])
    path = tmp_path / "ssfl_aggregation_round_7.npz"
    np.savez(
        path,
        votes_per_class=votes,
        participating_counts=votes.sum(axis=1),
        global_labels=np.array([0, 1, 0], dtype=np.int8),
        valid_mask=np.ones(3, dtype=bool),
    )

    row = round_row(path, truth, 0, 1)

    assert row["round"] == 7
    assert row["tie_count"] == 2
    assert row["tie_accuracy"] == pytest.approx(0.5)
    assert row["broadcast_accuracy"] == pytest.approx(2 / 3)
    # Nothing to compare against in a majority-only audit; the columns stay absent rather than
    # defaulting to the broadcast numbers, which would silently claim a fit that never ran.
    assert "dawid_skene_accuracy" not in row


def test_the_first_fallback_is_the_earliest_one_not_the_last():
    metrics = pd.DataFrame({"round": [1, 2, 3, 4], "ds_status": [0, 3, 0, 14]})

    assert first_fallback(metrics) == {"first_fallback_round": 2, "first_fallback_status": 3}
    assert first_fallback(pd.DataFrame({"round": [1], "ds_status": [0]})) == {
        "first_fallback_round": None,
        "first_fallback_status": None,
    }
