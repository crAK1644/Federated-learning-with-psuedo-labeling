"""Scenario 4 exists so the controlled experiment has exactly one moving part. These tests are
about that property, not about the allocation being pretty."""

from collections import Counter

import numpy as np
import pytest

from ssfl.data.partition import build_scenario, build_scenario_4

PRIVATE = 60
TARGETS = (3, 4)
DEVICES = {0: [0, 1, 2, 3, 4, 5], 1: [0, 1, 2, 3, 4, 5]}


def assignments(specialization: float, seed: int = 7):
    return build_scenario_4(DEVICES, PRIVATE, seed, TARGETS, specialization)


def test_every_client_holds_the_same_number_of_rows_at_any_specialization():
    """The whole design rests on this: an arm difference must not be a data-volume difference."""
    for specialization in (0.0, 0.25, 0.5, 0.75, 1.0):
        sizes = {a.num_examples for a in assignments(specialization)}
        assert sizes == {PRIVATE}


def test_every_private_row_is_used_exactly_once():
    for specialization in (0.0, 1.0):
        seen = Counter()
        for assignment in assignments(specialization):
            for label, indices in assignment.class_local_indices.items():
                seen.update((assignment.device_id, label, index) for index in indices)
        assert len(seen) == len(DEVICES) * len(DEVICES[0]) * PRIVATE
        assert max(seen.values()) == 1


def test_specialization_concentrates_the_target_pair_and_only_the_target_pair():
    balanced = assignments(0.0)
    specialist = assignments(1.0)

    def holders(assignment_list, label):
        return sum(1 for a in assignment_list if a.class_local_indices.get(label))

    for label in TARGETS:
        assert holders(balanced, label) > holders(specialist, label)
    # A background class must not become concentrated as a side effect, or the arms would differ
    # in more than the one variable the experiment is allowed to move.
    for label in (0, 1, 2, 5):
        assert holders(specialist, label) >= holders(balanced, label) - 1


def test_a_full_specialist_holds_one_target_class_and_not_the_other():
    for assignment in assignments(1.0):
        held = [label for label in TARGETS if assignment.class_local_indices.get(label)]
        assert len(held) == 1


def test_balanced_gives_every_client_a_share_of_both_target_classes():
    for assignment in assignments(0.0):
        for label in TARGETS:
            assert assignment.class_local_indices.get(label)


def test_the_partition_is_deterministic_and_seed_dependent():
    def shape(items):
        return [
            (a.client_id, {label: sorted(v) for label, v in a.class_local_indices.items()})
            for a in items
        ]

    assert shape(assignments(0.5, seed=7)) == shape(assignments(0.5, seed=7))
    assert shape(assignments(0.5, seed=8)) != shape(assignments(0.5, seed=7))


def test_a_device_without_the_target_pair_still_partitions_cleanly():
    """N-BaIoT devices do not all carry every class, and a missing target must not lose rows."""
    result = build_scenario_4({0: [0, 1, 2]}, PRIVATE, 7, (8, 9), 1.0)

    assert {a.num_examples for a in result} == {PRIVATE}
    assert sum(a.num_examples for a in result) == 3 * PRIVATE


def test_specialization_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="specialization"):
        build_scenario_4(DEVICES, PRIVATE, 7, TARGETS, 1.5)


def test_the_dispatcher_reaches_scenario_4():
    direct = build_scenario_4(DEVICES, PRIVATE, 7, TARGETS, 1.0)
    viadispatch = build_scenario(
        4, DEVICES, PRIVATE, 7, dirichlet_alpha=0.1, target_classes=TARGETS, specialization=1.0
    )

    assert [a.client_id for a in viadispatch] == [a.client_id for a in direct]
    assert np.array_equal(
        sorted(viadispatch[0].class_local_indices), sorted(direct[0].class_local_indices)
    )
