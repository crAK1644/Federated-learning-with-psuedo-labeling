"""``SSFLStrategy`` aggregation-mode selector: majority / shadow / active Dawid-Skene.

The estimator itself is covered in ``tests/protocol/test_dawid_skene.py``. What is proven here is
the seam: shadow mode is a real control (it changes no broadcast byte), active mode broadcasts the
Dawid-Skene labels only when the fit succeeded, and every failure path lands back on majority.
"""

import numpy as np
import pytest
from flwr.app.metadata import Metadata
from flwr.common import Message, MetricRecord, RecordDict

from ssfl.config import HardAggregation, VotingMode
from ssfl.protocols.dawid_skene import DawidSkeneSettings
from ssfl.records import array_record_from_numpy, numpy_from_array_record
from ssfl.strategies.ssfl import DS_STATUS_CODES, SSFLStrategy

NUM_CLASSES = 11
NUM_OPEN = 300
NUM_CLIENTS = 9


def _reply(labels: np.ndarray, src_node_id: int) -> Message:
    meta = Metadata(
        run_id=0,
        message_id=f"m{src_node_id}",
        src_node_id=src_node_id,
        dst_node_id=0,
        reply_to_message_id="",
        group_id="",
        created_at=0.0,
        ttl=100.0,
        message_type="train",
    )
    content = RecordDict(
        {
            "arrays": array_record_from_numpy({"pseudo_labels": labels.astype(np.int8)}),
            "metrics": MetricRecord(
                {"threshold": 0.5, "classifier_loss": 0.1, "discriminator_loss": 0.2}
            ),
        }
    )
    return Message(content=content, metadata=meta)


@pytest.fixture
def replies() -> list[Message]:
    """Two reliable clients outvoted by seven unreliable ones -- majority's worst case."""
    rng = np.random.default_rng(2023)
    truth = rng.integers(0, NUM_CLASSES, size=NUM_OPEN)
    accuracies = [0.95, 0.95] + [0.3] * (NUM_CLIENTS - 2)
    out = []
    for node_id, accuracy in enumerate(accuracies, start=1):
        wrong = (truth + rng.integers(1, NUM_CLASSES, size=NUM_OPEN)) % NUM_CLASSES
        labels = np.where(rng.random(NUM_OPEN) < accuracy, truth, wrong)
        out.append(_reply(labels, src_node_id=node_id))
    return out


def _strategy(mode: HardAggregation, tmp_path=None, **kwargs) -> SSFLStrategy:
    strategy = SSFLStrategy(
        scenario=1,
        dataset_manifest_hash="h",
        num_open=NUM_OPEN,
        num_clients=NUM_CLIENTS,
        voting_mode=VotingMode.enabled,
        audit_dir=tmp_path,
        hard_aggregation=mode,
        dawid_skene_settings=DawidSkeneSettings(**kwargs),
    )
    strategy._current_node_ids = list(range(1, NUM_CLIENTS + 1))
    return strategy


def _broadcast(arrays):
    out = numpy_from_array_record(arrays)
    return out["global_labels"], out["valid_mask"]


def test_majority_mode_emits_no_dawid_skene_metrics(replies):
    _, metrics = _strategy(HardAggregation.majority).aggregate_train(1, replies)
    assert not [key for key in metrics.keys() if key.startswith("ds_")]


def test_shadow_broadcasts_exactly_what_majority_would(replies):
    """The bit-identity gate: shadow is only a control if it perturbs nothing downstream."""
    majority_arrays, majority_metrics = _strategy(HardAggregation.majority).aggregate_train(
        1, replies
    )
    shadow_arrays, shadow_metrics = _strategy(HardAggregation.dawid_skene_shadow).aggregate_train(
        1, replies
    )

    majority_labels, majority_mask = _broadcast(majority_arrays)
    shadow_labels, shadow_mask = _broadcast(shadow_arrays)
    assert np.array_equal(majority_labels, shadow_labels)
    assert np.array_equal(majority_mask, shadow_mask)
    # Every metric majority reports keeps its exact value; shadow only adds ds_* diagnostics.
    for key in majority_metrics.keys():
        assert shadow_metrics[key] == majority_metrics[key], key
    assert shadow_metrics["ds_status"] == DS_STATUS_CODES["ok"]
    assert shadow_metrics["ds_applied"] == 0
    assert shadow_metrics["ds_disagreement_rate"] > 0.0


def test_active_mode_broadcasts_dawid_skene_labels(replies):
    majority_arrays, _ = _strategy(HardAggregation.majority).aggregate_train(1, replies)
    active_arrays, active_metrics = _strategy(HardAggregation.dawid_skene).aggregate_train(
        1, replies
    )
    assert active_metrics["ds_status"] == DS_STATUS_CODES["ok"]
    assert active_metrics["ds_applied"] == 1
    assert not np.array_equal(_broadcast(majority_arrays)[0], _broadcast(active_arrays)[0])


def test_active_mode_falls_back_to_majority_when_the_fit_fails(replies):
    majority_arrays, _ = _strategy(HardAggregation.majority).aggregate_train(1, replies)
    # min_clients above the number of senders is the cheapest deterministic failure to force.
    strategy = _strategy(HardAggregation.dawid_skene, min_clients=NUM_CLIENTS + 1)
    arrays, metrics = strategy.aggregate_train(1, replies)
    assert metrics["ds_status"] == DS_STATUS_CODES["insufficient_clients"]
    assert metrics["ds_applied"] == 0
    assert np.array_equal(_broadcast(majority_arrays)[0], _broadcast(arrays)[0])


def test_warmup_rounds_use_majority_and_say_so(replies):
    strategy = _strategy(HardAggregation.dawid_skene)
    strategy.dawid_skene_warmup_rounds = 3
    majority_arrays, _ = _strategy(HardAggregation.majority).aggregate_train(3, replies)

    _, warm = strategy.aggregate_train(3, replies)
    assert warm["ds_status"] == DS_STATUS_CODES["warmup"]
    assert warm["ds_applied"] == 0

    arrays, after = strategy.aggregate_train(4, replies)
    assert after["ds_status"] == DS_STATUS_CODES["ok"]
    assert after["ds_applied"] == 1
    assert not np.array_equal(_broadcast(majority_arrays)[0], _broadcast(arrays)[0])


def test_annotations_are_off_by_default_and_opt_in_by_round(replies, tmp_path):
    strategy = _strategy(HardAggregation.dawid_skene_shadow, tmp_path=tmp_path)
    strategy.aggregate_train(1, replies)
    audit = np.load(tmp_path / "ssfl_aggregation_round_1.npz")
    assert "annotations" not in audit
    assert audit["dawid_skene_status"] == "ok"

    strategy.save_annotations = True
    strategy.annotation_rounds = (2,)
    strategy.aggregate_train(1, replies)
    strategy.aggregate_train(2, replies)
    assert "annotations" not in np.load(tmp_path / "ssfl_aggregation_round_1.npz")
    saved = np.load(tmp_path / "ssfl_aggregation_round_2.npz")["annotations"]
    assert saved.shape == (NUM_CLIENTS, NUM_OPEN)
    assert saved.dtype == np.int8


def test_dawid_skene_consumes_no_random_state(replies):
    """Active mode must not draw from NumPy's global RNG, or the arms diverge for a reason that
    has nothing to do with the aggregator."""
    np.random.seed(1234)
    before = np.random.get_state()[2]
    _strategy(HardAggregation.dawid_skene).aggregate_train(1, replies)
    assert np.random.get_state()[2] == before
