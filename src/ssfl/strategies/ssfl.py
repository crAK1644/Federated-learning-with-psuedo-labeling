"""Custom ``Strategy`` for SSFL: proposal (train exchange) -> majority vote -> broadcast ->
distillation (evaluate exchange), matching ``protocols/ssfl.py`` exactly.

Only pseudo-labels/confidences (client->server) and global_labels/valid_mask (server->client)
cross the wire -- never model parameters, matching the privacy boundary tested in M4.
"""

from __future__ import annotations

from collections.abc import Iterable

from flwr.common import ArrayRecord, ConfigRecord, Message, MessageType, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords, sample_nodes

from ssfl.config import Algorithm, VotingMode
from ssfl.models import NUM_CLASSES
from ssfl.protocols.message import Envelope, ExpectedContext, ProtocolError, validate_envelope
from ssfl.protocols.ssfl import ProposalResult, aggregate_soft, aggregate_votes
from ssfl.records import array_record_from_numpy, numpy_from_array_record


class SSFLStrategy(Strategy):
    def __init__(
        self,
        scenario: int,
        dataset_manifest_hash: str,
        num_open: int,
        num_clients: int,
        voting_mode: VotingMode = VotingMode.enabled,
    ) -> None:
        self.scenario = scenario
        self.dataset_manifest_hash = dataset_manifest_hash
        self.num_open = num_open
        self.num_clients = num_clients
        self.voting_mode = voting_mode
        self._current_node_ids: list[int] = []

    def summary(self) -> None:
        pass  # ponytail: base Strategy.start() already logs round-by-round progress.

    def _all_nodes(self, grid: Grid) -> list[int]:
        node_ids, _ = sample_nodes(grid, self.num_clients, self.num_clients)
        self._current_node_ids = node_ids
        return node_ids

    def configure_train(
        self, server_round: int, arrays, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        node_ids = self._all_nodes(grid)
        config["server-round"] = server_round
        # ponytail: the proposal phase never needs model params or the prior round's broadcast
        # arrays (clients only read config["server-round"]); sending an empty ArrayRecord instead
        # of the framework-carried `arrays` avoids re-shipping the previous evaluate phase's
        # global_labels/valid_mask on every train message from round 2 onward.
        record = RecordDict({"arrays": ArrayRecord(), "config": config})
        return [Message(record, message_type=MessageType.TRAIN, dst_node_id=n) for n in node_ids]

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        expected = ExpectedContext(
            algorithm=Algorithm.ssfl,
            scenario=self.scenario,
            round=server_round,
            phase="proposal",
            dataset_manifest_hash=self.dataset_manifest_hash,
            valid_senders=frozenset(str(n) for n in self._current_node_ids),
        )
        seen_message_ids: set[str] = set()
        proposals: list[tuple[Envelope, ProposalResult]] = []
        for msg in replies:
            if msg.has_error():
                continue
            sender_id = str(msg.metadata.src_node_id)
            envelope = Envelope(
                algorithm=Algorithm.ssfl,
                scenario=self.scenario,
                round=server_round,
                phase="proposal",
                sender_id=sender_id,
                dataset_manifest_hash=self.dataset_manifest_hash,
            )
            try:
                validate_envelope(envelope, expected, seen_message_ids)
            except ProtocolError:
                continue
            seen_message_ids.add(envelope.message_id)

            arrays = numpy_from_array_record(msg.content["arrays"])
            metrics = msg.content["metrics"]
            proposals.append(
                (
                    envelope,
                    ProposalResult(
                        client_id=sender_id,
                        pseudo_labels=arrays.get("pseudo_labels"),
                        soft_probs=arrays.get("soft_probs"),
                        confidences=arrays["confidences"],
                        threshold=float(metrics["threshold"]),
                        classifier_loss=float(metrics["classifier_loss"]),
                        discriminator_loss=float(metrics["discriminator_loss"]),
                    ),
                )
            )

        if not proposals:
            return None, None

        if self.voting_mode == VotingMode.enabled:
            result = aggregate_votes(proposals, num_open=self.num_open, num_classes=NUM_CLASSES)
        else:
            result = aggregate_soft(proposals, num_open=self.num_open, num_classes=NUM_CLASSES)
        arrays_out = array_record_from_numpy(
            {"global_labels": result.global_labels, "valid_mask": result.valid_mask}
        )
        metrics_out = MetricRecord(
            {
                "valid_rate": float(result.valid_mask.mean()),
                "tie_count": result.tie_count,
                "all_abstain_count": result.all_abstain_count,
                "num_proposals": len(proposals),
            }
        )
        return arrays_out, metrics_out

    def configure_evaluate(
        self, server_round: int, arrays, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        config["server-round"] = server_round
        record = RecordDict({"arrays": arrays, "config": config})
        return [Message(record, message_type=MessageType.EVALUATE, dst_node_id=n) for n in self._current_node_ids]

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        contents = [msg.content for msg in replies if not msg.has_error()]
        if not contents:
            return None
        return aggregate_metricrecords(contents, "num-examples")
