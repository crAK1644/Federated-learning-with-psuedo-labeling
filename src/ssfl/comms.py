"""Per-message communication accounting.

``CommsTrackingStrategy`` wraps any concrete ``Strategy`` (custom or stock ``FedAvg``) and
intercepts every ``configure_train``/``aggregate_train``/``configure_evaluate``/
``aggregate_evaluate`` call -- the only places a ``Message`` crosses the client/server wire in
Flower's Message API. Wrapping there means every algorithm gets identical, complete accounting
without instrumenting each protocol's aggregation logic individually, and centralized
``evaluate_fn`` calls (which never leave the ServerApp process) are correctly excluded.

Two byte counts are recorded per message, matching REPRODUCIBILITY.md's guidance to never
force-match a single number to the paper:
  - ``logical_bytes``: raw ndarray payload only (what the algorithm actually needed to send).
  - ``serialized_bytes``: Flower's actual protobuf wire encoding (framing/type overhead included).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from flwr.common import Message, RecordDict
from flwr.common.serde import recorddict_to_proto
from flwr.serverapp.strategy import Strategy


@dataclass(frozen=True)
class CommsRow:
    algorithm: str
    scenario: int
    round: int
    phase: str
    direction: str
    sender: str
    recipient: str
    tensor_names: str
    dtypes: str
    shapes: str
    logical_bytes: int
    serialized_bytes: int


def record_dict_bytes(record: RecordDict) -> tuple[int, int, list[str], list[str], list[str]]:
    names, dtypes, shapes = [], [], []
    logical = 0
    for record_name, array_record in record.array_records.items():
        for array_name, array in array_record.items():
            arr = array.numpy()
            logical += arr.nbytes
            names.append(f"{record_name}.{array_name}")
            dtypes.append(str(arr.dtype))
            shapes.append("x".join(str(d) for d in arr.shape))
    serialized = recorddict_to_proto(record).ByteSize()
    return logical, serialized, names, dtypes, shapes


@dataclass
class CommsLedger:
    rows: list[CommsRow] = field(default_factory=list)

    def record_messages(
        self,
        messages: Iterable[Message],
        algorithm: str,
        scenario: int,
        round: int,
        phase: str,
        direction: str,
    ) -> None:
        for msg in messages:
            if msg.has_error():
                continue
            logical, serialized, names, dtypes, shapes = record_dict_bytes(msg.content)
            if direction == "server_to_client":
                sender, recipient = "server", str(msg.metadata.dst_node_id)
            else:
                sender, recipient = str(msg.metadata.src_node_id), "server"
            self.rows.append(
                CommsRow(
                    algorithm=algorithm,
                    scenario=scenario,
                    round=round,
                    phase=phase,
                    direction=direction,
                    sender=sender,
                    recipient=recipient,
                    tensor_names=",".join(names),
                    dtypes=",".join(dtypes),
                    shapes=",".join(shapes),
                    logical_bytes=logical,
                    serialized_bytes=serialized,
                )
            )

    def write_parquet(self, path: Path) -> None:
        import pandas as pd

        pd.DataFrame([asdict(r) for r in self.rows]).to_parquet(path, index=False)


class CommsTrackingStrategy(Strategy):
    """Delegates aggregation behavior to ``inner`` unchanged; only observes the ``Message``
    lists flowing through the four exchange points to populate ``self.ledger``."""

    def __init__(self, inner: Strategy, algorithm: str, scenario: int) -> None:
        self.inner = inner
        self.algorithm = algorithm
        self.scenario = scenario
        self.ledger = CommsLedger()

    def summary(self) -> None:
        self.inner.summary()

    def configure_train(self, server_round, arrays, config, grid) -> list[Message]:
        messages = list(self.inner.configure_train(server_round, arrays, config, grid))
        self.ledger.record_messages(
            messages, self.algorithm, self.scenario, server_round, "train", "server_to_client"
        )
        return messages

    def aggregate_train(self, server_round, replies: Iterable[Message]):
        replies = list(replies)
        self.ledger.record_messages(
            replies, self.algorithm, self.scenario, server_round, "train", "client_to_server"
        )
        return self.inner.aggregate_train(server_round, replies)

    def configure_evaluate(self, server_round, arrays, config, grid) -> list[Message]:
        messages = list(self.inner.configure_evaluate(server_round, arrays, config, grid))
        self.ledger.record_messages(
            messages, self.algorithm, self.scenario, server_round, "evaluate", "server_to_client"
        )
        return messages

    def aggregate_evaluate(self, server_round, replies: Iterable[Message]):
        replies = list(replies)
        self.ledger.record_messages(
            replies, self.algorithm, self.scenario, server_round, "evaluate", "client_to_server"
        )
        return self.inner.aggregate_evaluate(server_round, replies)


if __name__ == "__main__":
    import numpy as np
    from flwr.common import Array, ArrayRecord, MessageType

    rec = RecordDict({"arrays": ArrayRecord(array_dict={"x": Array.from_numpy_ndarray(np.zeros((4, 5), dtype=np.float32))})})
    logical, serialized, names, dtypes, shapes = record_dict_bytes(rec)
    assert logical == 4 * 5 * 4
    assert serialized > 0
    assert names == ["arrays.x"]

    ledger = CommsLedger()
    msg = Message(rec, message_type=MessageType.TRAIN, dst_node_id=7)
    ledger.record_messages([msg], "ssfl", 1, 1, "train", "server_to_client")
    assert len(ledger.rows) == 1
    assert ledger.rows[0].recipient == "7"
    print("comms.py self-check OK")
