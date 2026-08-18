"""Scenario 1/2/3 non-IID client partitioning over the private split.

All three scenarios operate purely on structure (which labels each device has, and the constant
per-class private row count) — never on raw features — so a manifest can be produced, inspected,
and unit-tested without touching the 7.6GB source data.

Algorithm (assumptions register #13, ``REPRODUCIBILITY.md``):

- **Scenarios 1 and 2** use the classic McMahan-style shard partition: concatenate a device's
  private rows sorted by label, cut into ``clients_per_device * shards_per_client`` equal
  contiguous shards, deterministically shuffle shard order, deal ``shards_per_client`` shards to
  each client. Scenario 1 = 3 clients/device, 2 shards/client (6 shards; shards straddle class
  boundaries since 700-row classes don't divide evenly into sixths -> mild label mixing).
  Scenario 2 = ``num_classes`` clients/device, 2 shards/client (``2*num_classes`` shards; each
  shard is exactly half of one class's 700 rows since it divides evenly -> shards are class-pure,
  giving materially more severe label skew than scenario 1 even though both shuffle-and-deal).
- **Scenario 3** reuses scenario 2's client-per-device count, but allocates each class's 700 rows
  across that device's clients via ``Dirichlet(alpha)`` proportions -> multinomial counts,
  redrawing (deterministically, via an incrementing attempt salt) until every client has at least
  one example.

- **Scenario 4** is not a paper scenario. It exists for the controlled Dawid-Skene experiment
  (``DAWID_SKENE_SONRAKI_ADIMLAR.md`` section 5), where the only variable allowed to move between
  arms is how specialized the clients are on one target class pair. It reuses scenario 2/3's
  client count and gives every client exactly ``private_count`` rows at any specialization level,
  so "balanced" and "specialist" runs differ in composition and in nothing else. It is opt-in:
  ``DataPrepConfig.extra_scenarios`` is empty by default, so the standard prepared dataset and
  therefore every existing ``run_id`` is unaffected.

Every private example is used by exactly one client; examples never cross devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_MAX_DIRICHLET_ATTEMPTS = 1000


@dataclass(frozen=True)
class ClientAssignment:
    client_id: str
    device_id: int
    scenario: int
    class_local_indices: dict[int, list[int]] = field(default_factory=dict)

    @property
    def num_examples(self) -> int:
        return sum(len(v) for v in self.class_local_indices.values())


def _shard_partition_one_device(
    device_id: int,
    labels: list[int],
    private_count: int,
    seed: int,
    clients_per_device: int,
    shards_per_client: int,
    scenario: int,
) -> list[ClientAssignment]:
    labels_sorted = sorted(labels)
    flat: list[tuple[int, int]] = [
        (label, local_idx) for label in labels_sorted for local_idx in range(private_count)
    ]
    total_rows = len(flat)
    total_shards = clients_per_device * shards_per_client

    shard_index_groups = np.array_split(np.arange(total_rows), total_shards)
    shards: list[list[tuple[int, int]]] = [[flat[i] for i in group] for group in shard_index_groups]

    seq = np.random.SeedSequence([seed, device_id, scenario])
    rng = np.random.default_rng(seq)
    shard_order = np.arange(total_shards)
    rng.shuffle(shard_order)

    assignments: list[ClientAssignment] = []
    for c in range(clients_per_device):
        dealt = shard_order[c * shards_per_client : (c + 1) * shards_per_client]
        class_local_indices: dict[int, list[int]] = {}
        for shard_id in dealt:
            for label, local_idx in shards[shard_id]:
                class_local_indices.setdefault(label, []).append(local_idx)
        assignments.append(
            ClientAssignment(
                client_id=f"s{scenario}-d{device_id}-c{c}",
                device_id=device_id,
                scenario=scenario,
                class_local_indices=class_local_indices,
            )
        )
    return assignments


def build_scenario_1(
    devices: dict[int, list[int]], private_count: int, seed: int
) -> list[ClientAssignment]:
    out: list[ClientAssignment] = []
    for device_id, labels in sorted(devices.items()):
        out.extend(
            _shard_partition_one_device(
                device_id, labels, private_count, seed,
                clients_per_device=3, shards_per_client=2, scenario=1,
            )
        )
    return out


def build_scenario_2(
    devices: dict[int, list[int]], private_count: int, seed: int
) -> list[ClientAssignment]:
    out: list[ClientAssignment] = []
    for device_id, labels in sorted(devices.items()):
        out.extend(
            _shard_partition_one_device(
                device_id, labels, private_count, seed,
                clients_per_device=len(labels), shards_per_client=2, scenario=2,
            )
        )
    return out


def build_scenario_3(
    devices: dict[int, list[int]], private_count: int, seed: int, dirichlet_alpha: float
) -> list[ClientAssignment]:
    out: list[ClientAssignment] = []
    for device_id, labels in sorted(devices.items()):
        labels_sorted = sorted(labels)
        num_clients = len(labels_sorted)

        for attempt in range(1, _MAX_DIRICHLET_ATTEMPTS + 1):
            seq = np.random.SeedSequence([seed, device_id, 3, attempt])
            rng = np.random.default_rng(seq)
            client_totals = np.zeros(num_clients, dtype=int)
            per_client: list[dict[int, list[int]]] = [dict() for _ in range(num_clients)]

            for label in labels_sorted:
                proportions = rng.dirichlet(dirichlet_alpha * np.ones(num_clients))
                counts = rng.multinomial(private_count, proportions)
                idx_pool = rng.permutation(private_count)
                cursor = 0
                for c in range(num_clients):
                    n = int(counts[c])
                    if n > 0:
                        per_client[c][label] = idx_pool[cursor : cursor + n].tolist()
                    cursor += n
                    client_totals[c] += n

            if np.all(client_totals > 0):
                break
        else:
            raise RuntimeError(
                f"scenario 3: device {device_id} failed to allocate every client at least one "
                f"example after {_MAX_DIRICHLET_ATTEMPTS} deterministic attempts "
                f"(dirichlet_alpha={dirichlet_alpha} too skewed for {num_clients} clients)"
            )

        for c in range(num_clients):
            out.append(
                ClientAssignment(
                    client_id=f"s3-d{device_id}-c{c}",
                    device_id=device_id,
                    scenario=3,
                    class_local_indices=per_client[c],
                )
            )
    return out


def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Round ``weights`` (which already sum to ``total``) to integers summing to exactly ``total``.

    Plain rounding loses or invents rows, and either one breaks the invariant that every private
    example belongs to exactly one client. Ties go to the lower index, so this is deterministic.
    """
    floored = np.floor(weights).astype(int)
    remainder = total - int(floored.sum())
    if remainder > 0:
        order = np.argsort(-(weights - floored), kind="stable")
        floored[order[:remainder]] += 1
    return floored


def _target_counts(
    private_count: int, num_clients: int, specialists: np.ndarray, specialization: float
) -> np.ndarray:
    """How many rows of one target class each client gets.

    ``specialization`` interpolates between a flat split (0.0) and giving the class entirely to
    its specialists (1.0). One continuous knob rather than two partition algorithms: the balanced
    and specialist arms then differ by a number, and nothing in the code path distinguishes them.
    """
    weights = np.full(num_clients, (1.0 - specialization) * private_count / num_clients)
    if len(specialists):
        weights[specialists] += specialization * private_count / len(specialists)
    return _largest_remainder(weights, private_count)


def build_scenario_4(
    devices: dict[int, list[int]],
    private_count: int,
    seed: int,
    target_classes: tuple[int, ...],
    specialization: float,
) -> list[ClientAssignment]:
    if not 0.0 <= specialization <= 1.0:
        raise ValueError(f"specialization must be in [0, 1], got {specialization}")
    out: list[ClientAssignment] = []
    for device_id, labels in sorted(devices.items()):
        labels_sorted = sorted(labels)
        num_clients = len(labels_sorted)
        rng = np.random.default_rng(np.random.SeedSequence([seed, device_id, 4]))

        present = [label for label in target_classes if label in labels_sorted]
        per_client: list[dict[int, list[int]]] = [dict() for _ in range(num_clients)]
        # Every client gets exactly this many rows whatever the specialization, so an accuracy
        # difference between arms cannot be a difference in how much data a client holds.
        budget = np.full(num_clients, private_count)

        for rank, label in enumerate(present):
            # Specialists are strided rather than contiguous: with a contiguous split the two
            # target classes would also be split along the device's client index, and any other
            # index-correlated effect would ride along with the specialization.
            specialists = np.arange(rank, num_clients, len(present))
            counts = _target_counts(private_count, num_clients, specialists, specialization)
            pool = rng.permutation(private_count)
            cursor = 0
            for client in range(num_clients):
                take = int(counts[client])
                if take:
                    per_client[client][label] = pool[cursor : cursor + take].tolist()
                cursor += take
                budget[client] -= take

        # The background classes fill the rest. Their supply equals the remaining budget exactly,
        # so a shuffled deal is both feasible and free of any per-class structure of its own --
        # the point of the scenario is that background composition is noise, not signal.
        background = [
            (label, index)
            for label in labels_sorted
            if label not in present
            for index in range(private_count)
        ]
        if len(background) != int(budget.sum()):
            raise RuntimeError(
                f"scenario 4: device {device_id} has {len(background)} background rows for "
                f"{int(budget.sum())} unfilled slots"
            )
        dealt = rng.permutation(len(background))
        cursor = 0
        for client in range(num_clients):
            for position in dealt[cursor : cursor + int(budget[client])]:
                label, index = background[position]
                per_client[client].setdefault(label, []).append(index)
            cursor += int(budget[client])

        for client in range(num_clients):
            out.append(
                ClientAssignment(
                    client_id=f"s4-d{device_id}-c{client}",
                    device_id=device_id,
                    scenario=4,
                    class_local_indices=per_client[client],
                )
            )
    return out


def build_scenario(
    scenario: int,
    devices: dict[int, list[int]],
    private_count: int,
    seed: int,
    dirichlet_alpha: float,
    target_classes: tuple[int, ...] = (),
    specialization: float = 0.0,
) -> list[ClientAssignment]:
    if scenario == 1:
        return build_scenario_1(devices, private_count, seed)
    if scenario == 2:
        return build_scenario_2(devices, private_count, seed)
    if scenario == 3:
        return build_scenario_3(devices, private_count, seed, dirichlet_alpha)
    if scenario == 4:
        return build_scenario_4(devices, private_count, seed, target_classes, specialization)
    raise ValueError(f"unknown scenario {scenario}")
