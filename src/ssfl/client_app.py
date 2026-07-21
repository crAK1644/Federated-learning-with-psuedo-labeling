"""Single Flower ``ClientApp``: dispatches ``train``/``evaluate`` messages to one of the four
protocol modules by ``exp_config.algorithm``.

Client identity comes from ``context.node_config["partition-id"]`` (injected by Flower's
simulation engine / set per-SuperNode in deployment), indexing into the scenario's client
assignment manifest. SSFL/FD/DS-FL persist their local model(s) across rounds in
``context.state`` (confirmed real cross-invocation persistence, not just in-process convenience);
FL does not need ``context.state`` since its model arrives fresh over the wire every round.

Round-0 model init for SSFL/FD/DS-FL never crosses the wire: every client independently calls
``seed_everything(seed [+ offset])`` immediately before building a fresh model, so all clients (and
the server) derive bit-identical initial weights from the shared config seed alone.
"""

from __future__ import annotations

import numpy as np
from flwr.clientapp import ClientApp
from flwr.common import ArrayRecord, Context, Message, MetricRecord, RecordDict

from ssfl import training
from ssfl.config import Algorithm, ExperimentConfig, experiment_config_from_run_config
from ssfl.data.datasets import (
    TensorFeatureDataset,
    load_client_assignments,
    load_client_private_data,
    load_open_data,
)
from ssfl.data.partition import ClientAssignment
from ssfl.device import resolve_device
from ssfl.models import SSFLModel, build_classifier, build_discriminator
from ssfl.protocols.dsfl import client_predict_step, distill_step
from ssfl.protocols.fd import client_class_logits_step
from ssfl.protocols.fd import client_distillation_step as fd_client_distillation_step
from ssfl.protocols.fl import client_train_step
from ssfl.protocols.ssfl import AggregationResult, client_distillation_step, client_proposal_step
from ssfl.records import array_record_from_numpy, numpy_from_array_record
from ssfl.seeding import seed_everything

app = ClientApp()


def _seed(base_seed: int, server_round: int, salt: int) -> int:
    return (base_seed * 1_000 + server_round * 10 + salt) % (2**31)


def _partition_client(context: Context, exp_config: ExperimentConfig) -> tuple[str, ClientAssignment]:
    assignments = load_client_assignments(exp_config.data_path, exp_config.scenario.value)
    expected = exp_config.num_clients()
    if len(assignments) != expected:
        raise RuntimeError(
            f"scenario {exp_config.scenario.value} manifest has {len(assignments)} clients, expected {expected}"
        )
    num_partitions = int(context.node_config["num-partitions"])
    if num_partitions != expected:
        raise RuntimeError(
            f"federation num-supernodes={num_partitions} does not match scenario "
            f"{exp_config.scenario.value}'s {expected} clients; set "
            f"--federation-config 'options.num-supernodes={expected}'"
        )
    partition_id = int(context.node_config["partition-id"])
    assignment = assignments[partition_id]
    return assignment.client_id, assignment


def _get_or_init_model(context: Context, key: str, exp_config: ExperimentConfig, seed_offset: int, builder) -> SSFLModel:
    if key in context.state:
        model = builder(exp_config.backbone)
        model.load_state_dict(context.state[key].to_torch_state_dict())
        return model
    seed_everything(exp_config.seed + seed_offset)
    return builder(exp_config.backbone)


def _get_or_init_classifier(context: Context, exp_config: ExperimentConfig) -> SSFLModel:
    return _get_or_init_model(context, "classifier", exp_config, 0, build_classifier)


def _get_or_init_discriminator(context: Context, exp_config: ExperimentConfig) -> SSFLModel:
    return _get_or_init_model(context, "discriminator", exp_config, 1, build_discriminator)


def _save_model(context: Context, key: str, model: SSFLModel) -> None:
    context.state[key] = ArrayRecord(torch_state_dict=model.state_dict())


# ---------------------------------------------------------------------------
# FL
# ---------------------------------------------------------------------------


def _fl_train(exp_config, device, server_round, client_id, private, message) -> Message:
    classifier = build_classifier(exp_config.backbone)
    classifier.load_state_dict(message.content["arrays"].to_torch_state_dict())
    update = client_train_step(
        client_id,
        classifier,
        private,
        device,
        epochs=exp_config.local_epochs,
        lr=exp_config.learning_rate,
        batch_size=exp_config.effective_batch_size,
        seed=_seed(exp_config.seed, server_round, 1),
    )
    reply = RecordDict(
        {
            "arrays": ArrayRecord(torch_state_dict=update.state_dict),
            "metrics": MetricRecord(
                {"loss": update.train_result.final_loss, "num-examples": update.num_examples}
            ),
        }
    )
    return Message(reply, reply_to=message)


def _fl_evaluate(exp_config, device, client_id, private, message) -> Message:
    classifier = build_classifier(exp_config.backbone)
    classifier.load_state_dict(message.content["arrays"].to_torch_state_dict())
    loader = training.make_loader(private, exp_config.effective_batch_size, shuffle=False, seed=exp_config.seed)
    metrics = training.evaluate(classifier, loader, device)
    reply = RecordDict(
        {
            "metrics": MetricRecord(
                {"loss": metrics["loss"], "accuracy": metrics["accuracy"], "num-examples": len(private)}
            )
        }
    )
    return Message(reply, reply_to=message)


# ---------------------------------------------------------------------------
# SSFL
# ---------------------------------------------------------------------------


def _ssfl_train(context, exp_config, device, server_round, client_id, private, message) -> Message:
    classifier = _get_or_init_classifier(context, exp_config)
    discriminator = _get_or_init_discriminator(context, exp_config)
    open_ds = load_open_data(exp_config.data_path, exp_config.backbone)

    result = client_proposal_step(
        client_id=client_id,
        classifier=classifier,
        discriminator=discriminator,
        private_dataset=private,
        open_dataset=open_ds,
        device=device,
        epochs=exp_config.local_epochs,
        lr=exp_config.learning_rate,
        batch_size=exp_config.effective_batch_size,
        threshold_policy=exp_config.ssfl_threshold_policy,
        seed=_seed(exp_config.seed, server_round, 0),
        discriminator_mode=exp_config.ssfl_discriminator_mode,
        label_representation=exp_config.ssfl_label_representation,
        soft_round_decimals=exp_config.ssfl_soft_label_round_decimals,
    )
    _save_model(context, "classifier", classifier)
    _save_model(context, "discriminator", discriminator)

    # Exactly one of pseudo_labels/soft_probs is set (matches ssfl_label_representation); only
    # send the populated one over the wire.
    payload = {"confidences": result.confidences}
    if result.pseudo_labels is not None:
        payload["pseudo_labels"] = result.pseudo_labels
    else:
        payload["soft_probs"] = result.soft_probs

    reply = RecordDict(
        {
            "arrays": array_record_from_numpy(payload),
            "metrics": MetricRecord(
                {
                    "threshold": result.threshold,
                    "classifier_loss": result.classifier_loss,
                    # discriminator_mode != enabled never trains a discriminator; MetricRecord
                    # can't hold None, so a disabled discriminator is unambiguously 0.0 here.
                    "discriminator_loss": result.discriminator_loss
                    if result.discriminator_loss is not None
                    else 0.0,
                    "num-examples": len(private),
                }
            ),
        }
    )
    return Message(reply, reply_to=message)


def _ssfl_evaluate(context, exp_config, device, server_round, client_id, private, message) -> Message:
    classifier = _get_or_init_classifier(context, exp_config)
    open_ds = load_open_data(exp_config.data_path, exp_config.backbone)

    arrays = numpy_from_array_record(message.content["arrays"])
    global_labels = arrays["global_labels"]
    valid_mask = arrays["valid_mask"].astype(bool)
    # ponytail: only .global_labels/.valid_mask are read by client_distillation_step; the rest of
    # AggregationResult is server-only audit data this client never receives, so fill placeholders.
    aggregation = AggregationResult(
        global_labels=global_labels,
        valid_mask=valid_mask,
        votes_per_class=np.zeros((len(global_labels), 1), dtype=np.int64),
        participating_counts=np.zeros(len(global_labels), dtype=np.int64),
        tie_count=0,
        all_abstain_count=0,
        rejected=(),
    )
    result = client_distillation_step(
        classifier,
        open_ds,
        aggregation,
        device,
        epochs=exp_config.local_epochs,
        lr=exp_config.learning_rate,
        batch_size=exp_config.effective_batch_size,
        seed=_seed(exp_config.seed, server_round, 2),
    )
    _save_model(context, "classifier", classifier)

    reply = RecordDict(
        {
            "metrics": MetricRecord(
                {"loss": result.final_loss, "num-examples": int(valid_mask.sum())}
            )
        }
    )
    return Message(reply, reply_to=message)


# ---------------------------------------------------------------------------
# FD
# ---------------------------------------------------------------------------


def _fd_train(context, exp_config, device, server_round, client_id, private, message) -> Message:
    classifier = _get_or_init_classifier(context, exp_config)
    loader = training.make_loader(
        private, exp_config.effective_batch_size, shuffle=True, seed=_seed(exp_config.seed, server_round, 0)
    )
    train_result = training.train_supervised(classifier, loader, device, exp_config.local_epochs, exp_config.learning_rate)
    upload = client_class_logits_step(
        client_id, classifier, private, device, exp_config.effective_batch_size,
        seed=_seed(exp_config.seed, server_round, 1),
    )
    _save_model(context, "classifier", classifier)

    reply = RecordDict(
        {
            "arrays": array_record_from_numpy(
                {"class_probs": upload.class_probs, "class_present": upload.class_present}
            ),
            "metrics": MetricRecord({"loss": train_result.final_loss, "num-examples": len(private)}),
        }
    )
    return Message(reply, reply_to=message)


def _fd_evaluate(context, exp_config, device, server_round, client_id, private, message) -> Message:
    classifier = _get_or_init_classifier(context, exp_config)
    arrays = numpy_from_array_record(message.content["arrays"])
    targets, valid = arrays["targets"], arrays["valid"].astype(bool)

    result = fd_client_distillation_step(
        classifier,
        private,
        targets,
        valid,
        device,
        epochs=exp_config.local_epochs,
        lr=exp_config.learning_rate,
        batch_size=exp_config.effective_batch_size,
        seed=_seed(exp_config.seed, server_round, 2),
    )
    _save_model(context, "classifier", classifier)

    # FD has no server-side model (personalized-per-client by design; see REPRODUCIBILITY.md); the
    # client reports its own post-distillation accuracy on its own private data as a monitoring
    # signal, never a true held-out test-set number.
    eval_loader = training.make_loader(private, exp_config.effective_batch_size, shuffle=False, seed=exp_config.seed)
    eval_metrics = training.evaluate(classifier, eval_loader, device)

    reply = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    "loss": result.final_loss,
                    "private_accuracy": eval_metrics["accuracy"],
                    "num-examples": len(private),
                }
            )
        }
    )
    return Message(reply, reply_to=message)


# ---------------------------------------------------------------------------
# DS-FL
# ---------------------------------------------------------------------------


def _dsfl_train(context, exp_config, device, server_round, client_id, private, message) -> Message:
    classifier = _get_or_init_classifier(context, exp_config)
    loader = training.make_loader(
        private, exp_config.effective_batch_size, shuffle=True, seed=_seed(exp_config.seed, server_round, 0)
    )
    train_result = training.train_supervised(classifier, loader, device, exp_config.local_epochs, exp_config.learning_rate)

    open_ds = load_open_data(exp_config.data_path, exp_config.backbone)
    upload = client_predict_step(
        client_id, classifier, open_ds, device, exp_config.effective_batch_size,
        seed=_seed(exp_config.seed, server_round, 1),
    )
    _save_model(context, "classifier", classifier)

    reply = RecordDict(
        {
            "arrays": array_record_from_numpy({"probs": upload.probs}),
            "metrics": MetricRecord({"loss": train_result.final_loss, "num-examples": len(private)}),
        }
    )
    return Message(reply, reply_to=message)


def _dsfl_evaluate(context, exp_config, device, server_round, client_id, private, message) -> Message:
    classifier = _get_or_init_classifier(context, exp_config)
    open_ds = load_open_data(exp_config.data_path, exp_config.backbone)
    arrays = numpy_from_array_record(message.content["arrays"])

    result = distill_step(
        classifier,
        open_ds,
        arrays["sharpened_targets"],
        device,
        epochs=exp_config.local_epochs,
        lr=exp_config.learning_rate,
        batch_size=exp_config.effective_batch_size,
        seed=_seed(exp_config.seed, server_round, 2),
    )
    _save_model(context, "classifier", classifier)

    reply = RecordDict(
        {"metrics": MetricRecord({"loss": result.final_loss, "num-examples": len(open_ds)})}
    )
    return Message(reply, reply_to=message)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@app.train()
def client_train(message: Message, context: Context) -> Message:
    exp_config = experiment_config_from_run_config(context.run_config)
    device = resolve_device(exp_config.device, exp_config.deterministic)
    server_round = int(message.content["config"]["server-round"])
    client_id, assignment = _partition_client(context, exp_config)
    private: TensorFeatureDataset = load_client_private_data(exp_config.data_path, assignment, exp_config.backbone)

    if exp_config.algorithm is Algorithm.ssfl:
        return _ssfl_train(context, exp_config, device, server_round, client_id, private, message)
    if exp_config.algorithm is Algorithm.fl:
        return _fl_train(exp_config, device, server_round, client_id, private, message)
    if exp_config.algorithm is Algorithm.fd:
        return _fd_train(context, exp_config, device, server_round, client_id, private, message)
    if exp_config.algorithm is Algorithm.dsfl:
        return _dsfl_train(context, exp_config, device, server_round, client_id, private, message)
    raise ValueError(f"unknown algorithm {exp_config.algorithm}")


@app.evaluate()
def client_evaluate(message: Message, context: Context) -> Message:
    exp_config = experiment_config_from_run_config(context.run_config)
    device = resolve_device(exp_config.device, exp_config.deterministic)
    server_round = int(message.content["config"]["server-round"])
    client_id, assignment = _partition_client(context, exp_config)
    private = load_client_private_data(exp_config.data_path, assignment, exp_config.backbone)

    if exp_config.algorithm is Algorithm.ssfl:
        return _ssfl_evaluate(context, exp_config, device, server_round, client_id, private, message)
    if exp_config.algorithm is Algorithm.fl:
        return _fl_evaluate(exp_config, device, client_id, private, message)
    if exp_config.algorithm is Algorithm.fd:
        return _fd_evaluate(context, exp_config, device, server_round, client_id, private, message)
    if exp_config.algorithm is Algorithm.dsfl:
        return _dsfl_evaluate(context, exp_config, device, server_round, client_id, private, message)
    raise ValueError(f"unknown algorithm {exp_config.algorithm}")
