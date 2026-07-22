"""Flower ``ServerApp`` entrypoint: builds the per-algorithm ``Strategy`` and drives
``strategy.start(...)``, which runs the full round loop (train exchange -> aggregate -> evaluate
exchange -> aggregate -> optional centralized eval) for all four algorithms uniformly.

FL reuses the built-in ``FedAvg`` unmodified (a stock aggregation pattern with no SSFL-specific
safety property to enforce beyond what FedAvg's own reply-consistency checks already guarantee).
SSFL/FD/DS-FL use the custom strategies in ``strategies/`` since none of them ever transmits model
parameters, which ``FedAvg`` assumes.
"""

from __future__ import annotations

import json

from flwr.common import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from ssfl.comms import CommsTrackingStrategy
from ssfl.config import Algorithm, experiment_config_from_run_config
from ssfl.data.datasets import load_open_data, load_test_data
from ssfl.device import resolve_device
from ssfl.logging_utils import bind, configure_logging, log_event
from ssfl.metrics import MetricsLedger, compute_classification_metrics
from ssfl.models import NUM_CLASSES, build_classifier
from ssfl.protocols import dsfl as dsfl_protocol
from ssfl.protocols import fl as fl_protocol
from ssfl.protocols.ssfl import AggregationResult, server_distillation_step
from ssfl.records import numpy_from_array_record
from ssfl.run_context import RunContext
from ssfl.seeding import configure_determinism, seed_everything
from ssfl.strategies.dsfl import DSFLStrategy
from ssfl.strategies.fd import FDStrategy
from ssfl.strategies.ssfl import SSFLStrategy

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    exp_config = experiment_config_from_run_config(context.run_config)
    device = resolve_device(exp_config.device, exp_config.deterministic)
    configure_determinism(exp_config.deterministic)

    manifest_path = exp_config.data_path / "dataset_manifest.json"
    manifest_hash = None
    if manifest_path.exists():
        manifest_hash = json.loads(manifest_path.read_text()).get("manifest_hash")

    run_context = RunContext.create(exp_config, dataset_manifest_path=manifest_path)
    metrics_ledger = MetricsLedger()

    events_stream = open(run_context.events_path, "a", buffering=1)
    events_logger = bind(
        configure_logging(stream=events_stream),
        run_id=run_context.run_id,
        algorithm=exp_config.algorithm.value,
        scenario=exp_config.scenario.value,
    )
    log_event(events_logger, "run_start", dataset_hash=manifest_hash)

    train_config = ConfigRecord()
    evaluate_config = ConfigRecord()

    if exp_config.algorithm is Algorithm.fl:
        seed_everything(exp_config.seed)
        initial_arrays = ArrayRecord(torch_state_dict=build_classifier(exp_config.backbone).state_dict())
        strategy = FedAvg(
            fraction_train=1.0,
            fraction_evaluate=1.0,
            min_train_nodes=exp_config.num_clients(),
            min_evaluate_nodes=exp_config.num_clients(),
            min_available_nodes=exp_config.num_clients(),
        )
        test_dataset = load_test_data(exp_config.data_path, exp_config.backbone)
        server_classifier = build_classifier(exp_config.backbone)

        def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
            eval_metrics = fl_protocol.server_evaluate(
                server_classifier,
                arrays.to_torch_state_dict(),
                test_dataset,
                device,
                batch_size=exp_config.effective_batch_size,
                seed=exp_config.seed,
            )
            classification = compute_classification_metrics(
                eval_metrics["y_true"], eval_metrics["y_pred"], NUM_CLASSES
            )
            metrics_ledger.record(
                algorithm=exp_config.algorithm.value,
                scenario=exp_config.scenario.value,
                round=server_round,
                loss=eval_metrics["loss"],
                metrics=classification,
            )
            return MetricRecord({"loss": eval_metrics["loss"], "accuracy": eval_metrics["accuracy"]})

    elif exp_config.algorithm is Algorithm.ssfl:
        if manifest_hash is None:
            raise RuntimeError(f"dataset_manifest.json missing or unhashed at {manifest_path}")
        open_dataset = load_open_data(exp_config.data_path, exp_config.backbone)
        test_dataset = load_test_data(exp_config.data_path, exp_config.backbone)
        strategy = SSFLStrategy(
            scenario=exp_config.scenario.value,
            dataset_manifest_hash=manifest_hash,
            num_open=len(open_dataset),
            num_clients=exp_config.num_clients(),
            voting_mode=exp_config.ssfl_voting_mode,
        )
        initial_arrays = ArrayRecord()
        seed_everything(exp_config.seed)
        server_classifier = build_classifier(exp_config.backbone)

        def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
            if server_round == 0:
                return None
            arrays_np = numpy_from_array_record(arrays)
            valid_mask = arrays_np["valid_mask"].astype(bool)
            aggregation = AggregationResult(
                global_labels=arrays_np["global_labels"],
                valid_mask=valid_mask,
                votes_per_class=None,
                participating_counts=None,
                tie_count=0,
                all_abstain_count=0,
                rejected=(),
            )
            train_result, eval_metrics = server_distillation_step(
                server_classifier,
                open_dataset,
                aggregation,
                test_dataset,
                device,
                epochs=exp_config.local_epochs,
                lr=exp_config.learning_rate,
                batch_size=exp_config.effective_batch_size,
                seed=exp_config.seed,
            )
            classification = compute_classification_metrics(
                eval_metrics["y_true"], eval_metrics["y_pred"], NUM_CLASSES
            )
            metrics_ledger.record(
                algorithm=exp_config.algorithm.value,
                scenario=exp_config.scenario.value,
                round=server_round,
                loss=eval_metrics["loss"],
                train_loss=train_result.final_loss,
                metrics=classification,
                valid_rate=float(valid_mask.mean()),
            )
            return MetricRecord(
                {"loss": eval_metrics["loss"], "accuracy": eval_metrics["accuracy"], "train_loss": train_result.final_loss}
            )

    elif exp_config.algorithm is Algorithm.fd:
        strategy = FDStrategy(num_clients=exp_config.num_clients())
        initial_arrays = ArrayRecord()
        evaluate_fn = None  # FD has no server-side global model; see strategies/fd.py docstring.

    elif exp_config.algorithm is Algorithm.dsfl:
        open_dataset = load_open_data(exp_config.data_path, exp_config.backbone)
        test_dataset = load_test_data(exp_config.data_path, exp_config.backbone)
        strategy = DSFLStrategy(
            temperature=exp_config.dsfl_temperature,
            num_clients=exp_config.num_clients(),
            num_open=len(open_dataset),
        )
        initial_arrays = ArrayRecord()
        seed_everything(exp_config.seed)
        server_classifier = build_classifier(exp_config.backbone)

        def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
            if server_round == 0:
                return None
            arrays_np = numpy_from_array_record(arrays)
            train_result = dsfl_protocol.distill_step(
                server_classifier,
                open_dataset,
                arrays_np["sharpened_targets"],
                device,
                epochs=exp_config.local_epochs,
                lr=exp_config.learning_rate,
                batch_size=exp_config.effective_batch_size,
                seed=exp_config.seed,
            )
            eval_metrics = dsfl_protocol.server_evaluate(
                server_classifier, test_dataset, device, batch_size=exp_config.effective_batch_size, seed=exp_config.seed
            )
            classification = compute_classification_metrics(
                eval_metrics["y_true"], eval_metrics["y_pred"], NUM_CLASSES
            )
            metrics_ledger.record(
                algorithm=exp_config.algorithm.value,
                scenario=exp_config.scenario.value,
                round=server_round,
                loss=eval_metrics["loss"],
                train_loss=train_result.final_loss,
                metrics=classification,
            )
            return MetricRecord(
                {"loss": eval_metrics["loss"], "accuracy": eval_metrics["accuracy"], "train_loss": train_result.final_loss}
            )

    else:
        raise ValueError(f"unknown algorithm {exp_config.algorithm}")

    tracked_strategy = CommsTrackingStrategy(
        strategy, exp_config.algorithm.value, exp_config.scenario.value, logger=events_logger
    )
    result = tracked_strategy.start(
        grid,
        initial_arrays,
        num_rounds=exp_config.num_server_rounds,
        train_config=train_config,
        evaluate_config=evaluate_config,
        evaluate_fn=evaluate_fn,
    )
    tracked_strategy.ledger.write_parquet(run_context.run_dir / "communication.parquet")
    metrics_ledger.write(run_context.run_dir)

    final_round = max(result.evaluate_metrics_serverapp) if result.evaluate_metrics_serverapp else None
    run_context.write_summary(
        {
            "run_id": run_context.run_id,
            "algorithm": exp_config.algorithm.value,
            "scenario": exp_config.scenario.value,
            "num_rounds": exp_config.num_server_rounds,
            "final_round": final_round,
            "final_centralized_metrics": (
                dict(result.evaluate_metrics_serverapp[final_round].items()) if final_round is not None else None
            ),
        }
    )
    log_event(events_logger, "run_end", final_round=final_round)
    events_stream.close()
