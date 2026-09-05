"""Custom ``Strategy`` for SSFL: proposal (train exchange) -> majority vote -> broadcast ->
distillation (evaluate exchange), matching ``protocols/ssfl.py`` exactly.

Only pseudo-labels (client->server) and global_labels/valid_mask (server->client)
cross the wire -- never model parameters, matching the privacy boundary tested in M4.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from flwr.common import ArrayRecord, ConfigRecord, Message, MessageType, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords, sample_nodes

from ssfl.config import Algorithm, HardAggregation, VotingMode
from ssfl.models import NUM_CLASSES
from ssfl.protocols.dawid_skene import (
    ABSTAIN,
    DawidSkeneFit,
    DawidSkeneSettings,
    DawidSkeneState,
    build_annotation_matrix,
    fit_dawid_skene,
)
from ssfl.protocols.message import Envelope, ExpectedContext, ProtocolError, validate_envelope
from ssfl.protocols.payload_limits import validate_ssfl_proposal_arrays
from ssfl.protocols.ssfl import ProposalResult, aggregate_soft, aggregate_votes
from ssfl.records import array_record_from_numpy, numpy_from_array_record

# MetricRecord holds numbers only, so the Dawid-Skene outcome is reported as a code. 0 is the only
# value in which Dawid-Skene labels were broadcast; everything else fell back to majority.
DS_STATUS_CODES = {
    "ok": 0,
    "not_attempted": 1,
    "warmup": 2,
    "no_annotations": 3,
    "no_observations": 4,
    "insufficient_clients": 5,
    "non_finite_parameters": 6,
    "non_finite_posterior": 7,
    "non_finite_objective": 8,
    "normalization_invariant_failed": 9,
    "objective_decreased": 10,
    "not_converged": 11,
    "permutation_check_diagonal": 12,
    "permutation_check_agreement": 13,
    "estimator_error": 14,
    "permutation_check_chance": 15,
    "not_converged_but_used": 16,
}


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


class SSFLStrategy(Strategy):
    def __init__(
        self,
        scenario: int,
        dataset_manifest_hash: str,
        num_open: int,
        num_clients: int,
        voting_mode: VotingMode = VotingMode.enabled,
        audit_dir: Path | None = None,
        hard_aggregation: HardAggregation = HardAggregation.majority,
        dawid_skene_settings: DawidSkeneSettings | None = None,
        dawid_skene_warmup_rounds: int = 0,
        save_annotations: bool = False,
        annotation_rounds: tuple[int, ...] = (),
        require_matching_valid_mask: bool = False,
    ) -> None:
        self.scenario = scenario
        self.dataset_manifest_hash = dataset_manifest_hash
        self.num_open = num_open
        self.num_clients = num_clients
        self.voting_mode = voting_mode
        self.audit_dir = audit_dir
        self.hard_aggregation = hard_aggregation
        self.dawid_skene_settings = dawid_skene_settings or DawidSkeneSettings()
        self.dawid_skene_warmup_rounds = dawid_skene_warmup_rounds
        self.save_annotations = save_annotations
        self.annotation_rounds = annotation_rounds
        self.require_matching_valid_mask = require_matching_valid_mask
        self.last_dawid_skene_metrics: dict[str, float] = {}
        self._current_node_ids: list[int] = []
        # Online-EM confusion statistics, carried across rounds. Server-side only: it never
        # goes on the wire and stays None unless dawid_skene_state_decay > 0.
        self._ds_state: DawidSkeneState | None = None

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
        replies = list(replies)
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
            try:
                validate_ssfl_proposal_arrays(
                    arrays, num_open=self.num_open, num_classes=NUM_CLASSES
                )
            except ProtocolError:
                continue
            metrics = msg.content["metrics"]
            proposals.append(
                (
                    envelope,
                    ProposalResult(
                        client_id=sender_id,
                        pseudo_labels=arrays.get("pseudo_labels"),
                        soft_probs=arrays.get("soft_probs"),
                        confidences=None,
                        threshold=float(metrics["threshold"]),
                        classifier_loss=float(metrics["classifier_loss"]),
                        discriminator_loss=float(metrics["discriminator_loss"]),
                    ),
                )
            )

        rejected_count = len(replies) - len(proposals)
        if not proposals:
            return None, None

        if self.voting_mode == VotingMode.enabled:
            result = aggregate_votes(proposals, num_open=self.num_open, num_classes=NUM_CLASSES)
        else:
            result = aggregate_soft(proposals, num_open=self.num_open, num_classes=NUM_CLASSES)

        annotations, ds_fit, ds_seconds, ds_reason = self._dawid_skene(
            server_round, proposals, result
        )
        broadcast_labels, broadcast_mask = result.global_labels, result.valid_mask
        ds_broadcast = False
        if self.hard_aggregation == HardAggregation.dawid_skene and ds_fit is not None:
            if ds_fit.ok:
                broadcast_labels, broadcast_mask = ds_fit.labels, ds_fit.valid_mask
                ds_broadcast = True
        elif self.hard_aggregation == HardAggregation.dawid_skene_only:
            # No majority path exists in this arm. A fit that merely failed a majority-anchored
            # gate is still broadcast; a fit that is numerically broken, or absent because the
            # estimator raised, stops the run. Falling back here would silently turn this arm into
            # the hybrid arm and make the three-way comparison meaningless.
            if ds_fit is None or not ds_fit.numerically_valid:
                raise RuntimeError(
                    f"round {server_round}: ssfl_hard_aggregation=dawid_skene_only and the "
                    f"Dawid-Skene fit produced no usable posterior (reason={ds_reason!r}). This "
                    "arm has no majority fallback by design: fix the estimator and rerun all "
                    "three arms on the same code version."
                )
            broadcast_labels = ds_fit.candidate_labels
            broadcast_mask = ds_fit.candidate_valid_mask
            ds_broadcast = True
            if ds_fit.status == "not_converged":
                # Reported distinctly from the hybrid arm's `not_converged`, which is a fallback.
                ds_reason = "not_converged_but_used"
        if (
            self.require_matching_valid_mask
            and ds_fit is not None
            and ds_fit.candidate_valid_mask is not None
        ):
            # The arms are only comparable while they label the same open-set items. A single
            # differing bit means they are being scored on different sample sets, so stop.
            if not np.array_equal(ds_fit.candidate_valid_mask, result.valid_mask):
                differing = int(
                    np.count_nonzero(ds_fit.candidate_valid_mask != result.valid_mask)
                )
                raise RuntimeError(
                    f"round {server_round}: Dawid-Skene and majority valid masks differ on "
                    f"{differing} of {self.num_open} open-set items, so the arms would be "
                    "compared on different sample sets. Check "
                    "dawid_skene_min_item_annotations and dawid_skene_posterior_threshold."
                )
        if self.audit_dir is not None:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = self.audit_dir / f"ssfl_aggregation_round_{server_round}.npz"
            audit_tmp = audit_path.with_suffix(".tmp.npz")
            payload = {
                "votes_per_class": result.votes_per_class,
                "participating_counts": result.participating_counts,
                "global_labels": broadcast_labels.astype(np.int8),
                "valid_mask": broadcast_mask,
            }
            if ds_fit is not None:
                payload["majority_labels"] = result.global_labels.astype(np.int8)
                payload["majority_valid_mask"] = result.valid_mask
                # The candidate, not the accepted fit: a rejected round still has to be
                # re-scorable offline, and in dawid_skene_only mode the candidate IS what was
                # broadcast. Falls back to `labels` only when EM produced nothing at all.
                candidate_labels = (
                    ds_fit.candidate_labels if ds_fit.numerically_valid else ds_fit.labels
                )
                candidate_mask = (
                    ds_fit.candidate_valid_mask if ds_fit.numerically_valid else ds_fit.valid_mask
                )
                payload["dawid_skene_labels"] = candidate_labels.astype(np.int8)
                payload["dawid_skene_valid_mask"] = candidate_mask
                payload["dawid_skene_status"] = np.array(ds_reason)
                if ds_fit.alignment_permutation:
                    payload["dawid_skene_alignment"] = np.array(
                        ds_fit.alignment_permutation, dtype=np.int64
                    )
            if annotations is not None and self._save_annotations_this_round(server_round):
                # Restricted diagnostic (DATA_CARD.md / the approval brief): raw client-by-sample
                # labels are off by default and this audit is deleted once settings are locked.
                payload["annotations"] = annotations
                if ds_fit is not None and ds_fit.confusion is not None:
                    # Per-client behaviour, restricted for the same reason and behind the same
                    # gate. Row order is not recoverable from the matrix, so it ships with it.
                    payload["dawid_skene_confusion"] = ds_fit.confusion
                    payload["dawid_skene_confusion_clients"] = np.array(ds_fit.eligible_senders)
            np.savez_compressed(audit_tmp, **payload)
            audit_tmp.replace(audit_path)
        valid_votes = result.votes_per_class[result.valid_mask]
        if len(valid_votes):
            sorted_votes = np.sort(valid_votes, axis=1)
            vote_margins = sorted_votes[:, -1] - sorted_votes[:, -2]
        else:
            vote_margins = np.array([], dtype=np.int64)
        label_counts = np.bincount(broadcast_labels[broadcast_mask], minlength=NUM_CLASSES)
        # Stashed so server_app can put the ds_* diagnostics in metrics.parquet next to accuracy:
        # aggregate_train's return value never reaches the evaluate callback.
        ds_metrics = self._dawid_skene_metrics(
            ds_fit, ds_seconds, ds_reason, result, annotations, ds_broadcast
        )
        self.last_dawid_skene_metrics = ds_metrics
        arrays_out = array_record_from_numpy(
            {"global_labels": broadcast_labels.astype("int8"), "valid_mask": broadcast_mask}
        )
        metrics_out = MetricRecord(
            {
                "valid_rate": float(broadcast_mask.mean()),
                "tie_count": result.tie_count,
                "all_abstain_count": result.all_abstain_count,
                "num_proposals": len(proposals),
                "rejected_count": rejected_count,
                "participating_min": int(result.participating_counts.min()),
                "participating_mean": float(result.participating_counts.mean()),
                "participating_max": int(result.participating_counts.max()),
                "vote_margin_mean": float(vote_margins.mean()) if len(vote_margins) else 0.0,
                "vote_margin_min": int(vote_margins.min()) if len(vote_margins) else 0,
                **{
                    f"global_class_{index}_count": int(count)
                    for index, count in enumerate(label_counts)
                },
                **ds_metrics,
            }
        )
        return arrays_out, metrics_out

    def _save_annotations_this_round(self, server_round: int) -> bool:
        if not self.save_annotations:
            return False
        return not self.annotation_rounds or server_round in self.annotation_rounds

    def _dawid_skene(
        self, server_round: int, proposals, majority
    ) -> tuple[np.ndarray | None, DawidSkeneFit | None, float, str]:
        """Build the client-by-sample matrix and fit, or explain why not.

        Runs in both shadow and active mode; only the caller's use of the result differs. Returns
        ``(annotations, fit, seconds, reason)``; ``fit`` is None when no fit was attempted and
        ``reason`` then names why.
        """
        if self.voting_mode != VotingMode.enabled:
            return None, None, 0.0, "not_attempted"
        if self.hard_aggregation == HardAggregation.majority:
            # No fit, but the matrix itself is still worth keeping when annotations are being
            # dumped: it is what makes a counterfactual Dawid-Skene analysis of the majority arm
            # possible after the fact (DENEY_1_SCENARIO_3_DENEY_PLANI.md section 10).
            if not self._save_annotations_this_round(server_round):
                return None, None, 0.0, "not_attempted"
            annotations, _ = build_annotation_matrix(
                [(envelope.sender_id, result.pseudo_labels) for envelope, result in proposals],
                num_open=self.num_open,
                num_classes=NUM_CLASSES,
            )
            return annotations, None, 0.0, "not_attempted"
        annotations, senders = build_annotation_matrix(
            [(envelope.sender_id, result.pseudo_labels) for envelope, result in proposals],
            num_open=self.num_open,
            num_classes=NUM_CLASSES,
        )
        if server_round <= self.dawid_skene_warmup_rounds:
            # Early-round majority labels are close to noise (0.186 accurate at round 1 on the
            # recorded scenario-1 run), so a fit there would estimate confusions from noise and
            # steer distillation when it matters most. Warm-up is a reported setting, not a guess.
            return annotations, None, 0.0, "warmup"
        started = time.perf_counter()
        try:
            fit = fit_dawid_skene(
                annotations,
                num_classes=NUM_CLASSES,
                majority_labels=majority.global_labels,
                settings=self.dawid_skene_settings,
                senders=senders,
                state=self._ds_state,
            )
        except (ValueError, FloatingPointError, MemoryError):
            # Any estimator defect falls back to majority rather than stopping the run; the code
            # is visible in metrics so a run that silently degrades to majority is still auditable.
            return annotations, None, time.perf_counter() - started, "estimator_error"
        if fit.state is not None:
            self._ds_state = fit.state
        return annotations, fit, time.perf_counter() - started, fit.status

    def _dawid_skene_metrics(
        self,
        fit: DawidSkeneFit | None,
        seconds: float,
        reason: str,
        majority,
        annotations: np.ndarray | None = None,
        ds_broadcast: bool = False,
    ) -> dict[str, float | int]:
        """Round diagnostics for metrics.parquet. Every key here is defined in
        DAWID_SKENE_GLOSSARY.md, and tests/unit/test_dawid_skene_glossary.py fails if a key is
        added, renamed or dropped without that file following. Note ``ds_applied`` answers what
        clients received, not whether the fit succeeded -- a clean fit in shadow mode reports 0,
        and a fit that failed a majority-anchored gate reports 1 in ``dawid_skene_only`` mode.
        """
        if self.hard_aggregation == HardAggregation.majority:
            return {}
        coverage = _annotation_coverage(annotations)
        if fit is None:
            return {
                "ds_status": DS_STATUS_CODES.get(reason, DS_STATUS_CODES["estimator_error"]),
                "ds_applied": 0,
                "ds_seconds": float(seconds),
                **coverage,
            }
        # The candidate, not the accepted fit: in dawid_skene_only a non-converged fit is still
        # what clients get, and `fit.valid_mask`/`fit.labels` stay empty on every non-ok path, so
        # reading those would report 0.0 coverage and 0.0 disagreement for the one arm whose
        # labels these describe.
        candidate_mask = (
            fit.candidate_valid_mask if fit.candidate_valid_mask is not None else fit.valid_mask
        )
        candidate_labels = fit.candidate_labels if fit.candidate_labels is not None else fit.labels
        comparable = candidate_mask & majority.valid_mask
        disagreement = (
            float((candidate_labels[comparable] != majority.global_labels[comparable]).mean())
            if comparable.any()
            else 0.0
        )
        identity = tuple(range(len(fit.alignment_permutation)))
        return {
            # `reason`, not `fit.status`: dawid_skene_only rewrites a used non-converged fit to
            # `not_converged_but_used`, and the code has to say which of the two happened.
            "ds_status": DS_STATUS_CODES.get(reason, DS_STATUS_CODES["estimator_error"]),
            "ds_applied": int(ds_broadcast),
            "ds_alignment_score": float(fit.alignment_score),
            "ds_alignment_identity": int(
                bool(fit.alignment_permutation) and fit.alignment_permutation == identity
            ),
            "ds_valid_mask_match": int(np.array_equal(candidate_mask, majority.valid_mask)),
            "ds_seconds": float(seconds),
            "ds_iterations": fit.iterations,
            "ds_converged": int(fit.converged),
            # Non-finite is the normal reading on an early-failure path; 0.0 keeps a NaN out of
            # metrics.parquet, and ds_status already says whether these two are meaningful.
            "ds_log_likelihood": _finite(fit.log_likelihood),
            "ds_objective": _finite(fit.objective),
            "ds_eligible_clients": fit.eligible_clients,
            "ds_excluded_clients": len(fit.excluded_clients),
            "ds_diagonal_fraction": fit.diagonal_fraction,
            # The permutation check is a ratio of these two, so the yardstick has to be reported
            # alongside the measurement or the accept/reject decision is not auditable.
            "ds_reference_diagonal_fraction": fit.reference_diagonal_fraction,
            "ds_majority_agreement": fit.majority_agreement,
            "ds_max_posterior_mean": fit.max_posterior_mean,
            "ds_disagreement_rate": disagreement,
            "ds_valid_rate": float(candidate_mask.mean()),
            **coverage,
        }

    def configure_evaluate(
        self, server_round: int, arrays, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        config["server-round"] = server_round
        record = RecordDict({"arrays": arrays, "config": config})
        return [
            Message(record, message_type=MessageType.EVALUATE, dst_node_id=n)
            for n in self._current_node_ids
        ]

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        valid_senders = frozenset(str(n) for n in self._current_node_ids)
        contents = [
            msg.content
            for msg in replies
            if not msg.has_error() and str(msg.metadata.src_node_id) in valid_senders
        ]
        if not contents:
            return None
        total_examples = sum(
            float(next(iter(content.metric_records.values()))["num-examples"])
            for content in contents
        )
        if total_examples == 0:
            return MetricRecord(
                {
                    "loss": 0.0,
                    "distillation_skipped": 1,
                    "distillation_examples": 0,
                    "responding_clients": len(contents),
                }
            )
        return aggregate_metricrecords(contents, "num-examples")


def _annotation_coverage(annotations: np.ndarray | None) -> dict[str, float]:
    """How much of the open set each client actually labelled, as a fraction.

    The spread is what matters: ``min`` is the client closest to the exclusion threshold, and
    a fit whose eligible clients each saw a different slice of the open set is a different
    situation from one where they all saw the same slice, even at the same client count.
    """
    if annotations is None or annotations.ndim != 2 or annotations.shape[1] == 0:
        return {}
    per_client = (annotations != ABSTAIN).mean(axis=1)
    return {
        "ds_coverage_min": float(per_client.min()),
        "ds_coverage_mean": float(per_client.mean()),
        "ds_coverage_max": float(per_client.max()),
    }
