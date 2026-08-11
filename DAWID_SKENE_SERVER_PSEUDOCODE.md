# Server-side Dawid-Skene aggregation pseudocode

This is intentionally non-executable design pseudocode. It describes the recommended server-only
Dawid-Skene label aggregation and its experiment hooks inside the existing workflow. It does not
define a combined method name, and it is not a claim of paper-faithful FedDS.

## 1. Types and configuration

```text
CONSTANT ABSTAIN = -1

ENUM HardAggregationMode:
    MAJORITY
    DAWID_SKENE_SHADOW   # compute DS; broadcast majority
    DAWID_SKENE          # compute and broadcast DS

STRUCT DawidSkeneOptions:
    max_iterations: integer                 # candidate 100; >= 1
    min_iterations: integer                 # candidate 2; <= max_iterations
    tolerance: float                        # candidate 1e-6; relative objective change
    initialization: enum                    # SMOOTHED_VOTES, ONE_HOT_MAJORITY, NEAR_DIAGONAL
    initialization_pseudocount: float       # candidate 0.01; > 0
    initial_diagonal_probability: float     # candidate 0.9; near-diagonal arm only
    confusion_pseudocount: float            # candidate 0.1; > 0
    class_prior_pseudocount: float           # candidate 1.0; > 0
    min_item_annotations: integer           # candidate 1
    min_client_annotations: integer         # candidate 1
    min_clients: integer                    # candidate 3; identifiability/support gate
    posterior_threshold: float              # candidate 0.0; in [0, 1]
    warm_start: boolean                      # false in primary experiment
    damping: float                           # candidate 1.0; in (0, 1]
    epsilon: float                           # candidate 1e-12
    internal_dtype: FLOAT64
    failure_policy: MAJORITY_ON_FAILURE_OR_NONCONVERGENCE
    save_annotations: boolean               # false except restricted shadow audit
    full_audit_rounds: sorted tuple[integer]  # unique/in-range for stable config hashing

STRUCT DawidSkeneResult:
    labels: int8[N]                          # ABSTAIN for invalid items
    valid_mask: bool[N]
    posterior: float[N, C]
    class_prior: float[C]
    confusion: float[J, C, C]               # [client, true class, observed class]
    client_ids: string[J]
    client_reliability: float[J]             # trace(confusion[j]) / C, diagnostic only
    item_annotation_counts: int[N]
    client_annotation_counts: int[J]
    iterations: integer
    converged: boolean
    data_log_likelihood_history: float[]
    objective_history: float[]
    stop_reason: string
    fallback_reason: optional string

UNION DawidSkeneFitOutcome:
    Success(result: DawidSkeneResult)
    Failure(reason: string,
            data_log_likelihood_history: float[],
            objective_history: float[])

STRUCT AggregationResult:                    # exact existing downstream contract
    global_labels: int8[N]
    valid_mask: bool[N]
    votes_per_class: int[N, C]
    participating_counts: int[N]
    tie_count: integer
    all_abstain_count: integer
    rejected: list[(sender_id, reason)]

FUNCTION validate_ds_options(options, configured_rounds, configured_num_classes):
    REQUIRE options.max_iterations >= 1
    REQUIRE 1 <= options.min_iterations <= options.max_iterations
    REQUIRE options.tolerance > 0
    REQUIRE options.initialization IN {SMOOTHED_VOTES, ONE_HOT_MAJORITY, NEAR_DIAGONAL}
    REQUIRE options.initialization_pseudocount > 0
    IF options.initialization == NEAR_DIAGONAL:
        REQUIRE (1 / configured_num_classes) < options.initial_diagonal_probability < 1
    REQUIRE options.confusion_pseudocount > 0
    REQUIRE options.class_prior_pseudocount > 0
    REQUIRE options.min_item_annotations >= 1
    REQUIRE options.min_client_annotations >= 1
    REQUIRE options.min_clients >= 3
    REQUIRE 0 <= options.posterior_threshold <= 1
    REQUIRE 0 < options.damping <= 1
    REQUIRE options.epsilon > 0
    REQUIRE options.internal_dtype == FLOAT64
    REQUIRE options.warm_start == false
        # Reject true until stable logical-client identity and resume-state keying exist.
    REQUIRE options.full_audit_rounds is sorted, unique, and within 1..configured_rounds
    RETURN valid

FUNCTION validate_ssfl_ds_config(config):
    IF config.ssfl_hard_aggregation IN {DAWID_SKENE_SHADOW, DAWID_SKENE}:
        REQUIRE config.run_kind == "extension"
        REQUIRE config.ssfl_label_representation == "hard"
        REQUIRE config.ssfl_voting_mode == "enabled"
        validate_ds_options(config.ds_options, config.rounds, config.num_classes)
        IF config.ds_options.save_annotations:
            REQUIRE config.ssfl_hard_aggregation == DAWID_SKENE_SHADOW
            REQUIRE restricted_research_audit_storage_is_configured
```

## 2. Server adapter: validated proposals to annotation matrix

```text
FUNCTION prepare_annotations(proposals, num_open, num_classes):
    accepted = empty list
    rejected = empty list

    # Group first: sorting only by sender and keeping the first conflicting retry would still make
    # the answer depend on arrival order.
    groups = group proposals by envelope.sender_id
    FOR sender_id IN sort(keys(groups)):
        sender_group = groups[sender_id]

        IF sender_group contains different label vectors:
            rejected.append((sender_id, "conflicting duplicate sender payloads"))
            CONTINUE

        # Identical retries are idempotent. Keep one canonical value and record the extras.
        envelope, proposal = deterministic_representative(sender_group)
        FOR each extra retry after the representative:
            rejected.append((sender_id, "identical duplicate sender retry"))

        labels = proposal.pseudo_labels
        REQUIRE labels exists
        REQUIRE shape(labels) == [num_open]
        REQUIRE integer_dtype(labels)
        REQUIRE every value is ABSTAIN or in [0, num_classes)

        accepted.append((sender_id, cast_int8(labels)))

    IF accepted is empty:
        RETURN PreparationFailure(reason="no accepted proposals", rejected=rejected)

    client_ids = tuple(sender_id for accepted rows)
    Y = stack(label vector for accepted rows)       # shape [J, N], int8
    observed = (Y != ABSTAIN)                       # shape [J, N]

    votes = zeros([num_open, num_classes], INT64)
    FOR observed pair (j, i):
        votes[i, Y[j, i]] += 1

    item_counts = sum(observed, axis=client)
    client_counts = sum(observed, axis=item)

    RETURN Y, observed, client_ids, votes, item_counts, client_counts, rejected
```

## 3. Deterministic majority reference and fallback

```text
FUNCTION majority_from_votes(votes):
    N, C = shape(votes)
    labels = full([N], ABSTAIN, INT8)
    valid = zeros([N], BOOL)
    tie_count = 0

    FOR item i IN 0..N-1:
        IF sum(votes[i]) == 0:
            CONTINUE

        winners = indices(votes[i] == max(votes[i]))
        labels[i] = min(winners)                    # preserve current deterministic tie rule
        valid[i] = true
        tie_count += (length(winners) > 1)

    RETURN labels, valid, tie_count
```

## 4. Vote-anchored initialization

```text
FUNCTION initialize_posterior(votes, options):
    N, C = shape(votes)
    a0 = options.initialization_pseudocount

    IF options.initialization == SMOOTHED_VOTES
       OR options.initialization == NEAR_DIAGONAL:
        Q = cast_float64(votes) + a0
        Q = Q / row_sum(Q)

    ELSE IF options.initialization == ONE_HOT_MAJORITY:
        Q = full([N, C], a0, FLOAT64)
        FOR item i IN 0..N-1:
            winner = lowest_index_argmax(votes[i])
            Q[i, winner] += 1
        Q = Q / row_sum(Q)

    ELSE:
        RAISE configuration_error("unknown initialization")

    # Items with no observations receive a neutral distribution but are never broadcast as valid.
    FOR item i WHERE sum(votes[i]) == 0:
        Q[i, :] = 1 / C

    REQUIRE finite(Q)
    REQUIRE every row_sum(Q) approximately 1
    RETURN Q

FUNCTION initialize_confusions(Q, Y, observed, options):
    J, N = shape(Y)
    C = number_of_columns(Q)

    IF options.initialization == NEAR_DIAGONAL:
        p = options.initial_diagonal_probability
        REQUIRE (1 / C) < p < 1
        confusion = full([J, C, C], (1 - p) / (C - 1), FLOAT64)
        FOR client j IN 0..J-1:
            FOR class c IN 0..C-1:
                confusion[j, c, c] = p
        RETURN confusion

    RETURN update_confusions(Q, Y, observed, old_confusion=NULL, options)

FUNCTION update_class_prior(Q, options):
    beta = options.class_prior_pseudocount
    prior = column_sum(Q) + beta
    prior = prior / sum(prior)
    REQUIRE finite_normalized(prior)
    RETURN prior

FUNCTION update_confusions(Q, Y, observed, old_confusion, options):
    J, N = shape(Y)
    C = number_of_columns(Q)
    alpha = options.confusion_pseudocount
    new_confusion = zeros([J, C, C], FLOAT64)

    FOR client j IN 0..J-1:
        client_items = indices(observed[j])

        FOR candidate_true_class c IN 0..C-1:
            counts = full([C], alpha, FLOAT64)

            FOR item i IN client_items:
                observed_label = Y[j, i]
                counts[observed_label] += Q[i, c]

            row = counts / sum(counts)

            IF old_confusion exists:
                d = options.damping
                row = d * row + (1 - d) * old_confusion[j, c, :]
                row = row / sum(row)

            new_confusion[j, c, :] = row

    REQUIRE every row is finite, nonnegative, and approximately normalized
    RETURN new_confusion
```

## 5. Log-space E-step and objective

```text
FUNCTION expectation_step(Y, observed, class_prior, confusion, options):
    J, N = shape(Y)
    C = length(class_prior)
    log_scores = repeat_rows(log(clip(class_prior, options.epsilon, 1)), N)

    # Use chunking/indexed gathers in production; do not allocate [J, N, C].
    FOR client j IN 0..J-1:
        labelled_items = indices(observed[j])
        labels = Y[j, labelled_items]

        FOR candidate_true_class c IN 0..C-1:
            probabilities = confusion[j, c, labels]
            log_scores[labelled_items, c] += log(clip(probabilities, options.epsilon, 1))

    log_normalizer = logsumexp(log_scores, axis=class)
    Q = exp(log_scores - log_normalizer[:, None])
    data_log_likelihood = sum(log_normalizer)

    REQUIRE finite(Q, data_log_likelihood)
    REQUIRE every row_sum(Q) approximately 1
    RETURN Q, data_log_likelihood

FUNCTION regularized_objective(data_log_likelihood, class_prior, confusion, options):
    # Adding beta/alpha to expected counts is a MAP M-step under Dirichlet priors whose
    # log-density coefficients are beta/alpha. The monitored monotonic quantity must include them.
    prior_term = options.class_prior_pseudocount
                 * sum(log(clip(class_prior, options.epsilon, 1)))
    confusion_term = options.confusion_pseudocount
                       * sum(log(clip(confusion, options.epsilon, 1)))
    value = data_log_likelihood + prior_term + confusion_term
    REQUIRE finite(value)
    RETURN value
```

## 6. Dawid-Skene fit with explicit failure policy

```text
FUNCTION fit_dawid_skene(Y, client_ids, num_classes, options):
    REQUIRE options were validated against the configured round count before the run starts
    # The implementation boundary converts any runtime invariant/numerical exception below into
    # Failure(reason, data_log_likelihood_history, objective_history); it never leaks partial state.
    observed = (Y != ABSTAIN)
    client_counts = sum(observed, axis=item)

    eligible_clients = (client_counts >= options.min_client_annotations)

    IF count(eligible_clients) < options.min_clients:
        RETURN Failure("too few eligible clients", [], [])

    Y_clients = rows(Y, eligible_clients)
    observed_clients = rows(observed, eligible_clients)
    client_ids_fit = values(client_ids, eligible_clients)
    fit_item_counts = sum(observed_clients, axis=client)
    estimable_items = (fit_item_counts >= options.min_item_annotations)

    IF count(estimable_items) == 0:
        RETURN Failure("no estimable items", [], [])

    # All-abstain columns are excluded from EM so neutral rows cannot distort the learned prior.
    Y_fit = columns(Y_clients, estimable_items)
    observed_fit = columns(observed_clients, estimable_items)

    votes = count_observed_labels_per_item(Y_fit, num_classes)
    Q = initialize_posterior(votes, options)
    class_prior = update_class_prior(Q, options)
    confusion = initialize_confusions(Q, Y_fit, observed_fit, options)

    objective_history = empty list
    data_log_likelihood_history = empty list
    converged = false
    stop_reason = "max_iterations"

    FOR iteration IN 1..options.max_iterations:
        Q_current, data_log_likelihood = expectation_step(
            Y_fit, observed_fit, class_prior, confusion, options
        )
        objective = regularized_objective(
            data_log_likelihood, class_prior, confusion, options
        )

        IF data_log_likelihood is non-finite OR objective is non-finite
           OR Q_current contains non-finite:
            RETURN Failure(
                "non-finite E-step",
                data_log_likelihood_history,
                objective_history,
            )

        IF objective_history is not empty:
            previous = last(objective_history)
            allowed_numeric_drop = 1e-10 * (1 + abs(previous))
            IF objective < previous - allowed_numeric_drop:
                RETURN Failure(
                    "objective decreased",
                    data_log_likelihood_history + [data_log_likelihood],
                    objective_history + [objective],
                )

        data_log_likelihood_history.append(data_log_likelihood)
        objective_history.append(objective)

        IF iteration >= options.min_iterations AND length(objective_history) >= 2:
            relative_change = abs(objective_history[-1] - objective_history[-2])
                              / (1 + abs(objective_history[-2]))
            IF relative_change <= options.tolerance:
                # Q_current and the returned parameters now describe the same EM half-step.
                Q = Q_current
                converged = true
                stop_reason = "relative_objective_tolerance"
                BREAK

        Q = Q_current

        # Do not apply an unobserved final M-step at the iteration cap. This keeps the returned
        # posterior synchronized with the returned parameters.
        IF iteration == options.max_iterations:
            BREAK

        class_prior = update_class_prior(Q, options)
        confusion = update_confusions(
            Q, Y_fit, observed_fit, old_confusion=confusion, options
        )

    IF NOT converged
       AND options.failure_policy == MAJORITY_ON_FAILURE_OR_NONCONVERGENCE:
        RETURN Failure(
            "iteration cap without convergence",
            data_log_likelihood_history,
            objective_history,
        )

    posterior_full = repeat_rows(class_prior, number_of_columns(Y))
    posterior_full[estimable_items, :] = Q
    labels = cast_int8(lowest_index_argmax(posterior_full, axis=class))
    confidence = max(posterior_full, axis=class)
    valid = (fit_item_counts >= options.min_item_annotations)
            AND (confidence >= options.posterior_threshold)
    labels[NOT valid] = ABSTAIN

    reliability = trace(confusion[j]) / num_classes FOR each fitted client j

    result = DawidSkeneResult(
        labels=labels,
        valid_mask=valid,
        posterior=posterior_full,
        class_prior=class_prior,
        confusion=confusion,
        client_ids=client_ids_fit,
        client_reliability=reliability,
        item_annotation_counts=fit_item_counts,
        client_annotation_counts=client_counts[eligible_clients],
        iterations=length(objective_history),
        converged=converged,
        data_log_likelihood_history=data_log_likelihood_history,
        objective_history=objective_history,
        stop_reason=stop_reason,
        fallback_reason=NULL,
    )
    RETURN Success(result)
```

## 7. SSFL aggregation mode selection

```text
FUNCTION aggregate_ssfl_hard_labels(proposals, num_open, num_classes, mode, ds_options):
    IF mode == MAJORITY:
        # Preserve the canonical path byte-for-byte; do not make majority depend on DS adapters.
        result = existing_aggregate_votes(proposals, num_open, num_classes)
        diagnostics = summarize_majority_without_ground_truth(result)
        RETURN result, diagnostics

    prepared = prepare_annotations(proposals, num_open, num_classes)
    IF prepared failed:
        result = AggregationResult(
            global_labels=full([num_open], ABSTAIN, INT8),
            valid_mask=zeros([num_open], BOOL),
            votes_per_class=zeros([num_open, num_classes], INT64),
            participating_counts=zeros([num_open], INT64),
            tie_count=0,
            all_abstain_count=num_open,
            rejected=prepared.rejected,
        )
        diagnostics = aggregation_failure_metrics(prepared.reason)
        RETURN result, diagnostics

    Y, observed, client_ids, votes, item_counts, client_counts, rejected = prepared
    majority_labels, majority_valid, tie_count = majority_from_votes(votes)

    ds_outcome = fit_dawid_skene(Y, client_ids, num_classes, ds_options)

    IF ds_outcome is Failure:
        chosen_labels = majority_labels
        chosen_valid = majority_valid
        fallback_reason = ds_outcome.reason
        ds = NULL
    ELSE IF mode == DAWID_SKENE_SHADOW:
        ds = ds_outcome.result
        chosen_labels = majority_labels            # causal safety: normal training continues
        chosen_valid = majority_valid
        fallback_reason = NULL
    ELSE:
        ds = ds_outcome.result
        chosen_labels = ds.labels
        chosen_valid = ds.valid_mask
        fallback_reason = NULL

    diagnostics = summarize_without_ground_truth(
        ds=ds,
        fit_outcome=ds_outcome,                    # includes failed fit histories and reason
        majority_labels=majority_labels,
        majority_valid=majority_valid,
        chosen_labels=chosen_labels,
        chosen_valid=chosen_valid,
        votes=votes,
        runtime=measured_runtime,
        fallback_reason=fallback_reason,
    )

    RETURN AggregationResult(
        global_labels=chosen_labels,
        valid_mask=chosen_valid,
        votes_per_class=votes,                   # retain current audit and margin metrics
        participating_counts=item_counts,
        tie_count=tie_count,
        all_abstain_count=count(item_counts == 0),
        rejected=rejected,
    ), diagnostics
```

## 8. Strategy integration

```text
METHOD SSFLStrategy.aggregate_train(server_round, replies):
    proposals = existing_validate_authorize_and_decode(replies)
        # Do not apply an order-sensitive first-wins deduplication here. The adapter groups all
        # replies per sender, accepts one identical retry, and rejects conflicting duplicates.

    IF no proposals:
        # Returning None would leave Flower's prior arrays in place and can rebroadcast stale
        # labels. Emit an explicit empty consensus so every downstream distillation step skips.
        result = AggregationResult(
            global_labels=full([num_open], ABSTAIN, INT8),
            valid_mask=zeros([num_open], BOOL),
            votes_per_class=zeros([num_open, NUM_CLASSES], INT64),
            participating_counts=zeros([num_open], INT64),
            tie_count=0,
            all_abstain_count=num_open,
            rejected=[],
        )
        record aggregation status "no_proposals"
        RETURN broadcast(result.global_labels, result.valid_mask), aggregation_metrics(result)

    IF configured voting_mode is DISABLED:
        # Existing config validation guarantees this is the soft-label representation.
        REQUIRE hard aggregation mode is MAJORITY
        result = existing aggregate_soft(proposals)
        aggregation_metrics = summarize_soft_without_ground_truth(result)
    ELSE:
        result, aggregation_metrics = aggregate_ssfl_hard_labels(
            proposals,
            num_open,
            NUM_CLASSES,
            hard_aggregation_mode,
            ds_options,
        )

    write existing vote/participation/label/mask attempt-local audit atomically

    IF DS mode:
        append round-level convergence/timing/disagreement metrics through a resume-aware
        AggregationDiagnosticsLedger injected by server_app.main
        write full posterior/prior/confusion only on configured audit rounds
        write annotation matrix only when explicit restricted shadow flag is true
        never put per-client confusion/reliability in general telemetry

    broadcast ONLY:
        global_labels as int8[N]
        valid_mask as bool[N]

    # on_round_end atomically promotes the successful attempt/round diagnostics to the run-level
    # canonical ledger. Resume discards uncommitted attempt files and rows beyond the checkpoint.
    RETURN broadcast_arrays, aggregation_metrics
```

## 9. Offline quality evaluator boundary

```text
OFFLINE FUNCTION evaluate_aggregation_quality(run_artifacts, sealed_open_audit):
    REQUIRE federated run is finished or artifacts are read-only
    REQUIRE manifest hash in run equals manifest hash in audit
    open_truth = filter sealed_open_audit WHERE split == "open"
    REQUIRE every open global index is aligned exactly once after this filter

    true_labels = load labels from open_truth              # forbidden in live server module
    majority = load shadow majority outputs
    ds = load shadow DS outputs

    FOR each round and each method:
        compute accuracy, macro/micro/weighted metrics
        compute per-class confusion and valid coverage
        compute matched-coverage comparison
        compute tie and non-tie flip outcomes
        compute posterior NLL, Brier, entropy, calibration
        compare estimated client confusion/reliability with audit-only empirical values

    write aggregation_quality.parquet
    never feed any result back into the already-running training process
```

## 10. Experiment orchestration

```text
PHASE 0:
    run unit, property, synthetic, determinism, and scenario-sized benchmark gates

PHASE 1:
    for scenario in [1, 2, 3]:
        run 50 rounds with majority active and DS shadow, seed 2023
    tune only on preregistered development slice
    lock every DS setting

PHASE 2:
    for scenario in [1, 2, 3]:
        for seed in [2023, 2024, 2025]:
            run paired 50-round majority and locked-DS jobs
    promote only if pilot gates pass

PHASE 3:
    for scenario in [1, 2, 3]:
        for seed in [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]:
            run paired 200-round majority and locked-DS jobs
    report scenario-specific effects and ten seed-block, equal-scenario paired effects

AT NO POINT:
    choose DS hyperparameters from active-run test accuracy
    read sealed open labels inside aggregation or training
    call this adapted protocol a paper-faithful FedDS reproduction
```
