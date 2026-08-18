# Dawid-Skene glossary

One name per concept, for the report, the metrics and the spoken explanation. Stage 1 of
`DAWID_SKENE_SONRAKI_ADIMLAR.md`. Where a term is also a metric key, the key here is the exact
string written to `metrics.parquet`.

`tests/unit/test_dawid_skene_glossary.py` fails if the code emits a key this file does not
define, or if this file defines one the code no longer emits.

## The four exclusions, which are not the same exclusion

A client can drop out of a round at four different points. Conflating them is the single most
likely way to misread a run.

| Term | Metric key | What it counts | What it is not |
| --- | --- | --- | --- |
| Rejected message | `rejected_count` | Reply messages discarded before aggregation: envelope validation failure, algorithm/scenario/round/phase mismatch, unknown sender, duplicate message id, or payload shape/dtype violation. Emitted by SSFL, FD and DS-FL. | Not a Dawid-Skene concept. A rejected message never reaches the annotation matrix, so the estimator cannot see it and never reports it. |
| Excluded client | `ds_excluded_clients` | Clients present in the annotation matrix but below `dawid_skene_min_client_annotations`. That threshold is 1 in the active configuration, so in practice this counts clients that annotated nothing at all. Dropped before EM starts. | Not down-weighting. Exclusion is binary and happens before the fit; a client is either in the matrix EM sees or it is not. |
| Eligible client | `ds_eligible_clients` | Clients that survived the check above and have a confusion matrix estimated for them. `ds_eligible_clients + ds_excluded_clients` is the number of clients whose proposal was accepted this round. | Not the number of clients sampled for the round, and not the number that sent a message. |
| Abstained item | `all_abstain_count` | Open-set items no participating client labelled. `ABSTAIN` is missing data in the model, never a class. | Not an item the aggregator got wrong. It carries no label, so it is absent from the likelihood and from every accuracy figure. |

## Fit, application and fallback

| Term | Metric key | Definition |
| --- | --- | --- |
| Fit status | `ds_status` | Integer code for the **first** failure the round hit, or 0 for `ok`. The codes are mutually exclusive by construction, so per-status round counts sum to the number of attempted rounds. Codes are listed below. |
| Applied | `ds_applied` | 1 only when the fit passed every gate **and** the run is in active mode (`hard_aggregation=dawid_skene`). In shadow mode a flawless fit still reports 0, because nothing was broadcast. `ds_applied` and `ds_status == ok` are different questions: the first is about what clients received, the second about what the estimator concluded. |
| Fallback | — | Round-level: this round's Dawid-Skene result is discarded and deterministic majority vote is broadcast instead. There is no per-client fallback and no partial fallback — a round either broadcasts the Dawid-Skene labels or the majority labels, never a mixture. A round is a fallback round when Dawid-Skene was attempted in active mode and `ds_applied` is 0. |
| Warm-up | `ds_status` = 2 | Rounds before `dawid_skene_warmup_rounds`, where no fit is attempted at all. These are not failures, and lumping them into a fallback rate overstates it. |
| Client weighting | — | **Proposed, not implemented.** Every eligible client's labels influence the aggregate class-conditionally through its learned confusion matrix, instead of the current binary include/exclude plus one equal vote each. Options A/B/C in `DAWID_SKENE_SONRAKI_ADIMLAR.md` are three ways to do this. Nothing in the code today weights clients: the active path is either unweighted majority or the unweighted Dawid-Skene argmax. Note this concerns pseudo-label aggregation only — no protocol here weights model updates, which is clarification question 1. |

### Fit status codes

| Code | `ds_status` | Meaning |
| --- | --- | --- |
| 0 | `ok` | Converged and passed every gate. |
| 1 | `not_attempted` | Majority-only configuration, or voting disabled. |
| 2 | `warmup` | Inside `dawid_skene_warmup_rounds`. |
| 3 | `no_annotations` | Empty or malformed annotation matrix. |
| 4 | `no_observations` | Matrix present, but no item has a single label. |
| 5 | `insufficient_clients` | Fewer than `dawid_skene_min_clients` eligible clients. |
| 6 | `non_finite_parameters` | Confusion matrices or prior went non-finite during EM. |
| 7 | `non_finite_posterior` | Posterior went non-finite during EM. |
| 8 | `non_finite_objective` | Regularised objective went non-finite. |
| 9 | `normalization_invariant_failed` | A posterior row stopped summing to 1. |
| 10 | `objective_decreased` | The MAP objective fell between iterations; EM must not do this. |
| 11 | `not_converged` | Hit `dawid_skene_max_iterations` without meeting the tolerance. |
| 12 | `permutation_check_diagonal` | Diagonal fraction below the required ratio of majority's own. |
| 13 | `permutation_check_agreement` | Agreement with majority below the floor. |
| 14 | `estimator_error` | The estimator raised; the round falls back rather than stopping the run. |
| 15 | `permutation_check_chance` | No coherent class-to-label mapping at all. |

## Quantities that sound alike

| Term | Metric key | Definition |
| --- | --- | --- |
| Broadcast valid rate | `valid_rate` | Fraction of open-set items actually broadcast this round, by whichever aggregator won the round. |
| Fit valid rate | `ds_valid_rate` | Fraction the Dawid-Skene fit itself considered usable. Reported even when the fit was not applied, which is the point: it is comparable across shadow and active runs. |
| Diagonal fraction | `ds_diagonal_fraction`, `ds_reference_diagonal_fraction` | Coherence of the fitted latent classes against the emitted label space. Meaningful **only** as a ratio against `ds_reference_diagonal_fraction`, majority's score on the same annotations. It cannot detect a global relabelling and has no useful absolute floor — on scenario 1 even sealed ground truth scores about 0.14 to 0.25. |
| Majority agreement | `ds_majority_agreement` | Fraction of items valid in both fits where Dawid-Skene and majority chose the same class. This is the actual permutation detector: it is the only gate statistic anchored outside the fit. |
| Disagreement rate | `ds_disagreement_rate` | `1 - ds_majority_agreement`, over the same mask. It is computed in the strategy rather than the estimator and kept for continuity of existing runs; do not read the two as independent evidence. |
| Annotation coverage | `ds_coverage_min`, `ds_coverage_mean`, `ds_coverage_max` | Fraction of the open set each client actually labelled, summarised across clients. Counted over every client in the annotation matrix, including ones later excluded, and reported during warm-up rounds too. Not the same axis as `participating_*`, which counts clients per item rather than items per client. |
| Vote margin | `vote_margin_mean`, `vote_margin_min` | Top-class votes minus runner-up votes, over items majority considered valid. A margin of 0 is a tie. |
| Tie count | `tie_count` | Items where the top vote count was shared. The current rule resolves these by lowest class index, which stage 3 measured at chance with a 0.17 to 0.79 spread across relabellings — see `TIE_BREAK_PLAN.md`. |
| Participation | `participating_min`, `participating_mean`, `participating_max` | Number of clients that labelled an item, across open-set items. |
| Proposal count | `num_proposals` | Client proposals accepted this round, before any Dawid-Skene eligibility check. |
| Fit cost | `ds_seconds`, `ds_iterations`, `ds_converged` | Wall-clock for the fit, EM steps taken, and whether the tolerance was met before the cap. |
| Fit objective | `ds_log_likelihood`, `ds_objective` | Unregularised log-likelihood and the MAP objective EM actually maximises. Reported as 0.0 rather than NaN on failure paths; `ds_status` says whether they mean anything. |
| Posterior confidence | `ds_max_posterior_mean` | Mean over items of the largest posterior entry. Not a calibrated probability of being correct, and not to be reported as "confidence" without that caveat. |
| Class counts | `global_class_<i>_count` | Broadcast label counts per class, one key per class. |

## Words to avoid

- **"Confidence"** for `ds_max_posterior_mean`. Say posterior mass. The client-side `confidence_*` metrics are a different quantity -- the softmax maximum each client thresholds its own pseudo-labels on, which is the paper's own term and stays as it is. Calling both of them confidence is what made the earlier write-ups ambiguous.
- **"Consensus"** for the Dawid-Skene output. It is an estimate of the latent class, and it can and does disagree with every kind of vote.
- **"Rejected"** for an excluded client, and **"excluded"** for a rejected message. See the first table.
- **"Client feedback"** without saying which: pseudo-label feedback or model-update feedback. Only the first exists in this codebase.
