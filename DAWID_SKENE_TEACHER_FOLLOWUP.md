# Dawid-Skene teacher follow-up

## Direct answers for the report

- Scenario 1 has 27 clients and 62,300 private training samples in total.
- Six clients have 1,400 samples each: `s1-d3-c0/c1/c2` and `s1-d7-c0/c1/c2`.
- Seven clients have 2,566 samples each: `s1-d1-c1`, `s1-d2-c1`, `s1-d4-c1`,
  `s1-d5-c2`, `s1-d6-c0`, `s1-d8-c1`, and `s1-d9-c0`.
- The remaining fourteen clients have 2,567 samples each: `s1-d1-c0/c2`,
  `s1-d2-c0/c2`, `s1-d4-c0/c2`, `s1-d5-c0/c1`, `s1-d6-c1/c2`,
  `s1-d8-c0/c2`, and `s1-d9-c1/c2`.
- The server and every client share the same 8,900-sample unlabeled open set. The 17,800-sample
  sealed test set is evaluation-only and is not used to fit Dawid-Skene.

The active run used Dawid-Skene in 39 rounds because those fits passed every gate. The other 161
rounds used majority vote: 34 fits hit the 100-iteration cap without convergence, 120 converged
but failed the class-identity/diagonal-ratio gate, and 7 passed that gate but had less than 50%
agreement with majority on comparable items. These are mutually exclusive first-failure statuses,
so `34 + 120 + 7 = 161`.

## Is the implementation correct?

The current evidence supports the model implementation, with one repaired edge case, but it is
important not to describe the `rater` package as a drop-in numerical oracle:

- Both implementations use the same Dawid-Skene observation model: a class prior and one
  true-class-by-emitted-class confusion matrix per rater/client.
- `rater` performs Bayesian inference in Stan (MCMC or numerical MAP optimisation), has different
  default priors, and supports repeated ratings by the same rater.
- This repository performs deterministic MAP-EM and receives at most one label from each client
  for each shared item. Exact parameter equality is therefore not expected under default settings.

The presentation's evaluation from approximately 9 minutes onward was reproduced behaviourally
with four classes, five raters, 100 items per replication, 300 replications, rater accuracy 0.7,
and errors distributed uniformly over the other classes:

| Informative raters | Spammers | Majority accuracy | DS accuracy | Median EM steps |
|---:|---:|---:|---:|---:|
| 5 | 0 | 0.9112 | 0.8861 | 19.5 |
| 4 | 1 | 0.8366 | 0.8419 | 25.0 |
| 3 | 2 | 0.6857 | 0.7738 | 33.0 |
| 2 | 3 | 0.4488 | 0.6456 | 53.0 |
| 1 | 4 | 0.3047 | 0.3510 | 68.0 |
| 0 | 5 | 0.2544 | 0.2522 | 68.0 |

This reproduces the two claims in the video: majority has a small advantage when all raters are
equally informative, while full Dawid-Skene becomes useful when systematic raters must be learned
and discounted. Run it with:

```bash
uv run python scripts/reproduce_rater_evaluation.py
```

During the audit, a missing-data edge case was found and fixed. All-abstain item columns were
excluded from the likelihood but were still entering the class-prior M-step, so appending unused
items could shift an otherwise identical fit. The estimator now excludes those columns from both.
A regression test appends 1,000 all-abstain columns and requires identical labels, confusion
matrices, iteration count, and objective on the observed items. This was not the main cause of the
39/161 result because the recorded run had more than 99% open-set coverage.

For a stricter cross-implementation check, use a complete synthetic design with one rating per
rater-item, fit `rater(..., method="optim")`, set its Dirichlet parameters to `1 +` this
repository's pseudocounts, and compare class priors, confusion matrices, and latent labels after an
optimal class permutation. The anesthesia data should not be used for exact parity without an
adapter because it contains three repeated ratings by rater 1 for every item.

## Controlled experiments

### 1. Algorithm-level tie and convergence controls (completed)

The deterministic tie test contains 400 anchor items and 200 target items. Two reliable raters
and two inverted raters cast a 2-2 tie on every target; a fifth rater abstains on targets but labels
the anchors. Majority gets 55% of tied targets correct because it resolves every tie toward the
lower class index. Dawid-Skene uses the anchors to identify the rater groups, reaches 100% on the
tied targets, and settles in 5 EM steps. When the anchors are removed, the problem is symmetric
and non-identifiable; the safety check correctly rejects it rather than inventing a winner.

A separate high-consensus control with seven 90%-accurate clients and 1,000 items reaches 99.9%
label accuracy and settles in 4 steps. These two deterministic tests cover the requested artificial
tie and small-`steps before settled` cases at the aggregation layer.

### 2. Closed-loop 50-round FL experiment (next run)

Use `gafgyt.combo` versus `gafgyt.junk` as the positive-control pair. They are difficult for a
linear probe (0.6786) but remain separable by a random forest (0.9933) and 1-nearest-neighbour
(0.9861), so a successful learner has real signal. Do not use the current prepared
`gafgyt.tcp`/`gafgyt.udp` pair as the only positive control: after the existing preprocessing they
are effectively non-identifiable (linear probe 0.5008, random forest 0.5000). Keep that pair as a
negative control, or repeat it only after the quantile-prepared data fix.

For each seed, compare:

1. balanced allocation: both target classes are split evenly across two designated clients;
2. specialist allocation: one designated client receives the combo examples and the other the
   junk examples; and
3. the same two allocations under majority, Dawid-Skene shadow, and Dawid-Skene active.

Keep the total number of private samples per client, the shared open set, the test set, model,
epochs, learning rate, and all non-aggregation settings fixed. Swap equal numbers of background
examples when moving the target classes so that allocation is the only treatment. Use one pilot
seed first, then five seeds if the instrumentation and direction are sound.

Report per round for 50 rounds: overall accuracy and macro-F1, minimum recall/F1 across the target
pair, target-pair confusion counts, number of tied open items, open-label accuracy, DS status,
whether DS was applied, EM iterations, diagonal ratio, and majority agreement. The primary
endpoint should be the mean of the weaker target-class recall over rounds 41-50, not final overall
accuracy alone.

The aggregation-layer controls are complete. The 50-round closed-loop FL matrix still requires a
separate copied dataset artifact (to preserve the original manifest) and GPU execution; it has not
yet been run.
