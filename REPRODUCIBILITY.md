# Reproducibility Record

## Repository / data / paper provenance

- Git: initial commit `c57f42b` on `master`. Raw dataset and paper PDF are gitignored (not tracked; too large / copyrighted).
- Data mount: `data/` — 89 flat CSVs named `<device_id>.<family>.<attack>.csv` (e.g. `1.benign.csv`,
  `4.mirai.udp.csv`), `device_info.csv` (DeviceID → name), `data_summary.csv` (per-file row counts),
  `README.md` (UCI dataset card). Total on-disk size ~7.6 GB. **No nested directories** — this is the
  flat variant of the official N-BaIoT layout, not the ZIP/nested hierarchy; `prepare_data` discovery
  handles both but only the flat form has been observed here.
- Devices 3 (`Ennio_Doorbell`) and 7 (`Samsung_SNH_1011_N_Webcam`) have 6 classes (benign + 5 gafgyt,
  no mirai files present) → `7×11 + 2×6 = 89` CSVs, confirmed by direct listing.
- 115 named numeric feature columns confirmed via header inspection of `1.benign.csv`. Every
  device/class file has far more than 1,000 rows per `data_summary.csv`.
- Paper PDF SHA-256:
  `0e751623fe0d910804112257cbeb2881fb9d5be688637c550a2f1a4723890a8a`
  (`Semisupervised_Federated-Learning-Based_Intrusion_Detection_Method_for_Internet_of_Things .pdf`).

## Host environment (build/verification machine)

- OS: macOS (Darwin 25.5.0), Apple Silicon `arm64` (Apple M4).
- CPU: 10 cores. RAM: 16 GiB (17179869184 bytes). Free disk at build start: ~21 GiB (data dir is
  external to prepared artifacts, which are small — the mini dataset is ~89,000 rows × 115 float32
  features ≈ tens of MB, so disk is not a constraint for `artifacts/`).
- Accelerator: **MPS available, CUDA not available** (`torch.backends.mps.is_available() == True`,
  `torch.cuda.is_available() == False`). Device selection must support `cpu` / `cuda` / `mps`; CI and
  the CPU-smoke acceptance gate run with `device=cpu` for determinism (MPS float32 kernels are not
  bit-reproducible across runs in the same way CPU is), GPU-scale `paper` runs are gated on
  user-provided CUDA hardware and are not executed as part of this build.
- **The project directory's absolute path must not contain spaces.** This checkout lives under
  `.../Federated learning with psuedo labeling/...`. `flwr run`'s local-simulation SuperLink
  initializes its SQLite state via Alembic, and the installed `flwr` package builds Alembic's
  `version_locations` config value by joining path(s) with a plain space (`flwr/supercore/state/
  alembic/utils.py::build_alembic_config`); Alembic itself then *splits* that value on whitespace to
  support multiple search paths. A repo path containing spaces gets shredded into bogus fragments,
  Alembic silently discovers zero migration scripts, and the state DB ends up with only an
  `alembic_version` table — every `flwr run` then fails opaquely (`no such table: fab`/`task`/...).
  Workaround applied here: physically relocate `.venv` to a space-free path
  (`~/.venvs/ssfl`) and leave a symlink at `.venv` inside the repo — `Path(__file__).resolve()`
  follows the symlink to the real, space-free location, so every path Alembic builds from
  `flwr`'s own installed location is clean. This is an environment workaround, not a code
  change; recreating `.venv` fresh with a plain `uv sync` (which would put it back inside the
  spaced path) reintroduces the bug — re-apply the move+symlink after any `uv venv`/`uv sync`
  that recreates `.venv` from scratch.

## Dependency resolution

Resolved via `uv sync` against `requires-python = ">=3.12"`, **no hard version pins** (per user
decision: "latest installable" rather than the spec's literal `torch==2.13.0` / `flwr==1.32.1`).
Resolution on this machine, recorded here for exact reproducibility:

| Package | Resolved version |
|---|---|
| Python | 3.12 |
| torch | 2.13.0 |
| flwr (`flwr[simulation]`) | 1.32.1 |
| ray | 2.55.1 |
| pandas | 3.0.3 |
| pyarrow | 25.0.0 |
| scikit-learn | 1.9.0 |
| pydantic | 2.13.4 |
| numpy | 2.5.1 |
| pyyaml | 6.0.3 |
| matplotlib | 3.11.1 |

Coincidentally, `torch` and `flwr` resolved to exactly the versions the original spec hard-pins
(2.13.0 / 1.32.1) — no conflict between "latest installable" and the spec's literal pins on this
machine at build time. Future re-resolution is not guaranteed to match; `uv.lock` is the source of
truth going forward.

## Flower Message API surface (flwr 1.32.1) — confirmed by direct inspection

This project targets the **modern message-based API**, not the deprecated `NumPyClient`/legacy
`Client` interface. Confirmed present in the installed package:

- `flwr.app`: `Message`, `RecordDict`, `ArrayRecord`, `Array`, `ConfigRecord`, `ConfigRecordValues`,
  `MetricRecord`, `MetricRecordValues`, `Context`, `Metadata`, `Error`.
- `flwr.clientapp.ClientApp`: decorator-based handler registration —
  `@app.train()`, `@app.evaluate()`, `@app.query()`, each wrapping
  `def handler(message: Message, context: Context) -> Message`. A single `ClientApp` instance is
  used per algorithm; SSFL's two logical sub-phases (proposal, distillation) are both registered
  under `@app.train()` and dispatched internally on `message.content["config"]["phase"]` (a string
  in the shared `ConfigRecord` contract), since both are training-type tasks from the client's
  perspective and Flower's `MessageType` taxonomy does not have a third slot for "phase B".
- `flwr.serverapp.ServerApp`: `@app.main()` decorator wraps the server entrypoint.
- `flwr.serverapp.strategy.strategy.Strategy` (custom-strategy base class) — abstract methods
  `configure_train`, `aggregate_train`, `configure_evaluate`, `aggregate_evaluate`, `summary`;
  concrete `start(grid, initial_arrays, num_rounds, timeout, train_config, evaluate_config,
  evaluate_fn) -> Result`. `start()` is **overridable** (not abstract) — SSFL/FD/DS-FL override it
  directly to run two `Grid.send_and_receive()` exchanges per paper round (proposal + distillation),
  matching the spec's explicit instruction ("custom Strategy whose `start` method runs two Message-API
  exchanges per paper communication round"). FL reuses the standard single
  configure_train/aggregate_train round shape and is implemented either as a thin subclass of the
  library-provided `flwr.serverapp.strategy.FedAvg` or an equivalent custom `Strategy` — decided at
  M4 implementation time based on whether FedAvg's built-in aggregation exactly satisfies the
  sample-weighted spec without a wrapper fighting the abstraction.
- `flwr.serverapp.Grid`: `create_message`, `get_node_ids`, `pull_messages`, `push_messages`, `run`,
  `send_and_receive`, `set_run` — the send/receive primitive `Strategy.start()` overrides use directly.
- `Context(run_id, node_id, node_config, state: RecordDict, run_config, series_id=0)` — `context.state`
  is a `RecordDict`, confirming the spec's design of persisting client classifier/discriminator
  weights in `Context.state` between rounds (never placed in outgoing `Message` content for SSFL).
- `flwr.serverapp.strategy` ships built-in `FedAvg`, `FedProx`, `FedAdam`, `FedYogi`, `Krum`,
  `Bulyan`, DP-wrapped variants, etc. — available as reference/reuse candidates for the FL baseline
  and later robust-aggregation extension profiles (M12/security extensions), not required for
  SSFL/FD/DS-FL which need bespoke two-phase logic.

## Implementation assumptions register

The paper leaves the following underspecified. Each is resolved with an explicit, documented default,
exposed as configuration where a reader might reasonably want the alternative.

| # | Ambiguity | Resolution | Rationale |
|---|---|---|---|
| 1 | Batch size 80 (Table I) vs 100 (Section V-C prose) | Canonical = **80** (Table I, structural). `paper_text_batch` profile override = 100. | Table I is the more concrete, structural source; prose is more likely to be an approximation/typo. |
| 2 | CNN "number of fully connected layers" (prose vs Table I) | **Table I** governs: one 128-unit dense hidden layer before each output head. | Table I gives exact per-layer filter/kernel/stride counts and output shapes; prose is contradictory and less falsifiable. |
| 3 | Min-max scaler fitting scope (paper only says values end up in [0,1]) | Canonical `paper` profile = **`all_mini`** (fit across the complete sampled mini-dataset, all splits transformed). `private_only` implemented as a leakage-safe secondary profile. | Matches the paper's blanket "[0,1]" claim most directly; `private_only` is offered because `all_mini` technically leaks open/test feature ranges into private-only training in a real deployment. |
| 4 | DS-FL sharpening temperature | **T = 0.1**, fixed per spec instruction (not derived from paper text). | Spec-mandated default; flagged as assumption since the paper's own DS-FL comparison does not specify T. |
| 5 | FD loss composition (ground-truth CE vs teacher-CE weighting) | **Equal weighting** (unweighted sum / average of the two CE terms). | Paper does not report a coefficient; equal weighting is the simplest unbiased default. |
| 6 | Optimizer-state persistence across rounds | **Adam state reinitialized at the start of every Flower task; model weights persist** across rounds via `Context.state`. | Spec-mandated; avoids stale second-moment estimates biasing a freshly-relevant-data local step while keeping the learned representation. |
| 7 | Server classifier initialization (SSFL Phase B) | **Common seeded checkpoint at round 0**, shared by all clients' initial classifiers and the server's persistent classifier; discriminators from a separately seeded common checkpoint. | Ensures fair cross-algorithm comparison — no client/server starts from a different random basin. |
| 8 | No-vote (all-abstain) handling | Emit global label **`-1`** with `valid_mask=False`; **never fabricate a class**. | Spec-mandated; fabricating a label under all-abstain would silently corrupt distillation targets with no signal backing them. |
| 9 | Tied-vote handling | **Lowest class index wins**, matching ordinary deterministic `argmax` tie-breaking (`numpy`/`torch` default). | Spec-mandated; keeps aggregation deterministic and reproducible without an arbitrary random tiebreaker. |
| 10 | MLP / LSTM architectures (not specified by the paper at all) | **MLP**: flatten 115 → 512 → 256 → 128 → output, ReLU between hidden layers, no dropout/BN. **LSTM**: reshape as 5 timesteps × 23 features, 2 layers hidden size 128 batch-first, final hidden state → 128 → 128 → output, ReLU between dense layers. | Fixed, documented architectures chosen to be reasonably capacity-matched to the CNN rather than tuned; the paper gives no basis for any specific alternative. |
| 11 | Meaning of a "communication round" in SSFL's two-phase exchange | **One paper round = exactly two Message-API exchanges** (Phase A proposal/vote, Phase B distillation), both counted as a single logical round for round-indexed comparisons (rounds 10/50/100/150/200, Table III). | Matches the spec's explicit protocol description; keeps round indexing comparable across SSFL/FL/FD/DS-FL despite SSFL doing more wire exchanges per round. |
| 12 | Device selection determinism | Deterministic-kernel mode is enabled and enforced in the `paper` profile on **CPU**; **MPS is not treated as a deterministic backend** (Apple's MPS float32 kernels do not currently guarantee the same run-to-run bitwise reproducibility as CPU). CUDA path uses `torch.use_deterministic_algorithms(True)` + cuBLAS workspace config when available, but is unverified on this machine (no CUDA device present). | The CPU-smoke acceptance gate is the determinism contract this build actually verifies; GPU determinism is configured but not exercised here. |
| 13 | Scenario 1/2/3 client-partition shard algorithm (paper only names the scenarios, not a concrete allocation procedure) | A single McMahan-style shard partition for **both** scenario 1 and 2: sort a device's private rows by label, cut into `clients_per_device * shards_per_client` equal contiguous shards, deterministically shuffle shard order (seeded via `numpy.random.SeedSequence([seed, device_id, scenario])`), deal `shards_per_client=2` shards per client. Scenario 1 = 3 clients/device, 6 shards (700-row classes don't divide evenly into sixths -> shards straddle class boundaries -> mild mixing). Scenario 2 = `num_classes` clients/device, `2*num_classes` shards (divides evenly into class halves -> shards are always class-pure -> after shuffle+deal most clients land on exactly 1-2 classes, materially more severe skew, confirmed visually in `artifacts/data/plots/scenario_{1,2}_allocation.png`). Scenario 3 reuses scenario 2's client-per-device count; each class's 700 rows are allocated across that device's clients via `Dirichlet(alpha=0.1)` proportions -> `multinomial` counts, redrawn deterministically (incrementing attempt salt in the `SeedSequence`) until every client has >=1 example. Per-(device,class) sampling RNG seeds use `SeedSequence([seed, device_id, label])` (entropy mixing, not plain addition) so e.g. (device=2,label=3) and (device=3,label=2) never collide. | Standard, well-known non-IID shard partitioning reused instead of inventing a bespoke scheme; parametrizing only shard/client counts per scenario keeps one algorithm auditable instead of three bespoke ones, and the resulting severity ordering (S1 mild, S2/S3 severe) matches the qualitative behavior the paper's scenario framing implies. |
| 14 | FD ground-truth vs teacher-CE loss weighting (paper describes both loss terms but not their relative weight) | **Equal weight, 1:1 sum**: `loss = CE(logits, true_label) + teacher_distribution_loss(logits, leave_self_out_target)`, restricted to private examples whose class has a valid (>=1 other contributor) leave-self-out target. See `src/ssfl/protocols/fd.py::client_distillation_step`. | No basis in the paper for any other ratio; equal weighting is the simplest assumption and keeps both signals influencing training rather than one dominating by construction. |
| 15 | FD centralized-evaluation methodology (paper is silent — FD has no natural global/server model, unlike SSFL and DS-FL which both train a persistent server-side classifier on the same broadcast signal clients receive) | **No `evaluate_fn` is registered for FD** in `server_app.py` — there is no server-side model to evaluate. Each FD client instead reports a `private_accuracy` metric during the evaluate exchange: its own post-distillation training accuracy on its own private labeled data. This is explicitly **not** a held-out test-set number and must never be compared directly to SSFL/FL/DS-FL's centralized test accuracy. | FD is inherently personalized-per-client (`leave_self_out_targets` gives every client a different teacher signal); inventing a fake shared global model to get a comparable centralized number would misrepresent what FD actually produces. |
| 16 | Per-round/per-phase random seed derivation (needed so e.g. SSFL's classifier-training, discriminator-training, and distillation sub-phases within one round don't share RNG state, while staying deterministic given the run seed) | `_seed(base_seed, server_round, salt) = (base_seed * 1000 + server_round * 10 + salt) % 2**31`, with a distinct integer `salt` per sub-phase (e.g. 0/1/2 for classifier/discriminator/distillation). See `src/ssfl/client_app.py`. | Simple, collision-free (for realistic `server_round`/salt ranges) affine derivation; avoids re-seeding all sub-phases of a round identically, which would correlate their randomness (e.g. classifier and discriminator dropout/shuffle order becoming coupled) with no benefit. |
| 17 | CPU thread allocation under `flwr run` simulation (many client processes run concurrently on one machine; PyTorch's default is one full-core OMP thread pool *per process*) | `resolve_device` calls `torch.set_num_threads(1)` whenever it resolves to `cpu` (`src/ssfl/device.py::_cpu_device`). Discovered by `sample`-profiling a live simulation worker stuck at low wall-clock progress despite real backward-pass compute: with `max_concurrent_clients: 8` and a 10-core host, N concurrent actors each spawning a 10-thread OMP pool oversubscribes the machine by ~N×; for this model's tiny per-op tensor sizes (23 channels, length 5) the OpenMP fork/join barrier overhead dominates the actual FLOPs, so 1 thread/process is faster in aggregate than 10. | Not a correctness fix — a performance one. Left unbounded, a nominal 2-round CPU smoke run took 100+ minutes; capped at 1 thread/process it completes in ~11 minutes end to end. |
| 18 | `context.run_config` vs `pyproject.toml`'s `[tool.flwr.app.config]` (Flower populates `run_config` from **every** key in that table on **every** run, not just keys explicitly passed via `flwr run --run-config`; it also rejects a `--run-config` override for any key not already declared there — `Key 'X' is not present in the main dictionary`) | `[tool.flwr.app.config]` declares only `profile = "paper"` plus a **small fixed allowlist** of knobs genuinely meant to be picked per invocation on top of a profile: `algorithm`, `scenario`, `device`. `experiment_config_from_run_config` (`src/ssfl/config.py`) loads `configs/<profile>.yaml` as the base and layers the full `run_config` dict on top — so any *other* key baked into `pyproject.toml` would be present in `run_config` on every run (profile-selected or not) and would silently clobber that field from any non-`paper` profile YAML, since there is no way to distinguish "value came from a real CLI override" from "value came from the pyproject.toml default" once Flower has flattened them into one dict. Consequence of getting this wrong (found live): a `flwr run --run-config "profile='smoke' ..."` invocation resolved to `num_server_rounds: 200`/`local_epochs: 5` (the old baked-in paper-shaped defaults) instead of smoke's `2`/`1` — the run wasn't hung, it was correctly executing a full paper-scale job under a smoke label. | Profile-defining fields (`num-server-rounds`, `local-epochs`, `batch-size`, `run-kind`, ...) must never be added to `[tool.flwr.app.config]` — express a different combination as a new `configs/<name>.yaml` and select it via `profile=<name>` instead. `algorithm`/`scenario`/`device` are the exception because every invocation needs to state them explicitly anyway; their `pyproject.toml` defaults mirror `paper.yaml` so selecting `profile="paper"` alone still needs no other overrides. |
| 19 | Flower's `Strategy.start()` round loop threads a single `arrays` variable through **both** phases of every round: it is updated only by `aggregate_train`'s return, never by `aggregate_evaluate` (which SSFL/FD/DS-FL only ever return metrics from) — so whatever `configure_evaluate` broadcast in round N is, by construction, exactly what `configure_train` receives as its `arrays` argument in round N+1, even though the proposal/local-training phase never reads it. | `configure_train` in all three custom strategies (`strategies/ssfl.py`, `strategies/fd.py`, `strategies/dsfl.py`) now ignores the framework-supplied `arrays` argument and always sends an empty `ArrayRecord()` on the train-phase message; clients only read `config["server-round"]` there. Found via the M6 comms ledger (`comms.py`) itself: round 2's SSFL train-phase message logged 80,100 logical bytes of `global_labels`/`valid_mask` that `client_app.py::_ssfl_train` never touches. | Without this, every round beyond the first silently over-counts communication for SSFL (stale `global_labels`/`valid_mask`), FD (stale `global_sum`/`contributor_counts`), and DS-FL (stale `sharpened_targets`, the largest of the three) — inflating the very comm-vs-accuracy numbers (Table IV) these protocols are supposed to look efficient on. FL is unaffected (stock `FedAvg` legitimately needs the real model arrays every train phase). |
| 20 | `flwr run`'s `--run-config` value is parsed as **TOML**, not shell-style `key=value` — an unquoted string value (e.g. `profile=smoke`) makes the CLI reject the *entire* `--run-config` string with `The provided configuration string is in an invalid format`, even though the same syntax is accepted (and shown) in Flower's own `--help` examples for simple single-key cases. | Every string-typed override in `run_suite.py`'s generated `--run-config` (`profile`, `algorithm`, `device`) is wrapped in literal double quotes (`profile="smoke_ablation_no_discriminator"`); int-typed overrides (`scenario`) are left unquoted, since TOML would reject a quoted integer the same way. | Found live: the first real `run_suite.py` invocation against `configs/experiments_smoke.yaml` reported all 3 matrix entries as `"failed"`; the underlying `flwr` stderr (only visible by reading the suite's tee'd log, not the JSON report) named the exact parse error. Any future field added to the generated `--run-config` string must be quoted or not per its TOML type, not by habit. |
| 21 | `flwr run --federation-config` no longer accepts the `options.`-prefixed key names shown in older Flower docs/examples (e.g. `options.num-supernodes=27`) — the installed flwr 1.32.1 hard-rejects it with `Unknown simulation config field(s): options.num_supernodes`, not merely a deprecation warning (a deprecation notice about `options.` fields *is* also printed, which can read as "this still works, just discouraged" when it does not). | `run_suite.py` passes the bare key: `--federation-config "num-supernodes=<count>"`. | Second of the three failures behind the same all-3-entries-failed `run_suite.py` report as #20 — fixed in the same pass once the TOML-quoting fix alone didn't clear the error. |
| 22 | Without `--stream`, `flwr run` submits the run to the SuperLink and returns almost immediately (`Successfully started run <id>`) instead of blocking until the run finishes — easy to misread as "the run is fast" or "the run is hanging" depending on what's watched next, since nothing about the CLI's own output signals that it detached. | `run_suite.py`'s `cmd` list always includes `--stream`, which is what makes the `subprocess.run(cmd, ...)` call in `run_suite()` actually block until the simulation completes and `summary.json` exists to check. | Confirmed by direct A/B: omitting `--stream` returned control in under a second with `ps aux` showing no Ray actor processes ever spawned and the target run directory's `summary.json` mtime unchanged from a prior run hours earlier; re-running the identical command with `--stream` ran past a 600-second foreground timeout and had to continue as a background task, consistent with this repo's real ~10+ minute-per-entry CPU smoke duration. Also interacts with #21's leaked-process risk: a detached (non-`--stream`) run still spawns a SuperLink/Ray process tree in the background with no local handle to await or clean up if the invoking shell/task exits first. |

## Open items carried to later milestones

- Exact `Strategy.start()` override signature usage (how `evaluate_fn`, `train_config`/
  `evaluate_config` interact with a fully custom two-exchange loop) will be finalized against real
  code in M4/M5, not just introspection — introspection here is sufficient to unblock M1–M3.
- `flwr.compat` exists in 1.32.1 (legacy `NumPyClient`/`Client` compatibility shim) — confirmed
  present but intentionally unused; this project targets the message API exclusively per spec.
