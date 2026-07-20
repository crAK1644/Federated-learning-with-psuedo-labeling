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

## Open items carried to later milestones

- Exact `Strategy.start()` override signature usage (how `evaluate_fn`, `train_config`/
  `evaluate_config` interact with a fully custom two-exchange loop) will be finalized against real
  code in M4/M5, not just introspection — introspection here is sufficient to unblock M1–M3.
- `flwr.compat` exists in 1.32.1 (legacy `NumPyClient`/`Client` compatibility shim) — confirmed
  present but intentionally unused; this project targets the message API exclusively per spec.
