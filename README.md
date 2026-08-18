<div align="center">

# SSFL

**Semi-supervised federated learning for IoT intrusion detection**

Reproduce four federated protocols on N-BaIoT—across three non-IID client scenarios—with
deterministic experiments, exhaustive telemetry, and resumable runs.

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Federated_with-Flower-5B3FD3)](https://flower.ai/)
[![GitHub stars](https://img.shields.io/github/stars/crAK1644/Federated-learning-with-psuedo-labeling?style=flat&color=yellow)](https://github.com/crAK1644/Federated-learning-with-psuedo-labeling/stargazers)
[![Last commit](https://img.shields.io/github/last-commit/crAK1644/Federated-learning-with-psuedo-labeling?style=flat)](https://github.com/crAK1644/Federated-learning-with-psuedo-labeling/commits/main)

[Quickstart](#quickstart) ·
[Protocols](#four-protocols-one-framework) ·
[Experiments](#run-the-experiments) ·
[Artifacts](#everything-a-run-records) ·
[Documentation](#documentation)

</div>

---

This repository implements the experimental setup from *“Semisupervised
Federated-Learning-Based Intrusion Detection Method for Internet of Things”*. It compares the
proposed SSFL method with FL, FD, and DS-FL under the same data pipeline, model families, metrics,
and execution framework.

The goal is not a black-box demo. Every assumption, protocol message, checkpoint, metric, and
known departure from the paper is made inspectable.

## What happens in one experiment

```text
┌──────────────────┐   deterministic    ┌──────────────────┐
│  N-BaIoT         │ ── prepare/split ─▶ │  3 non-IID       │
│  9 IoT devices   │                    │  client scenarios │
└──────────────────┘                    └────────┬─────────┘
                                                │
                                     SSFL · FL · FD · DS-FL
                                                │
                                                ▼
┌──────────────────┐                    ┌──────────────────┐
│ reports, figures │ ◀── aggregate ──── │ metrics, comms,  │
│ and audit trails │                    │ telemetry, ckpts │
└──────────────────┘                    └──────────────────┘
```

| Dataset | Task | Classes | Devices | Prepared records | Client scenarios |
| --- | --- | ---: | ---: | ---: | --- |
| [N-BaIoT](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot) | Multi-class IoT intrusion detection | 11 | 9 | 89,000 | 27 / 89 / 89 clients |

## Why this repository

- **Comparable protocols.** SSFL, FL, FD, and DS-FL share one Flower + PyTorch runtime.
- **Reproducible runs.** Seeded preparation, deterministic kernels, resolved configs, and dataset
  manifests make experiment identities explicit.
- **Non-IID by design.** Three scenarios cover shard-based and Dirichlet client partitions.
- **Auditable communication.** Wire payloads, tensor metadata, and paper-equivalent byte counts are
  recorded without copying raw private samples into logs.
- **Built to resume.** Per-round checkpoints and attempt ledgers recover interrupted long runs.
- **Honest about ambiguity.** Every interpretation or deviation from the paper is documented in
  [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Quickstart

**Requirements:** Python 3.12+, [`uv`](https://docs.astral.sh/uv/), `unzip`, `bsdtar`, and at least
15 GiB of free disk space for dataset extraction. CUDA is optional for smoke tests and required by
the canonical paper profile.

```bash
git clone https://github.com/crAK1644/Federated-learning-with-psuedo-labeling.git
cd Federated-learning-with-psuedo-labeling
uv sync
```

Fetch and prepare N-BaIoT:

```bash
scripts/fetch_nbaiot.sh

uv run python -m ssfl.data.prepare_data \
  --input data \
  --output artifacts/data \
  --seed 2023
```

Run the fast test suite and a two-round CPU experiment:

```bash
uv run pytest -q

uv run flwr run . \
  --run-config 'profile="smoke" algorithm="ssfl" scenario=1 device="cpu"' \
  --federation-config 'num-supernodes=27 client-resources-num-cpus=1 client-resources-num-gpus=0 init-args-num-cpus=8' \
  --stream
```

That is the smallest end-to-end path: real prepared data, 27 simulated clients, two communication
rounds, and the complete artifact pipeline.

> [!TIP]
> Already have the UCI archive? Use `scripts/fetch_nbaiot.sh --zip /path/to/archive.zip` to skip
> downloading it again.

## Four protocols, one framework

| Protocol | What is exchanged | Core idea |
| --- | --- | --- |
| **SSFL** | Pseudo-label proposals, votes, and open-set distillation messages | Clients label a shared open set, aggregate consensus, then distill without sharing private samples. |
| **FL** | Model parameters | Standard sample-weighted federated averaging baseline. |
| **FD** | Per-class logits | Federated distillation through class-level knowledge rather than model averaging. |
| **DS-FL** | Open-set logits | Distillation-based semi-supervised baseline over the shared open data. |

Choose a protocol with `algorithm="ssfl"`, `"fl"`, `"fd"`, or `"dsfl"`. Classifiers can use the
paper CNN or the included MLP and LSTM backbones.

## Three non-IID scenarios

| Scenario | Clients | Partition | Purpose |
| ---: | ---: | --- | --- |
| **1** | 27 | McMahan-style shards | Three clients per physical IoT device. |
| **2** | 89 | McMahan-style shards | Fine-grained client population across the 89 device/class subsets. |
| **3** | 89 | Dirichlet, α = 0.1 | Strong heterogeneous class allocation. |

The paper names the scenarios but does not fully specify their allocation mechanics. The exact
interpretation used here is recorded as assumption 13 in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Run the experiments

### CPU smoke matrix

Run short checks across the configured protocol/scenario combinations:

```bash
uv run python -m ssfl.experiments.run_suite \
  --matrix configs/experiments_smoke.yaml
```

### RTX 3090 paper profile

The canonical profile uses CUDA, deterministic kernels, 200 rounds, 5 local epochs, Adam at
`1e-4`, batch size 80, checkpoints every round, and eight concurrent actors at `0.125` GPU each.

```bash
uv run flwr run . \
  --run-config 'profile="paper" algorithm="ssfl" scenario=1 device="cuda"' \
  --federation-config 'num-supernodes=27 client-resources-num-cpus=1 client-resources-num-gpus=0.125 init-args-num-cpus=8 init-args-num-gpus=1' \
  --stream
```

For scenarios 2 and 3, change `scenario` and set `num-supernodes=89`.

Run the proposed SSFL-CNN solution for all three scenarios first:

```bash
uv run python -m ssfl.experiments.run_suite \
  --matrix configs/experiments_solution.yaml \
  --resume
```

Run the complete paper matrix—including protocol comparisons, ablations, threshold policies, and
label-representation studies:

```bash
uv run python -m ssfl.experiments.run_suite \
  --matrix configs/experiments.yaml \
  --resume
```

The solution entries have the same deterministic identities as the full matrix, so the later run
skips results that already completed.

### Resume one interrupted run

Keep the original profile, algorithm, scenario, device, and federation configuration, then add:

```text
resume-from="artifacts/runs/<run-id>"
```

For background runs, `scripts/run_solution_with_alert.sh` records completion in
`artifacts/logs/solution_scenarios.status` and sends a Linux desktop notification when available.

## Everything a run records

Paper runs flush every epoch, client phase, aggregation phase, communication round, evaluation,
checkpoint action, and one-second GPU/system sample.

```text
artifacts/runs/<run-id>/
├── metrics.parquet                 per-round macro/micro/weighted metrics
├── per_class_metrics.parquet       precision, recall, F1, and support by class
├── confusion_matrices.npz          raw confusion matrix for every evaluated round
├── communication.parquet           message shapes, dtypes, and byte accounting
├── checkpoints/                    milestones, final state, and resumable client weights
└── attempts/<attempt-id>/
    ├── events.jsonl[.gz]           compact server aggregation stream
    ├── telemetry/server.jsonl[.gz] server rounds, phases, GPU/system, and evaluation events
    ├── telemetry/clients/*.jsonl[.gz]
    └── aggregation_audit/*.npz     SSFL votes, participation, labels, and masks
```

Per-mini-batch events are off by default to keep the full experiment matrix practical. Enable
`log_every_batch: true` only for a short diagnostic profile. Completed JSONL streams are
losslessly gzip-compressed; active or interrupted runs remain readable for live tailing.

> [!IMPORTANT]
> Checkpoints contain model state and should be treated as sensitive artifacts. Raw private
> samples, model weights, gradients, and secrets are deliberately excluded from JSON telemetry.

## Build the report

Turn completed runs into the paper-style tables and figures:

```bash
uv run python -m ssfl.reporting.build_report \
  --runs artifacts/runs \
  --output artifacts/report
```

## Repository map

```text
src/ssfl/
├── config.py                 validated experiment profiles
├── data/                     discovery, sampling, partitioning, scaling, preparation
├── models.py                 CNN, MLP, and LSTM backbones
├── training.py               supervised and distillation loops
├── protocols/                SSFL, FL, FD, and DS-FL aggregation contracts
├── strategies/               Flower strategies for each protocol
├── client_app.py             Flower ClientApp
├── server_app.py             Flower ServerApp
├── experiments/run_suite.py  deterministic matrix runner
└── reporting/build_report.py tables and figures

configs/                      smoke, paper, deployment, and experiment matrices
deployment/                   SuperLink/SuperNode launch scaffold and TLS tooling
tests/                        unit, protocol, privacy, integration, and deployment tests
artifacts/                    prepared data and run outputs (gitignored)
```

## Simulation and real deployment

The commands above use Flower's local Simulation/Ray backend. The separate `deployment/` scaffold
runs real SuperLink and SuperNode processes with insecure-development and TLS profiles. See
[deployment/README.md](deployment/README.md) for launch and certificate instructions.

## Reproducibility boundary

This is a research reproduction, not an official implementation from the paper's authors. Some
paper details require explicit interpretation, and those choices can affect comparability.

Before treating a run as a reproduced result, review:

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for numbered assumptions, deviations, and known issues.
- [DATA_CARD.md](DATA_CARD.md) for source data, transformations, split counts, and leakage controls.
- [MODEL_CARD.md](MODEL_CARD.md) for architecture, intended use, evaluation, and limitations.
- [SECURITY.md](SECURITY.md) for the threat model and protocol privacy boundaries.

## Documentation

| Document | What it answers |
| --- | --- |
| [Implementation plan](SSFL_IMPLEMENTATION_PLAN.md) | What was built and how the paper maps to the codebase. |
| [Reproducibility notes](REPRODUCIBILITY.md) | Which ambiguities were resolved, what changed, and why. |
| [Data card](DATA_CARD.md) | Where N-BaIoT comes from and how it is prepared. |
| [Model card](MODEL_CARD.md) | What the models are intended for and where they fall short. |
| [Security notes](SECURITY.md) | What crosses the wire and what remains private. |
| [Dawid-Skene glossary](DAWID_SKENE_GLOSSARY.md) | What every label-aggregation metric means, and which four exclusions are not the same. |
| [Deployment guide](deployment/README.md) | How to move from local simulation to real Flower processes. |

---

<div align="center">

**Private data stays local. Experimental decisions stay visible.**

</div>
