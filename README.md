# SSFL — Semisupervised Federated Learning for IoT Intrusion Detection

Reproduction of *"Semisupervised Federated-Learning-Based Intrusion Detection Method for Internet
of Things"* on N-BaIoT: four federated protocols (SSFL, FL, FD, DS-FL) across three non-IID client
scenarios, built on Flower + PyTorch. See `SSFL_IMPLEMENTATION_PLAN.md` for the full spec this repo
implements, and `REPRODUCIBILITY.md` for every deviation from the paper and why.

## Layout

```
src/ssfl/
  config.py                pydantic profile config (smoke/paper/deployment/...)
  data/                     prepare_data, discovery, partition, datasets, scaling
  models.py  training.py    CNN/MLP/LSTM backbones, train/eval loops
  protocols/                per-algorithm aggregation logic (ssfl, fl, fd, dsfl) + message contract
  strategies/                per-algorithm Flower Strategy
  client_app.py  server_app.py
  experiments/run_suite.py  reporting/build_report.py
deployment/                 real (non-simulation) SuperLink/SuperNode launch scripts + TLS certs
configs/                    smoke / paper / deployment / experiments*.yaml profiles
tests/                      unit, protocol/security, integration, deployment
artifacts/                  prepared data + run outputs (gitignored)
```

## Quickstart (CPU smoke)

```bash
uv sync

uv run python -m ssfl.data.prepare_data --input data --output artifacts/data --seed 2023

uv run pytest -q                    # unit + protocol/security tests (~45s)
uv run pytest -m slow -q            # + real flwr simulations and deployment processes (minutes)

uv run flwr run . --run-config 'profile="smoke" algorithm="ssfl" scenario=1 device="cpu"'

uv run python -m ssfl.experiments.run_suite --matrix configs/experiments_smoke.yaml
uv run python -m ssfl.reporting.build_report --runs artifacts/runs --output artifacts/report
```

`algorithm` is one of `ssfl`/`fl`/`fd`/`dsfl`; `scenario` is `1` (27 clients), `2`, or `3` (89
clients each). Swap `profile="smoke"` for `profile="paper"` for the real 200-round runs — those are
configured here but gated on GPU hardware, not executed in this repo (see `REPRODUCIBILITY.md`).

## Real (non-simulation) deployment

`flwr run` above uses Flower's local Simulation/Ray backend. `deployment/` has a separate scaffold
for real SuperLink + SuperNode processes (insecure-dev and TLS profiles) — see `deployment/README.md`.

## Docs

- `REPRODUCIBILITY.md` — every assumption, deviation, and resolved ambiguity vs. the paper, numbered.
- `DATA_CARD.md` — N-BaIoT source, transformations, splits, leakage mitigations.
- `MODEL_CARD.md` — backbones, intended use, evaluation, limitations.
- `SECURITY.md` — threat model, verified privacy boundaries per protocol, deployment gap notes.
