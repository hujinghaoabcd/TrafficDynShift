# TrafficDynShift

**From Network Morphology to Propagation Dynamics: Cross-Network Traffic Forecasting under Distribution Shift**

TrafficDynShift is a research codebase for testing whether traffic **propagation dynamics** provide a more transferable spatial representation than network morphology alone in zero-shot cross-network traffic forecasting.

> Status: research prototype / paper implementation in active development.

## Core idea

The project deliberately separates two roles:

- **Topology constrains feasible interactions.**
- **Propagation dynamics describe how predictive interactions unfold over temporal lags.**

For every candidate source-target edge, a shared estimator learns a lag-response profile. Irregular neighborhoods are then reorganized into a fixed number of shared propagation slots. The same forecasting parameters can therefore operate on networks with different node counts and local morphologies.

The learned response is **not** claimed to be physical travel time or a strict causal effect. It is a task-driven predictive propagation response.

## Why this repository is structured as a research project

The repository uses a modern `src/` layout, grouped Hydra configuration, tests, CI and persistent design/protocol/handoff documents. The goal is reproducible scientific experimentation rather than a one-off paper script.

## Parameter management: Hydra

Hydra YAML groups are the single source of truth:

```text
configs/
├── config.yaml
├── data/pems.yaml
├── model/pdr.yaml
├── training/default.yaml
├── experiment/
│   ├── zero_shot.yaml
│   └── morphology_ood.yaml
└── runtime/default.yaml
```

Override only what changes:

```bash
python scripts/train.py \
  experiment.target_region=PeMSD8 \
  experiment.source_regions='[PeMSD3,PeMSD4,PeMSD7]' \
  runtime.seed=42 \
  model.num_lags=6 \
  model.num_slots=6
```

Every run records Hydra's config and overrides and also writes a fully resolved `resolved_config.yaml` beside checkpoints and metrics.

## Project layout

```text
TrafficDynShift/
├── configs/                 # Hydra config groups
├── data/                    # local data only; raw files ignored
├── docs/                    # design, theory, protocol, status, handoff
├── outputs/                 # run artifacts; ignored except .gitkeep
├── scripts/                 # train/evaluate/demo entry points
├── src/trafficdynshift/
│   ├── analysis/
│   ├── data/
│   ├── graph/
│   ├── models/
│   ├── propagation/
│   ├── training/
│   └── utils/
└── tests/
```

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install
```

## Smoke test

```bash
python scripts/demo.py
pytest
```

The same model instance is tested on graphs with different node counts.

## Data

See [`data/README.md`](data/README.md). The default config expects PeMSD3, PeMSD4, PeMSD7 and PeMSD8 under `data/<region>/` with an NPZ traffic file and adjacency matrix. Raw datasets are not committed.

## Strict zero-shot experiment

Example held-out PeMSD8 run:

```bash
python scripts/train.py \
  experiment.target_region=PeMSD8 \
  experiment.source_regions='[PeMSD3,PeMSD4,PeMSD7]' \
  runtime.seed=42
```

Model selection uses source-region validation only. Target data are excluded from training, early stopping and hyperparameter selection.

## Development rules

1. Keep the paper centered on **network morphology -> propagation dynamics**.
2. Do not introduce target-specific node embeddings in strict zero-shot experiments.
3. Do not treat marginal domain alignment as the core objective.
4. Do not call the learned response physical causality/travel time without new evidence.
5. Add complexity only when a specific experiment demonstrates that it is necessary.

See [`docs/DESIGN.md`](docs/DESIGN.md), [`docs/THEORY.md`](docs/THEORY.md), and [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) before changing the method.
