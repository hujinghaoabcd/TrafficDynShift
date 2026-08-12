# Handoff

## Current research question

Can propagation dynamics provide a more transferable spatial description than network morphology for cross-network traffic forecasting under distribution shift?

## Current implementation

`PDRForecaster` consumes normalized history `(B,H,N,C)` and sparse candidate edges `(2,E)`. A shared response estimator produces `(B,E,L)` lag profiles, which are mapped to `(B,N,K,D)` shared propagation slots. The node count is preserved and no node-specific learned embeddings are used.

## Configuration

Use Hydra YAML groups under `configs/`. Do not reintroduce large argparse blocks or copy hyperparameters into scripts. Example:

```bash
python scripts/train.py experiment.target_region=PeMSD8 \
  experiment.source_regions='[PeMSD3,PeMSD4,PeMSD7]' \
  runtime.seed=42 model.num_lags=6 model.num_slots=6
```

Hydra writes the run config and overrides; `train.py` also writes `resolved_config.yaml`.

## Immediate next check

Place one actual PeMS region in the expected data layout, run the loader, verify tensor orientation and adjacency semantics, then run a one-epoch smoke experiment before the full LORO grid.
