# Experiment Protocol

## Datasets

Only PeMSD3, PeMSD4, PeMSD7, and PeMSD8 are in scope for the paper.

## Main protocol: leave-one-region-out zero-shot

Run four held-out targets:

- D4 + D7 + D8 -> D3
- D3 + D7 + D8 -> D4
- D3 + D4 + D8 -> D7
- D3 + D4 + D7 -> D8

Target data must not be used for training, early stopping, hyperparameter selection, or fine-tuning. The target graph and observed target input window are allowed at inference.

## Seeds

Primary results use at least five seeds: `42, 52, 62, 72, 82`.

## Normalization

Cross-network experiments use causal input-window normalization. Target-dataset training statistics must not be used in strict zero-shot evaluation.

## Metrics

Report MAE, RMSE, MAPE, plus horizon-specific results (15/30/60 min) when the final runner is extended.

## Mechanism diagnostics

1. morphology similarity vs transferability;
2. propagation-response similarity vs transferability;
3. conditional region probe;
4. predictive sufficiency probe;
5. delay-as-weight vs delay-as-representation ablation.

## Controlled spatial OOD

A within-dataset morphology-controlled subgraph protocol will be added after the primary pipeline is validated. Its purpose is to reduce the temporal confounding present across the four PeMS collection periods.
