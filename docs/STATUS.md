# Status

## Implemented

- [x] project scaffold
- [x] Hydra parameter management
- [x] sparse candidate-edge propagation response
- [x] fixed-slot propagation-dynamics representation
- [x] graph-size-independent shared forecaster
- [x] propagation residual auxiliary objective
- [x] causal input-window normalization
- [x] generic PeMS data/adjacency loaders
- [x] strict source-validation / held-out-target training skeleton
- [x] tests and CI configuration
- [x] morphology / JS diagnostic helpers

## Not yet validated with real paper data

- [ ] exact PeMSD3/4/7/8 local file formats
- [ ] directed adjacency convention for the selected preprocessing files
- [ ] full four-target x five-seed experiment grid
- [ ] baseline implementations
- [ ] morphology-controlled subgraph construction
- [ ] conditional-region and sufficiency probes
- [ ] statistical significance reporting

## Rule

Do not add OOD alignment, contrastive, MoE, Mamba, causal, or other modules unless a specific failed hypothesis/experiment requires them.
