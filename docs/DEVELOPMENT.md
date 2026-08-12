# Development Log

## 2026-08-13 — v0.1 scaffold

- initialized modern `src/` package layout;
- added Hydra grouped configuration;
- added sparse propagation-response estimator;
- added propagation-dynamics representation and shared forecast head;
- added causal-window normalization for strict zero-shot use;
- added generic PeMS NPZ/adjacency loaders;
- added leave-one-region-out training entry point;
- added pytest, Ruff, Black, mypy, pre-commit and GitHub Actions;
- documented theory, protocol, status and handoff notes.

Next development should validate the loader against the exact local PeMS files before adding more model complexity.
