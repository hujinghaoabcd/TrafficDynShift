# Design

## Research principle

TrafficDynShift tests one central claim: cross-network transferability may be better characterized by traffic propagation dynamics than by network morphology alone.

The implementation is intentionally split into:

1. **topological support** — restricts which interactions are plausible;
2. **propagation-response estimation** — learns lag-dependent predictive responses;
3. **propagation-dynamics representation** — maps variable neighborhoods into shared slots;
4. **shared forecasting rule** — contains no target-network-specific node parameters.

## Non-goals

The learned response is not claimed to be physical travel time, shock-wave speed, or a strict causal effect. The v0.1 model is also not tied to STGNNs; the simple forecast head is deliberately replaceable.

## Sparse edge design

The prototype initially used dense `(N,N,L)` pair tensors. The repository implementation scores only candidate edges `(E,L)`, which is materially safer for PeMSD7 (`N=883`).

## Parameter management

Hydra config groups are the single source of truth. Python dataclasses provide typed model construction, but experiment parameters must not be duplicated in scripts. Every run records Hydra's config/overrides plus a fully resolved configuration.
