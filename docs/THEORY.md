# Theory notes

Working paper title:

> **From Network Morphology to Propagation Dynamics: Cross-Network Traffic Forecasting under Distribution Shift**

## Central hypothesis

Network morphology and predictive dynamics are not equivalent. The empirical claim to test is that a propagation-dynamics representation yields a more stable conditional forecasting mechanism across held-out traffic networks than a morphology-bound representation.

## Simplified transport motivation

For

`u_t + v_c(x) u_x = 0`,

the transformation

`tau_c(x) = integral dx / v_c(x)`

reduces the system to

`u_t + u_tau = 0`.

This is a motivating sufficient case, not a claim that PeMS traffic follows linear advection exactly.

## Validation criteria

A useful representation should be both:

- **predictively sufficient**: adding raw topology/history should not recover a large amount of missing forecasting information;
- **conditionally stable**: region identity should add little forecasting power after conditioning on the learned propagation representation.

Failure of these diagnostics should weaken the theory rather than trigger unrelated module stacking.
