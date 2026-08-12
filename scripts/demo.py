from __future__ import annotations

import torch

from trafficdynshift import ModelConfig, PDRForecaster
from trafficdynshift.graph import k_hop_edge_index


def ring_adjacency(n: int) -> torch.Tensor:
    a = torch.zeros(n, n)
    for i in range(n):
        a[i, (i - 1) % n] = 1.0
        a[i, (i + 1) % n] = 1.0
    return a


def run(model: PDRForecaster, n: int) -> None:
    cfg = model.cfg
    x = torch.randn(2, cfg.in_len, n, cfg.in_channels)
    edge_index = k_hop_edge_index(ring_adjacency(n), hops=2)
    out = model(x, edge_index)
    print(
        f"N={n:>3}  E={edge_index.shape[1]:>4}  "
        f"prediction={tuple(out['prediction'].shape)}  "
        f"response={tuple(out['response_probs'].shape)}  "
        f"representation={tuple(out['representation'].shape)}"
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = PDRForecaster(cfg)
    run(model, 8)
    run(model, 13)
