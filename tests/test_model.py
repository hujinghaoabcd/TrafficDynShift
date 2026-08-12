import torch

from trafficdynshift import ModelConfig, PDRForecaster
from trafficdynshift.graph import k_hop_edge_index


def chain_adjacency(n: int) -> torch.Tensor:
    a = torch.zeros(n, n)
    for i in range(n - 1):
        a[i + 1, i] = 1.0
    return a


def test_forward_sparse_shapes() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(num_lags=4, num_slots=5, value_dim=16)
    model = PDRForecaster(cfg)
    n = 7
    x = torch.randn(3, cfg.in_len, n, cfg.in_channels)
    edge_index = k_hop_edge_index(chain_adjacency(n), hops=2)
    out = model(x, edge_index)
    assert out["prediction"].shape == (3, cfg.out_len, n)
    assert out["response_probs"].shape == (3, edge_index.shape[1], cfg.num_lags)
    assert out["representation"].shape == (3, n, cfg.num_slots, cfg.value_dim)
    assert torch.allclose(
        out["response_probs"].sum(dim=-1),
        torch.ones(3, edge_index.shape[1]),
    )


def test_same_model_accepts_different_node_counts() -> None:
    torch.manual_seed(1)
    cfg = ModelConfig(num_lags=3, num_slots=4)
    model = PDRForecaster(cfg)
    for n in (5, 11):
        x = torch.randn(2, cfg.in_len, n, cfg.in_channels)
        edges = k_hop_edge_index(chain_adjacency(n), hops=2)
        assert model(x, edges)["prediction"].shape == (2, cfg.out_len, n)


def test_losses_backpropagate() -> None:
    torch.manual_seed(2)
    cfg = ModelConfig(num_lags=4, num_slots=4)
    model = PDRForecaster(cfg)
    n = 6
    x = torch.randn(2, cfg.in_len, n, cfg.in_channels)
    y = torch.randn(2, cfg.out_len, n)
    edges = k_hop_edge_index(chain_adjacency(n), hops=1)
    losses = model.losses(model(x, edges), y)
    losses.total.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
