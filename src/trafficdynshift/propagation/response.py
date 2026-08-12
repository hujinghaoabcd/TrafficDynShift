from __future__ import annotations

import torch
from torch import nn

from trafficdynshift.config_schema import ModelConfig
from trafficdynshift.utils.normalization import causal_window_zscore


class PropagationResponseEstimator(nn.Module):
    """Estimate lag-dependent predictive responses on candidate graph edges.

    The module is node-count agnostic and scores only sparse candidate edges,
    avoiding an ``N x N`` pair tensor for large PeMS graphs.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        c = cfg.in_channels
        d = cfg.response_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(cfg.in_len * c, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, d),
        )
        self.source_lag_encoder = nn.Sequential(
            nn.Linear(c, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.pair_scorer = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return lag probabilities shaped ``(B,E,L)``.

        ``edge_index[0]`` contains target nodes and ``edge_index[1]`` sources.
        """
        if x.ndim != 4:
            raise ValueError("x must have shape (B,H,N,C)")
        b, h, n, c = x.shape
        if h != self.cfg.in_len or c != self.cfg.in_channels:
            raise ValueError(
                f"Expected H={self.cfg.in_len}, C={self.cfg.in_channels}; got H={h}, C={c}"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2,E)")
        if edge_index.numel() and int(edge_index.max()) >= n:
            raise ValueError("edge_index contains a node id outside x")

        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        target, source = edge_index[0], edge_index[1]
        xn = causal_window_zscore(x)
        node_hist = xn.permute(0, 2, 1, 3).reshape(b, n, h * c)
        node_emb = self.node_encoder(node_hist)

        target_emb = node_emb[:, target, :]
        source_emb = node_emb[:, source, :]
        logits: list[torch.Tensor] = []
        for lag in range(1, self.cfg.num_lags + 1):
            source_value = xn[:, -lag, source, :]
            source_lag_emb = self.source_lag_encoder(source_value)
            pair = torch.cat(
                [target_emb, source_emb, target_emb * source_emb, source_lag_emb], dim=-1
            )
            logits.append(self.pair_scorer(pair).squeeze(-1))
        scores = torch.stack(logits, dim=-1)
        return torch.softmax(scores / self.cfg.temperature, dim=-1)
