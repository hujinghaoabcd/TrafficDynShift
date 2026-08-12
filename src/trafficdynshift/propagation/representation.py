from __future__ import annotations

import torch
from torch import nn

from trafficdynshift.config_schema import ModelConfig


class PropagationDynamicsRepresentation(nn.Module):
    """Reorganize irregular graph neighborhoods into shared lag-response slots."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.value_encoder = nn.Sequential(
            nn.Linear(cfg.in_channels, cfg.value_dim),
            nn.GELU(),
            nn.Linear(cfg.value_dim, cfg.value_dim),
        )

        lag_positions = torch.linspace(1.0 / cfg.num_lags, 1.0, cfg.num_lags)
        slot_centers = torch.linspace(0.0, 1.0, cfg.num_slots)
        basis = torch.exp(
            -0.5
            * ((lag_positions[:, None] - slot_centers[None, :]) / cfg.basis_sigma) ** 2
        )
        basis = basis / basis.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("basis", basis)

    def forward(
        self,
        x: torch.Tensor,
        response_probs: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``(B,N,K,D)`` while preserving the original node count ``N``."""
        b, _, n, _ = x.shape
        e = edge_index.shape[1]
        if response_probs.shape != (b, e, self.cfg.num_lags):
            raise ValueError(
                "response_probs must have shape "
                f"{(b, e, self.cfg.num_lags)}, got {tuple(response_probs.shape)}"
            )
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        target, source = edge_index[0], edge_index[1]

        lag_values = torch.stack(
            [x[:, -lag, :, :] for lag in range(1, self.cfg.num_lags + 1)], dim=1
        )
        values = self.value_encoder(lag_values)
        edge_values = values[:, :, source, :].permute(0, 2, 1, 3)

        weights = response_probs[..., None] * self.basis[None, None, :, :]
        edge_num = torch.einsum("belk,beld->bekd", weights, edge_values)
        edge_den = weights.sum(dim=2)

        z_num = x.new_zeros((b, n, self.cfg.num_slots, self.cfg.value_dim))
        z_den = x.new_zeros((b, n, self.cfg.num_slots))
        for batch_idx in range(b):
            z_num[batch_idx].index_add_(0, target, edge_num[batch_idx])
            z_den[batch_idx].index_add_(0, target, edge_den[batch_idx])
        return z_num / z_den[..., None].clamp_min(1e-8)
