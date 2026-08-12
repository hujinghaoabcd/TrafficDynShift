from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from trafficdynshift.config_schema import ModelConfig
from trafficdynshift.propagation import (
    PropagationDynamicsRepresentation,
    PropagationResponseEstimator,
)


@dataclass
class PDRLosses:
    total: torch.Tensor
    forecast: torch.Tensor
    propagation: torch.Tensor
    self_baseline: torch.Tensor


class SharedForecastHead(nn.Module):
    """Simple node-wise forecast head independent of graph size."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                cfg.in_len * cfg.in_channels + cfg.num_slots * cfg.value_dim,
                cfg.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.out_len),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        b, h, n, c = x.shape
        hist = x.permute(0, 2, 1, 3).reshape(b, n, h * c)
        prop = z.reshape(b, n, -1)
        return self.net(torch.cat([hist, prop], dim=-1))


class PDRForecaster(nn.Module):
    """Propagation-dynamics forecaster for variable-size traffic graphs."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.response = PropagationResponseEstimator(cfg)
        self.representation = PropagationDynamicsRepresentation(cfg)
        self.forecast_head = SharedForecastHead(cfg)
        hist_dim = cfg.in_len * cfg.in_channels
        prop_dim = cfg.num_slots * cfg.value_dim
        self.self_head = nn.Sequential(
            nn.Linear(hist_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.out_len),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(prop_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.out_len),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        probs = self.response(x, edge_index)
        z = self.representation(x, probs, edge_index)
        pred_bnp = self.forecast_head(x, z)
        b, h, n, c = x.shape
        hist = x.permute(0, 2, 1, 3).reshape(b, n, h * c)
        self_bnp = self.self_head(hist)
        residual_bnp = self.residual_head(z.reshape(b, n, -1))
        return {
            "prediction": pred_bnp.permute(0, 2, 1),
            "response_probs": probs,
            "representation": z,
            "self_prediction": self_bnp.permute(0, 2, 1),
            "residual_prediction": residual_bnp.permute(0, 2, 1),
        }

    def losses(self, outputs: dict[str, torch.Tensor], y: torch.Tensor) -> PDRLosses:
        pred = outputs["prediction"]
        self_pred = outputs["self_prediction"]
        residual_pred = outputs["residual_prediction"]
        if pred.shape != y.shape:
            raise ValueError(f"Expected y shape {tuple(pred.shape)}, got {tuple(y.shape)}")
        forecast = torch.mean(torch.abs(pred - y))
        self_loss = torch.mean(torch.abs(self_pred - y))
        residual_target = (y - self_pred).detach()
        propagation = torch.mean(torch.abs(residual_pred - residual_target))
        total = forecast + self.cfg.lambda_prop * propagation + self.cfg.lambda_self * self_loss
        return PDRLosses(total, forecast, propagation, self_loss)
