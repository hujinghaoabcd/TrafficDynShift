from __future__ import annotations

import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mask = torch.abs(target) > eps
    if not torch.any(mask):
        return torch.tensor(float("nan"), device=pred.device)
    return torch.mean(torch.abs((pred[mask] - target[mask]) / target[mask])) * 100.0
