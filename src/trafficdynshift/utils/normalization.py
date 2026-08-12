from __future__ import annotations

import torch


def causal_window_stats(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample, per-node stats using only the observed input window.

    Parameters
    ----------
    x:
        Tensor shaped ``(B, H, N, C)``.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x with shape (B,H,N,C), got {tuple(x.shape)}")
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return mean, std


def causal_window_zscore(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean, std = causal_window_stats(x, eps=eps)
    return (x - mean) / std


def normalize_forecast_pair(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize input and target with statistics computed from the input only.

    ``x``: ``(B,H,N,C)``; ``y``: ``(B,P,N)`` for the flow-only setting.
    """
    mean, std = causal_window_stats(x, eps=eps)
    xn = (x - mean) / std
    node_mean = mean[:, 0, :, 0]
    node_std = std[:, 0, :, 0]
    yn = (y - node_mean[:, None, :]) / node_std[:, None, :]
    return xn, yn, node_mean, node_std


def denormalize_forecast(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return y * std[:, None, :] + mean[:, None, :]
