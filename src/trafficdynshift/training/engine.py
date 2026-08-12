from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from trafficdynshift.models import PDRForecaster
from trafficdynshift.training.metrics import mae, mape, rmse
from trafficdynshift.utils.normalization import denormalize_forecast, normalize_forecast_pair


@dataclass
class EvaluationResult:
    mae: float
    rmse: float
    mape: float


def train_one_region(
    model: PDRForecaster,
    loader: DataLoader,
    edge_index: torch.Tensor,
    optimizer: Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
) -> float:
    model.train()
    total = 0.0
    count = 0
    edge_index = edge_index.to(device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        xn, yn, _, _ = normalize_forecast_pair(x, y)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(xn, edge_index)
        losses = model.losses(outputs, yn)
        losses.total.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(losses.total.detach()) * x.shape[0]
        count += x.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def evaluate(
    model: PDRForecaster,
    loader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
) -> EvaluationResult:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    edge_index = edge_index.to(device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        xn, _, mean, std = normalize_forecast_pair(x, y)
        pred_n = model(xn, edge_index)["prediction"]
        pred = denormalize_forecast(pred_n, mean, std)
        preds.append(pred.cpu())
        targets.append(y.cpu())
    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    return EvaluationResult(
        mae=float(mae(pred_all, target_all)),
        rmse=float(rmse(pred_all, target_all)),
        mape=float(mape(pred_all, target_all)),
    )


def save_checkpoint(model: PDRForecaster, path: str | Path, epoch: int, metric: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch, "metric": metric}, path)


def mean_region_mae(results: Iterable[EvaluationResult]) -> float:
    values = [r.mae for r in results]
    if not values:
        raise ValueError("No region results supplied")
    return sum(values) / len(values)
