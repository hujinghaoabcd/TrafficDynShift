from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TrafficWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding windows from a single traffic region.

    Input shape per sample is ``(H,N,1)`` and target shape is ``(P,N)``.
    """

    def __init__(self, flow: np.ndarray, in_len: int, out_len: int) -> None:
        if flow.ndim != 2:
            raise ValueError("flow must have shape (T,N)")
        if len(flow) < in_len + out_len:
            raise ValueError("time series is shorter than one input/output window")
        self.flow = torch.as_tensor(flow, dtype=torch.float32)
        self.in_len = in_len
        self.out_len = out_len

    def __len__(self) -> int:
        return self.flow.shape[0] - self.in_len - self.out_len + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.flow[index : index + self.in_len].unsqueeze(-1)
        y = self.flow[index + self.in_len : index + self.in_len + self.out_len]
        return x, y


@dataclass
class RegionData:
    name: str
    adjacency: torch.Tensor
    train: TrafficWindowDataset
    val: TrafficWindowDataset
    test: TrafficWindowDataset


def load_flow(path: str | Path, feature_index: int = 0) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        archive = np.load(path)
        if "data" in archive:
            arr = archive["data"]
        else:
            arr = archive[archive.files[0]]
    else:
        raise ValueError(f"Unsupported flow file: {path}")

    if arr.ndim == 3:
        arr = arr[..., feature_index]
    if arr.ndim != 2:
        raise ValueError(f"Expected traffic data shaped (T,N) or (T,N,F), got {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def load_adjacency(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        archive = np.load(path)
        arr = archive["adjacency"] if "adjacency" in archive else archive[archive.files[0]]
    elif path.suffix == ".csv":
        arr = pd.read_csv(path, header=None).to_numpy()
    else:
        raise ValueError(f"Unsupported adjacency file: {path}")
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Adjacency must be square, got {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def chronological_split(
    flow: np.ndarray, train_ratio: float, val_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Require 0 < train_ratio < 1 and train_ratio + val_ratio < 1")
    t = flow.shape[0]
    train_end = int(t * train_ratio)
    val_end = int(t * (train_ratio + val_ratio))
    return flow[:train_end], flow[train_end:val_end], flow[val_end:]


def load_region(
    name: str,
    data_path: str | Path,
    adjacency_path: str | Path,
    in_len: int,
    out_len: int,
    feature_index: int,
    train_ratio: float,
    val_ratio: float,
) -> RegionData:
    flow = load_flow(data_path, feature_index=feature_index)
    adjacency_np = load_adjacency(adjacency_path)
    if adjacency_np.shape[0] != flow.shape[1]:
        raise ValueError(
            f"{name}: adjacency nodes={adjacency_np.shape[0]} but flow nodes={flow.shape[1]}"
        )
    train, val, test = chronological_split(flow, train_ratio, val_ratio)
    return RegionData(
        name=name,
        adjacency=torch.from_numpy(adjacency_np),
        train=TrafficWindowDataset(train, in_len, out_len),
        val=TrafficWindowDataset(val, in_len, out_len),
        test=TrafficWindowDataset(test, in_len, out_len),
    )


def make_loader(
    dataset: TrafficWindowDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
