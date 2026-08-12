from __future__ import annotations

import torch


def adjacency_to_edge_index(
    adjacency: torch.Tensor, threshold: float = 0.0, include_self: bool = False
) -> torch.Tensor:
    """Convert dense adjacency to ``(target, source)`` edge indices.

    The convention throughout the project is ``adjacency[target, source] > 0``
    when the source is allowed to send information to the target.
    """
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    mask = adjacency > threshold
    if not include_self:
        mask.fill_diagonal_(False)
    return mask.nonzero(as_tuple=False).t().contiguous()


def k_hop_edge_index(
    adjacency: torch.Tensor,
    hops: int = 1,
    threshold: float = 0.0,
    include_self: bool = False,
) -> torch.Tensor:
    """Build a sparse candidate support up to ``hops`` graph steps."""
    if hops < 1:
        raise ValueError("hops must be >= 1")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")

    base = (adjacency > threshold).to(torch.float32)
    reach = base > 0
    frontier = base.clone()
    for _ in range(2, hops + 1):
        frontier = frontier @ base
        reach |= frontier > 0

    if include_self:
        reach.fill_diagonal_(True)
    else:
        reach.fill_diagonal_(False)
    return reach.nonzero(as_tuple=False).t().contiguous()
