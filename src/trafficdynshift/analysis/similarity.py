from __future__ import annotations

import numpy as np


def morphology_descriptors(adjacency: np.ndarray) -> dict[str, float]:
    """Small, transparent descriptor set for morphology diagnostics."""
    a = np.asarray(adjacency) > 0
    n = a.shape[0]
    degree = a.sum(axis=1).astype(np.float64)
    return {
        "num_nodes": float(n),
        "edge_density": float(a.sum() / max(n * (n - 1), 1)),
        "mean_degree": float(degree.mean()),
        "std_degree": float(degree.std()),
        "max_degree": float(degree.max(initial=0.0)),
    }


def jensen_shannon(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))
