from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Typed model configuration resolved from Hydra YAML."""

    in_len: int = 12
    out_len: int = 12
    in_channels: int = 1
    hidden_dim: int = 64
    response_dim: int = 32
    value_dim: int = 32
    num_lags: int = 6
    num_slots: int = 6
    basis_sigma: float = 0.18
    dropout: float = 0.1
    temperature: float = 1.0
    lambda_prop: float = 0.2
    lambda_self: float = 0.1

    def validate(self) -> None:
        if self.in_len < 1 or self.out_len < 1:
            raise ValueError("in_len and out_len must be positive")
        if not 1 <= self.num_lags <= self.in_len:
            raise ValueError("num_lags must satisfy 1 <= num_lags <= in_len")
        if self.num_slots < 1:
            raise ValueError("num_slots must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.basis_sigma <= 0:
            raise ValueError("basis_sigma must be positive")
