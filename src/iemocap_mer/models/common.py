from __future__ import annotations

import torch
import torch.nn as nn
import math


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class SequenceProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.network(values)
        positions = torch.arange(values.shape[1], device=values.device, dtype=values.dtype)
        frequencies = torch.exp(
            torch.arange(0, values.shape[2], 2, device=values.device, dtype=values.dtype)
            * (-math.log(10_000.0) / values.shape[2])
        )
        angles = positions[:, None] * frequencies[None, :]
        encoding = torch.zeros(values.shape[1], values.shape[2], device=values.device, dtype=values.dtype)
        encoding[:, 0::2] = torch.sin(angles)
        if values.shape[2] > 1:
            encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
        return values + encoding.unsqueeze(0)


class PredictionHeads(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.emotion = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.va = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2), nn.Tanh())

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.emotion(pooled), "va": self.va(pooled)}
