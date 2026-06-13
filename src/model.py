from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PIQCrossfireNet(nn.Module):
    """Multi-task text model for incident evidence analysis."""

    def __init__(
        self,
        input_dim: int,
        quality_count: int,
        hidden_dim: int = 160,
        status_count: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
        )
        self.quality_head = nn.Linear(hidden_dim // 2, quality_count)
        self.strength_head = nn.Linear(hidden_dim // 2, 1)
        self.status_head = nn.Linear(hidden_dim // 2, status_count)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        quality_logits = self.quality_head(encoded)
        strength = torch.sigmoid(self.strength_head(encoded)).squeeze(-1)
        status_logits = self.status_head(encoded)
        return quality_logits, strength, status_logits

