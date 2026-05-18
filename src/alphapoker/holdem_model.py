"""Neural policy model for fixed-limit Hold'em."""

from __future__ import annotations

import torch
import torch.nn as nn


class HoldemPolicyNet(nn.Module):
    def __init__(self, input_dim: int = 117, hidden_dim: int = 256, n_actions: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

