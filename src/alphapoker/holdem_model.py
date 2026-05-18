"""Neural policy model for fixed-limit Hold'em."""

from __future__ import annotations

import torch
import torch.nn as nn

from alphapoker.holdem_features import HOLDEM_FEATURE_DIM


class HoldemPolicyNet(nn.Module):
    def __init__(
        self,
        input_dim: int = HOLDEM_FEATURE_DIM,
        hidden_dim: int = 256,
        n_actions: int = 5,
    ) -> None:
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


class HoldemEquityNet(nn.Module):
    def __init__(self, input_dim: int = HOLDEM_FEATURE_DIM, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
