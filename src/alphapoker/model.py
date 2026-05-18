"""Small policy/value model for Kuhn poker strategy distillation."""

from __future__ import annotations

import torch
import torch.nn as nn


class KuhnPolicyValueNet(nn.Module):
    """MLP with fixed action logits and a scalar value head."""

    def __init__(self, input_dim: int = 9, hidden_dim: int = 64, n_actions: int = 4) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(features)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

