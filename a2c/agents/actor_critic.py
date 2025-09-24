from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCriticNet(nn.Module):
    """Shared MLP backbone with separate policy and value heads."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        features = self.backbone(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        dist = Categorical(logits=logits)
        return dist, value

    @property
    def policy_parameters(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters()) + list(self.policy_head.parameters())

    @property
    def value_parameters(self) -> list[nn.Parameter]:
        return list(self.value_head.parameters())
