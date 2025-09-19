from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch


@dataclass
class TrajStats:
    returns: List[float]
    lengths: List[int]
    avg_entropy: float
    avg_logp: float


def collect_episodes(
    env: gym.Env,
    policy: torch.nn.Module,
    episodes: int,
    device: torch.device,
    gamma: float,
) -> Tuple[List[torch.Tensor], List[float], TrajStats]:
    """
    Phase 1 variant: compute per-episode sum of log-probs (as Tensor with grad)
    and total (discounted) return.
    Returns
      - sum_logps: list[Tensor] length=episodes (each requires grad)
      - returns: list[float] length=episodes
      - stats: TrajStats with averages for logging
    """
    policy.eval()
    sum_logps: List[torch.Tensor] = []
    returns: List[float] = []

    entropies: List[float] = []
    logps_all: List[float] = []
    lengths: List[int] = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        term = False
        trunc = False
        # Differentiable accumulator for log-probs over the episode
        ep_logp_sum = torch.zeros((), dtype=torch.float32, device=device)
        ep_rewards: List[float] = []
        steps = 0
        while not (term or trunc):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            # No no_grad: we need gradient through policy logits
            dist = policy(obs_t)
            action = dist.sample()
            logp = dist.log_prob(action)
            entropy = dist.entropy()

            next_obs, reward, term, trunc, _ = env.step(action.item())

            ep_logp_sum = ep_logp_sum + logp.squeeze()
            ep_rewards.append(float(reward))
            entropies.append(float(entropy.item()))
            logps_all.append(float(logp.item()))
            steps += 1
            obs = next_obs

        # Discounted total return for the whole episode
        G = 0.0
        for r in reversed(ep_rewards):
            G = r + gamma * G
        returns.append(G)
        sum_logps.append(ep_logp_sum)
        lengths.append(steps)

    stats = TrajStats(
        returns=returns,
        lengths=lengths,
        avg_entropy=float(np.mean(entropies)) if entropies else 0.0,
        avg_logp=float(np.mean(logps_all)) if logps_all else 0.0,
    )
    return sum_logps, returns, stats


def actor_loss(sum_logps: List[torch.Tensor], returns: List[float], device: torch.device | None = None) -> torch.Tensor:
    """Phase 1 actor loss: - E[ sum_t log pi(a|s) * R_total ]"""
    if device is None:
        from reinforce.utils.device import get_torch_device
        device = get_torch_device()
    # Stack differentiable per-episode sums of log-probs
    sum_logps_t = torch.stack(sum_logps).to(device)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
    # Mean over episodes
    loss = -(sum_logps_t * returns_t).mean()
    return loss
