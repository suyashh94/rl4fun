from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor


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


@dataclass
class StepStats(TrajStats):
    gt_mean: float
    gt_std: float
    gt_min: float
    gt_max: float


def collect_episodes_steps(
    env: gym.Env,
    policy: torch.nn.Module,
    episodes: int,
    device: torch.device,
) -> Tuple[List[List[Tensor]], List[List[float]], List[List[Sequence[float]]], StepStats]:
    """Collect episodes returning per-step logps (tensors), rewards, and observations.

    Returns
      - all_logps: list over episodes of list[Tensor] per step
      - all_rewards: list over episodes of list[float] per step
      - all_obs: list over episodes of list[np.ndarray] per step
      - stats: includes per-episode returns/lengths and policy stats
    """
    policy.eval()
    all_logps: List[List[Tensor]] = []
    all_rewards: List[List[float]] = []
    all_obs: List[List[Sequence[float]]] = []
    returns: List[float] = []
    lengths: List[int] = []
    entropies: List[float] = []
    logps_all: List[float] = []

    for _ in range(episodes):
        obs, _ = env.reset()
        term = False
        trunc = False
        ep_logps: List[Tensor] = []
        ep_rewards: List[float] = []
        ep_obs: List[Sequence[float]] = []
        steps = 0
        while not (term or trunc):
            ep_obs.append(np.asarray(obs, dtype=np.float32))
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            dist = policy(obs_t)
            action = dist.sample()
            logp = dist.log_prob(action).squeeze()
            entropy = dist.entropy().squeeze()

            next_obs, reward, term, trunc, _ = env.step(action.item())

            ep_logps.append(logp)
            ep_rewards.append(float(reward))
            entropies.append(float(entropy.item()))
            logps_all.append(float(logp.item()))
            steps += 1
            obs = next_obs

        # Episode stats: use undiscounted episodic return for logging (common in CartPole)
        returns.append(float(sum(ep_rewards)))
        lengths.append(steps)
        all_logps.append(ep_logps)
        all_rewards.append(ep_rewards)
        all_obs.append(ep_obs)

    # Rough per-step reward stats (not RTG); RTG stats computed in the loss path
    flat_rewards = [r for ep in all_rewards for r in ep]
    gt_mean = float(np.mean(flat_rewards)) if flat_rewards else 0.0
    gt_std = float(np.std(flat_rewards)) if flat_rewards else 0.0
    gt_min = float(np.min(flat_rewards)) if flat_rewards else 0.0
    gt_max = float(np.max(flat_rewards)) if flat_rewards else 0.0

    stats = StepStats(
        returns=returns,
        lengths=lengths,
        avg_entropy=float(np.mean(entropies)) if entropies else 0.0,
        avg_logp=float(np.mean(logps_all)) if logps_all else 0.0,
        gt_mean=gt_mean,
        gt_std=gt_std,
        gt_min=gt_min,
        gt_max=gt_max,
    )
    return all_logps, all_rewards, all_obs, stats


def discounted_reward_to_go(rewards: Sequence[float], gamma: float) -> List[float]:
    G = 0.0
    rtg: List[float] = [0.0] * len(rewards)
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        rtg[t] = G
    return rtg


def actor_loss_rtg(
    all_logps: List[List[Tensor]],
    all_rewards: List[List[float]],
    gamma: float,
    device: torch.device | None = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """Reward-to-go loss: -(mean over all steps of logpi_t * G_t)."""
    if device is None:
        from reinforce.utils.device import get_torch_device
        device = get_torch_device()

    terms: List[Tensor] = []
    gt_all: List[float] = []
    for ep_logps, ep_rewards in zip(all_logps, all_rewards):
        rtg = discounted_reward_to_go(ep_rewards, gamma)
        gt_all.extend(rtg)

        # Accumulate per-step terms; detach RTG so no grad flows into rewards
        G_t = torch.as_tensor(rtg, dtype=torch.float32, device=device)
        logps_t = torch.stack(ep_logps).to(device)
        terms.append(-(logps_t * G_t).mean())

    loss = torch.stack(terms).mean() if terms else torch.zeros((), dtype=torch.float32, device=device)
    diag = {
        "gt_mean": float(np.mean(gt_all)) if gt_all else 0.0,
        "gt_std": float(np.std(gt_all)) if gt_all else 0.0,
        "gt_min": float(np.min(gt_all)) if gt_all else 0.0,
        "gt_max": float(np.max(gt_all)) if gt_all else 0.0,
    }
    return loss, diag


def actor_critic_loss(
    all_logps: List[List[Tensor]],
    all_rewards: List[List[float]],
    all_obs: List[List[Sequence[float]]],
    gamma: float,
    value_net: torch.nn.Module,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Dict[str, float]]:
    """Compute actor/critic losses with a learned baseline."""
    actor_terms: List[Tensor] = []
    critic_terms: List[Tensor] = []
    value_maes: List[Tensor] = []
    adv_all: List[Tensor] = []
    gt_all: List[float] = []

    for ep_logps, ep_rewards, ep_obs in zip(all_logps, all_rewards, all_obs):
        if not ep_rewards:
            continue
        rtg = discounted_reward_to_go(ep_rewards, gamma)
        gt_all.extend(rtg)

        obs_t = torch.as_tensor(np.asarray(ep_obs, dtype=np.float32), device=device)
        returns_t = torch.as_tensor(rtg, dtype=torch.float32, device=device)
        values = value_net(obs_t).squeeze(-1)

        advantage = (returns_t - values.detach())
        logps_t = torch.stack(ep_logps).to(device)
        actor_terms.append(-(logps_t * advantage).mean())

        critic_error = returns_t - values
        critic_terms.append(0.5 * (critic_error.pow(2)).mean())
        value_maes.append(critic_error.abs().mean())
        adv_all.append(advantage)

    device_tensor = torch.zeros((), dtype=torch.float32, device=device)
    actor_loss = torch.stack(actor_terms).mean() if actor_terms else device_tensor
    critic_loss = torch.stack(critic_terms).mean() if critic_terms else device_tensor
    value_mae = torch.stack(value_maes).mean() if value_maes else device_tensor
    adv_cat = torch.cat(adv_all) if adv_all else device_tensor

    diag = {
        "critic_loss": float(critic_loss.detach().cpu().item()) if critic_terms else 0.0,
        "value_mae": float(value_mae.detach().cpu().item()) if value_maes else 0.0,
        "adv_mean": float(adv_cat.detach().cpu().mean().item()) if adv_all else 0.0,
        "adv_std": float(adv_cat.detach().cpu().std(unbiased=False).item()) if adv_all else 0.0,
        "gt_mean": float(np.mean(gt_all)) if gt_all else 0.0,
        "gt_std": float(np.std(gt_all)) if gt_all else 0.0,
        "gt_min": float(np.min(gt_all)) if gt_all else 0.0,
        "gt_max": float(np.max(gt_all)) if gt_all else 0.0,
    }

    return actor_loss, critic_loss, diag
