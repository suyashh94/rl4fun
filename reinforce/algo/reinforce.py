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
) -> Tuple[List[torch.Tensor], List[float], List[List[Tensor]], TrajStats]:
    """
    Phase 1 variant: compute per-episode sum of log-probs (as Tensor with grad)
    and total (discounted) return.
    Returns
      - sum_logps: list[Tensor] length=episodes (each requires grad)
      - returns: list[float] length=episodes
      - entropies: list over episodes of list[Tensor] per step (with grad)
      - stats: TrajStats with averages for logging
    """
    policy.eval()
    sum_logps: List[torch.Tensor] = []
    returns: List[float] = []
    all_entropies: List[List[Tensor]] = []

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
        ep_entropies: List[Tensor] = []
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
            ep_entropies.append(entropy.squeeze())
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
        all_entropies.append(ep_entropies)

    stats = TrajStats(
        returns=returns,
        lengths=lengths,
        avg_entropy=float(np.mean(entropies)) if entropies else 0.0,
        avg_logp=float(np.mean(logps_all)) if logps_all else 0.0,
    )
    return sum_logps, returns, all_entropies, stats


def actor_loss(
    sum_logps: List[torch.Tensor],
    returns: List[float],
    device: torch.device | None = None,
) -> torch.Tensor:
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
) -> Tuple[
    List[List[Tensor]],
    List[List[float]],
    List[List[Sequence[float]]],
    List[List[Tensor]],
    StepStats,
]:
    """Collect episodes returning per-step logps (tensors), rewards, and observations.

    Returns
      - all_logps: list over episodes of list[Tensor] per step
      - all_rewards: list over episodes of list[float] per step
      - all_obs: list over episodes of list[np.ndarray] per step
      - all_entropies: list over episodes of list[Tensor] per step
      - stats: includes per-episode returns/lengths and policy stats
    """
    policy.eval()
    all_logps: List[List[Tensor]] = []
    all_rewards: List[List[float]] = []
    all_obs: List[List[Sequence[float]]] = []
    all_entropies: List[List[Tensor]] = []
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
        ep_entropies: List[Tensor] = []
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
            ep_entropies.append(entropy)
            logps_all.append(float(logp.item()))
            steps += 1
            obs = next_obs

        # Episode stats: use undiscounted episodic return for logging (common in CartPole)
        returns.append(float(sum(ep_rewards)))
        lengths.append(steps)
        all_logps.append(ep_logps)
        all_rewards.append(ep_rewards)
        all_obs.append(ep_obs)
        all_entropies.append(ep_entropies)

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
    return all_logps, all_rewards, all_obs, all_entropies, stats


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
    normalize_adv: bool = False,
) -> Tuple[Tensor, Dict[str, float]]:
    """Reward-to-go loss: -(mean over all steps of logpi_t * G_t)."""
    if device is None:
        from reinforce.utils.device import get_torch_device
        device = get_torch_device()

    terms: List[Tensor] = []
    gt_all: List[float] = []
    rtg_tensors: List[Tensor] = []
    logps_tensors: List[Tensor] = []
    lengths: List[int] = []

    for ep_logps, ep_rewards in zip(all_logps, all_rewards):
        rtg = discounted_reward_to_go(ep_rewards, gamma)
        gt_all.extend(rtg)

        returns_t = torch.as_tensor(rtg, dtype=torch.float32, device=device)
        rtg_tensors.append(returns_t)
        logps_tensors.append(torch.stack(ep_logps).to(device))
        lengths.append(len(rtg))

    cat_rtg = torch.cat(rtg_tensors) if rtg_tensors else torch.tensor([], dtype=torch.float32, device=device)
    cat_for_loss = cat_rtg
    if normalize_adv and rtg_tensors:
        mean = cat_rtg.mean()
        std = cat_rtg.std(unbiased=False)
        cat_for_loss = (cat_rtg - mean) / (std + 1e-8)
        splits = torch.split(cat_for_loss, lengths)  # type: ignore[arg-type]
        rtg_tensors = list(splits)

    for logps_t, returns_t in zip(logps_tensors, rtg_tensors):
        terms.append(-(logps_t * returns_t).mean())

    loss = torch.stack(terms).mean() if terms else torch.zeros((), dtype=torch.float32, device=device)
    diag = {
        "gt_mean": float(np.mean(gt_all)) if gt_all else 0.0,
        "gt_std": float(np.std(gt_all)) if gt_all else 0.0,
        "gt_min": float(np.min(gt_all)) if gt_all else 0.0,
        "gt_max": float(np.max(gt_all)) if gt_all else 0.0,
        "adv_mean": float(cat_for_loss.detach().cpu().mean().item()) if cat_for_loss.numel() > 0 else 0.0,
        "adv_std": float(cat_for_loss.detach().cpu().std(unbiased=False).item()) if cat_for_loss.numel() > 0 else 0.0,
    }
    return loss, diag


def actor_critic_loss(
    all_logps: List[List[Tensor]],
    all_rewards: List[List[float]],
    all_obs: List[List[Sequence[float]]],
    gamma: float,
    value_net: torch.nn.Module,
    device: torch.device,
    normalize_adv: bool = False,
) -> Tuple[Tensor, Tensor, Dict[str, float]]:
    """Compute actor/critic losses with a learned baseline."""
    critic_terms: List[Tensor] = []
    value_maes: List[Tensor] = []
    advantages: List[Tensor] = []
    logps_tensors: List[Tensor] = []
    lengths: List[int] = []
    gt_all: List[float] = []

    for ep_logps, ep_rewards, ep_obs in zip(all_logps, all_rewards, all_obs):
        if not ep_rewards:
            continue
        rtg = discounted_reward_to_go(ep_rewards, gamma)
        gt_all.extend(rtg)

        obs_t = torch.as_tensor(np.asarray(ep_obs, dtype=np.float32), device=device)
        returns_t = torch.as_tensor(rtg, dtype=torch.float32, device=device)
        values = value_net(obs_t).squeeze(-1)

        advantage = returns_t - values.detach()
        logps_tensors.append(torch.stack(ep_logps).to(device))
        advantages.append(advantage)
        lengths.append(len(rtg))

        critic_error = returns_t - values
        critic_terms.append(0.5 * (critic_error.pow(2)).mean())
        value_maes.append(critic_error.abs().mean())

    device_tensor = torch.zeros((), dtype=torch.float32, device=device)
    critic_loss = torch.stack(critic_terms).mean() if critic_terms else device_tensor
    value_mae = torch.stack(value_maes).mean() if value_maes else device_tensor

    adv_cat_raw = torch.cat(advantages) if advantages else torch.tensor([], dtype=torch.float32, device=device)
    adv_for_loss = adv_cat_raw
    if normalize_adv and advantages:
        mean = adv_cat_raw.mean()
        std = adv_cat_raw.std(unbiased=False)
        adv_for_loss = (adv_cat_raw - mean) / (std + 1e-8)
        splits = torch.split(adv_for_loss, lengths)  # type: ignore[arg-type]
        advantages = [split for split in splits]
    elif advantages:
        splits = torch.split(adv_for_loss, lengths)  # type: ignore[arg-type]
        advantages = list(splits)

    actor_terms = [-(logps_t * adv).mean() for logps_t, adv in zip(logps_tensors, advantages)] if advantages else []
    actor_loss = torch.stack(actor_terms).mean() if actor_terms else device_tensor

    diag = {
        "critic_loss": float(critic_loss.detach().cpu().item()) if critic_terms else 0.0,
        "value_mae": float(value_mae.detach().cpu().item()) if value_maes else 0.0,
        "adv_mean": float(adv_for_loss.detach().cpu().mean().item()) if adv_for_loss.numel() else 0.0,
        "adv_std": float(adv_for_loss.detach().cpu().std(unbiased=False).item()) if adv_for_loss.numel() else 0.0,
        "gt_mean": float(np.mean(gt_all)) if gt_all else 0.0,
        "gt_std": float(np.std(gt_all)) if gt_all else 0.0,
        "gt_min": float(np.min(gt_all)) if gt_all else 0.0,
        "gt_max": float(np.max(gt_all)) if gt_all else 0.0,
    }

    return actor_loss, critic_loss, diag
