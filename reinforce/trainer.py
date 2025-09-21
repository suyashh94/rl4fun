from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from platform import python_version
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from reinforce.agents import CategoricalPolicy, ValueNet
from reinforce.utils.metadata import append_jsonl, get_git_info, get_versions, iso_now, write_json
from reinforce.utils.seeding import set_global_seeds
from reinforce.utils.tb import TbLogger


@dataclass
class TrainerConfig:
    env_id: str = "CartPole-v1"
    seed: int = 1
    hidden: int = 128
    episodes_per_update: int = 10
    max_updates: int = 500
    gamma: float = 0.99
    lr: float = 3e-3
    critic_lr: float = 1e-3
    log_dir: str = "reinforce/experiments/runs"
    tag: Optional[str] = None
    normalize_obs: bool = False
    use_rtg: bool = False
    use_baseline: bool = False
    normalize_adv: bool = False
    entropy_coef: float = 0.0
    grad_clip: float = 0.0


@dataclass
class RolloutBatch:
    observations: Tensor  # [B, T, obs_dim]
    rewards: Tensor  # [B, T]
    log_probs: Tensor  # [B, T]
    entropies: Tensor  # [B, T]
    masks: Tensor  # [B, T]
    lengths: Tensor  # [B]
    rtg: Tensor  # [B, T]
    discounted_returns: Tensor  # [B]
    raw_returns: Tensor  # [B]


@dataclass
class TrainingHistory:
    updates: List[int] = field(default_factory=list)
    return_mean: List[float] = field(default_factory=list)
    return_ma: List[float] = field(default_factory=list)
    avg_entropy: List[float] = field(default_factory=list)
    avg_logp: List[float] = field(default_factory=list)


def make_env(env_id: str, seed: int, normalize_obs: bool) -> gym.Env:
    env = gym.make(env_id)
    env.reset(seed=seed)
    if normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
    return env


def _discounted_to_go(rewards: Sequence[float], gamma: float) -> List[float]:
    total = 0.0
    out: List[float] = [0.0] * len(rewards)
    for idx in reversed(range(len(rewards))):
        total = rewards[idx] + gamma * total
        out[idx] = total
    return out


class ReinforceTrainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = set_global_seeds(cfg.seed)

        self.env, self.obs_dim, self.act_dim = self._setup_environment()

        self.policy = CategoricalPolicy(self.obs_dim, self.act_dim, hidden=cfg.hidden).to(self.device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

        self.value_net: Optional[ValueNet]
        self.value_optim: Optional[torch.optim.Optimizer]
        if cfg.use_baseline:
            self.value_net = ValueNet(self.obs_dim, hidden=cfg.hidden).to(self.device)
            self.value_optim = torch.optim.Adam(self.value_net.parameters(), lr=cfg.critic_lr)
        else:
            self.value_net = None
            self.value_optim = None

        self.history = TrainingHistory()

        self.tag, self.log_path, self.tb = self._setup_logging()

        meta = {
            "created_at": iso_now(),
            "tag": self.tag,
            "seed": int(self.seed),
            "device": str(self.device),
            "env_id": cfg.env_id,
            "config": cfg.__dict__,
            "versions": get_versions(),
            "git": get_git_info(),
            "tb_log_dir": str(self.log_path),
            "python": python_version(),
        }
        write_json(self.log_path / "config.json", meta)
        append_jsonl(Path(cfg.log_dir) / "registry.jsonl", meta)

    def close(self) -> None:
        self.tb.close()
        self.env.close()

    def train(self) -> TrainingHistory:
        ret_ma: Optional[float] = None
        best_ret_mean = -float("inf")

        for update in range(1, self.cfg.max_updates + 1):
            batch = self._collect_rollout()
            metrics = self._update(batch)

            ret_ma, stats = self._handle_logging(batch, metrics, update, ret_ma)
            best_ret_mean = max(best_ret_mean, stats["ret_mean"])

            if update % 10 == 0 or update == 1:
                msg = (
                    f"update {update:4d} | ret_mean {stats['ret_mean']:7.2f} | ret_ma {stats['ret_ma']:7.2f} | "
                    f"policy {metrics['policy_loss']:.4f} | H {stats['avg_entropy']:.3f}"
                )
                if self.value_net is not None and metrics.get("baseline_loss") is not None:
                    msg += f" | baseline {metrics['baseline_loss']:.4f}"
                print(msg)

        summary = {
            "finished_at": iso_now(),
            "best_return_mean": float(best_ret_mean),
            "last_return_ma": float(ret_ma if ret_ma is not None else 0.0),
            "updates": int(self.cfg.max_updates),
        }
        write_json(self.log_path / "summary.json", summary)
        return self.history

    def _setup_environment(self) -> Tuple[gym.Env, int, int]:
        env = make_env(self.cfg.env_id, self.seed, self.cfg.normalize_obs)
        obs_shape = env.observation_space.shape
        if obs_shape is None or len(obs_shape) != 1:
            raise ValueError("ReinforceTrainer only supports flat observation spaces")
        obs_dim = obs_shape[0]
        if not hasattr(env.action_space, "n"):
            raise ValueError("ReinforceTrainer currently supports discrete action spaces")
        act_dim = env.action_space.n
        return env, obs_dim, act_dim

    def _setup_logging(self) -> Tuple[str, Path, TbLogger]:
        base_dir = Path(self.cfg.log_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        tag = self.cfg.tag or time.strftime("%Y%m%d_%H%M%S")
        path = base_dir / tag
        if path.exists() and any(path.iterdir()):
            suffix = time.strftime("%Y%m%d_%H%M%S")
            tag = f"{tag}_{suffix}"
            path = base_dir / tag
        tb = TbLogger(path)
        return tag, path, tb

    def _handle_logging(
        self,
        batch: RolloutBatch,
        metrics: Dict[str, float],
        update: int,
        ret_ma: Optional[float],
    ) -> Tuple[float, Dict[str, float]]:
        stats = self._compute_stats(batch, ret_ma)
        self._update_history(stats, update)
        self._log_scalars(stats, metrics, update)
        return stats["ret_ma"], stats

    def _compute_stats(self, batch: RolloutBatch, ret_ma: Optional[float]) -> Dict[str, float]:
        returns_np = batch.raw_returns.detach().cpu().numpy()
        ret_mean = float(np.mean(returns_np)) if returns_np.size else 0.0
        ret_med = float(np.median(returns_np)) if returns_np.size else 0.0
        ret_max = float(np.max(returns_np)) if returns_np.size else 0.0
        new_ret_ma = ret_mean if ret_ma is None else 0.9 * ret_ma + 0.1 * ret_mean

        total_steps_tensor = batch.masks.sum()
        total_steps = float(total_steps_tensor.detach().cpu().item()) if batch.masks.numel() else 1.0
        avg_entropy = 0.0
        avg_logp = 0.0
        if total_steps > 0:
            avg_entropy = float((batch.entropies * batch.masks).sum().detach().cpu().item() / total_steps)
            avg_logp = float((batch.log_probs * batch.masks).sum().detach().cpu().item() / total_steps)

        return {
            "ret_mean": ret_mean,
            "ret_med": ret_med,
            "ret_max": ret_max,
            "ret_ma": new_ret_ma,
            "total_steps": total_steps,
            "avg_entropy": avg_entropy,
            "avg_logp": avg_logp,
        }

    def _update_history(self, stats: Dict[str, float], update: int) -> None:
        self.history.updates.append(update)
        self.history.return_mean.append(stats["ret_mean"])
        self.history.return_ma.append(stats["ret_ma"])
        self.history.avg_entropy.append(stats["avg_entropy"])
        self.history.avg_logp.append(stats["avg_logp"])

    def _log_scalars(self, stats: Dict[str, float], metrics: Dict[str, float], update: int) -> None:
        self.tb.log_scalar("train/return_mean", stats["ret_mean"], update)
        self.tb.log_scalar("train/return_median", stats["ret_med"], update)
        self.tb.log_scalar("train/return_max", stats["ret_max"], update)
        self.tb.log_scalar("train/return_ma", stats["ret_ma"], update)
        self.tb.log_scalar("train/policy_loss", metrics["policy_loss"], update)
        self.tb.log_scalar("train/policy_grad_norm", metrics["policy_grad_norm"], update)
        self.tb.log_scalar("train/avg_entropy", stats["avg_entropy"], update)
        self.tb.log_scalar("train/avg_logp", stats["avg_logp"], update)
        self.tb.log_scalar("train/steps", stats["total_steps"], update)

        if self.value_net is not None:
            for key in ("baseline_loss", "value_mae", "adv_mean", "adv_std", "baseline_grad_norm"):
                if key in metrics and metrics[key] is not None:
                    self.tb.log_scalar(f"train/{key}", metrics[key], update)

        for key in ("rtg_mean", "rtg_std", "rtg_min", "rtg_max"):
            if key in metrics:
                self.tb.log_scalar(f"train/{key}", metrics[key], update)

        if self.cfg.entropy_coef > 0 and metrics.get("entropy_bonus") is not None:
            self.tb.log_scalar("train/entropy_bonus", metrics["entropy_bonus"], update)

    def _collect_rollout(self) -> RolloutBatch:
        obs_sequences: List[Tensor] = []
        reward_sequences: List[Tensor] = []
        logp_sequences: List[Tensor] = []
        entropy_sequences: List[Tensor] = []
        rtg_sequences: List[Tensor] = []
        discounted_returns: List[float] = []
        raw_returns: List[float] = []

        for _ in range(self.cfg.episodes_per_update):
            obs_list: List[np.ndarray] = []
            rewards_list: List[float] = []
            logps_list: List[Tensor] = []
            entropy_list: List[Tensor] = []

            obs, _ = self.env.reset()
            terminated = False
            truncated = False

            while not (terminated or truncated):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                dist = self.policy(obs_tensor)
                action = dist.sample()
                logp = dist.log_prob(action).squeeze(0)
                entropy = dist.entropy().squeeze(0)

                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())

                obs_list.append(np.asarray(obs, dtype=np.float32))
                rewards_list.append(float(reward))
                logps_list.append(logp)
                entropy_list.append(entropy)

                obs = next_obs

            if not rewards_list:
                continue
            
            raw_returns.append(float(sum(rewards_list)))
            rtg = _discounted_to_go(rewards_list, self.cfg.gamma)
            discounted_returns.append(float(rtg[0]))

            obs_sequences.append(torch.as_tensor(np.asarray(obs_list), dtype=torch.float32, device=self.device))
            reward_sequences.append(torch.as_tensor(np.asarray(rewards_list), dtype=torch.float32, device=self.device))
            logp_sequences.append(torch.stack(logps_list).to(self.device))
            entropy_sequences.append(torch.stack(entropy_list).to(self.device))
            rtg_sequences.append(torch.as_tensor(np.asarray(rtg), dtype=torch.float32, device=self.device))

        if not obs_sequences:
            raise RuntimeError("No episodes collected; check environment setup")

        masks = pad_sequence(
            [torch.ones(seq.size(0), dtype=torch.float32, device=self.device) for seq in reward_sequences],
            batch_first=True,
            padding_value=0.0,
        )

        observations = pad_sequence(obs_sequences, batch_first=True, padding_value=0.0)
        rewards = pad_sequence(reward_sequences, batch_first=True, padding_value=0.0)
        log_probs = pad_sequence(logp_sequences, batch_first=True, padding_value=0.0)
        entropies = pad_sequence(entropy_sequences, batch_first=True, padding_value=0.0)
        rtg = pad_sequence(rtg_sequences, batch_first=True, padding_value=0.0)
        lengths = torch.as_tensor([seq.size(0) for seq in reward_sequences], dtype=torch.long, device=self.device)

        return RolloutBatch(
            observations=observations,
            rewards=rewards,
            log_probs=log_probs,
            entropies=entropies,
            masks=masks,
            lengths=lengths,
            rtg=rtg,
            discounted_returns=torch.as_tensor(discounted_returns, dtype=torch.float32, device=self.device),
            raw_returns=torch.as_tensor(raw_returns, dtype=torch.float32, device=self.device),
        )

    def _update(self, batch: RolloutBatch) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        self.policy.train()

        masks = batch.masks
        total_steps = masks.sum().clamp_min(1.0)

        if self.cfg.use_baseline and self.value_net is not None and self.value_optim is not None:
            assert self.value_net is not None
            self.value_net.train()

            values = self.value_net(batch.observations.view(-1, self.obs_dim)).view_as(batch.rewards)
            advantages = batch.rtg - values.detach()
            advantages = self._maybe_normalise_signal(advantages, masks)

            policy_loss = self._batch_policy_loss(batch.log_probs, advantages, masks, total_steps)
            policy_loss = self._apply_entropy_bonus(policy_loss, batch.entropies, masks, total_steps, metrics)

            baseline_error = (batch.rtg - values) * masks
            baseline_loss = 0.5 * (baseline_error.pow(2).sum() / total_steps)
            value_mae = baseline_error.abs().sum() / total_steps

            policy_grad_norm = self._apply_policy_update(policy_loss)
            baseline_grad_norm = self._apply_baseline_update(baseline_loss)

            adv_mean, adv_std = self._masked_mean_std(advantages, masks)

            metrics.update(
                policy_loss=float(policy_loss.detach().cpu().item()),
                policy_grad_norm=policy_grad_norm,
                baseline_loss=float(baseline_loss.detach().cpu().item()),
                baseline_grad_norm=baseline_grad_norm,
                value_mae=float(value_mae.detach().cpu().item()),
                adv_mean=adv_mean,
                adv_std=adv_std,
            )

            metrics.update(self._rtg_stats(batch.rtg, masks))
            return metrics

        if self.cfg.use_rtg:
            returns = self._maybe_normalise_signal(batch.rtg, masks)
            policy_loss = self._batch_policy_loss(batch.log_probs, returns, masks, total_steps)
            policy_loss = self._apply_entropy_bonus(policy_loss, batch.entropies, masks, total_steps, metrics)

            policy_grad_norm = self._apply_policy_update(policy_loss)

            metrics.update(
                policy_loss=float(policy_loss.detach().cpu().item()),
                policy_grad_norm=policy_grad_norm,
            )

            metrics.update(self._rtg_stats(batch.rtg, masks))

            return metrics

        # Vanilla REINFORCE (Phase 1)
        returns = batch.discounted_returns

        if self.cfg.normalize_adv:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        policy_loss = self._episode_policy_loss(batch.log_probs, masks, returns)
        policy_loss = self._apply_entropy_bonus(policy_loss, batch.entropies, masks, total_steps, metrics)

        policy_grad_norm = self._apply_policy_update(policy_loss)

        metrics.update(
            policy_loss=float(policy_loss.detach().cpu().item()),
            policy_grad_norm=policy_grad_norm,
        )
        return metrics

    def _batch_policy_loss(self, log_probs: Tensor, signal: Tensor, masks: Tensor, total_steps: Tensor) -> Tensor:
        return -((log_probs * signal * masks).sum() / total_steps)

    def _episode_policy_loss(self, log_probs: Tensor, masks: Tensor, returns: Tensor) -> Tensor:
        episode_logp_sum = (log_probs * masks).sum(dim=1)
        return -(episode_logp_sum * returns).mean()

    def _apply_policy_update(self, loss: Tensor) -> float:
        self.policy_optim.zero_grad(set_to_none=True)
        loss.backward()
        max_norm = self.cfg.grad_clip if self.cfg.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=max_norm).item()
        self.policy_optim.step()
        return float(grad_norm)

    def _apply_baseline_update(self, loss: Tensor) -> float:
        assert self.value_optim is not None and self.value_net is not None
        self.value_optim.zero_grad(set_to_none=True)
        loss.backward()
        max_norm = self.cfg.grad_clip if self.cfg.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=max_norm).item()
        self.value_optim.step()
        return float(grad_norm)

    def _maybe_normalise_signal(self, signal: Tensor, masks: Tensor) -> Tensor:
        if not self.cfg.normalize_adv:
            return signal
        values = self._masked_values(signal, masks)
        if values.numel() <= 1:
            return signal
        mean = values.mean()
        std = values.std(unbiased=False)
        return (signal - mean) / (std + 1e-8)

    def _apply_entropy_bonus(
        self, loss: Tensor, entropies: Tensor, masks: Tensor, total_steps: Tensor, metrics: Dict[str, float]
    ) -> Tensor:
        if self.cfg.entropy_coef <= 0:
            return loss
        bonus = -self.cfg.entropy_coef * (entropies * masks).sum() / total_steps
        metrics["entropy_bonus"] = float(bonus.detach().cpu().item())
        return loss + bonus

    def _masked_values(self, tensor: Tensor, masks: Tensor) -> Tensor:
        flat = (tensor * masks).view(-1)
        mask = masks.view(-1) > 0
        return flat[mask]

    def _masked_mean_std(self, tensor: Tensor, masks: Tensor) -> tuple[float, float]:
        values = self._masked_values(tensor, masks)
        if values.numel() == 0:
            return 0.0, 0.0
        mean = float(values.mean().detach().cpu().item())
        std = float(values.std(unbiased=False).detach().cpu().item()) if values.numel() > 1 else 0.0
        return mean, std

    def _rtg_stats(self, rtg: Tensor, masks: Tensor) -> Dict[str, float]:
        values = self._masked_values(rtg, masks)
        if values.numel() == 0:
            return {}
        rtg_mean = float(values.mean().detach().cpu().item())
        rtg_std = float(values.std(unbiased=False).detach().cpu().item()) if values.numel() > 1 else 0.0
        rtg_min = float(values.min().detach().cpu().item())
        rtg_max = float(values.max().detach().cpu().item())
        return {
            "rtg_mean": rtg_mean,
            "rtg_std": rtg_std,
            "rtg_min": rtg_min,
            "rtg_max": rtg_max,
        }
