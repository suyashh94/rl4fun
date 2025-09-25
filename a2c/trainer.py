from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from platform import python_version
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import numpy as np
import torch
from torch import Tensor

from a2c.agents import ActorCriticNet
from rl_common.env import make_env
from rl_common.metadata import append_jsonl, get_git_info, get_versions, iso_now, write_json
from rl_common.seeding import set_global_seeds
from rl_common.tb import TbLogger


@dataclass
class TrainerConfig:
    env_id: str = "CartPole-v1"
    seed: int = 1
    hidden: int = 128
    rollout_length: int = 20
    num_envs: int = 8
    max_updates: int = 500
    eval_freq: int = 100
    n_eval_episodes: int = 10
    gamma: float = 0.99
    learning_rate: float = 7e-4  # Unified learning rate like SB3
    value_loss_coef: float = 0.5
    log_dir: str = "a2c/experiments/runs"
    tag: Optional[str] = None
    normalize_obs: bool = True
    normalize_adv: bool = False
    entropy_coef: float = 0.0
    grad_clip: float = 0.0
    # use_td removed - we always use GAE now
    use_gae: bool = False
    gae_lambda: float = 0.95


@dataclass
class RolloutBatch:
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    log_probs: Tensor
    entropies: Tensor
    values: Tensor
    next_values: Tensor
    dones: Tensor
    rtg: Tensor
    masks: Tensor
    lengths: Tensor
    raw_returns: Tensor


@dataclass
class TrainingHistory:
    updates: List[int] = field(default_factory=list)
    return_mean: List[float] = field(default_factory=list)
    return_ma: List[float] = field(default_factory=list)
    avg_entropy: List[float] = field(default_factory=list)
    avg_logp: List[float] = field(default_factory=list)
    eval_returns: List[float] = field(default_factory=list)
    eval_updates: List[int] = field(default_factory=list)




class A2CTrainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = set_global_seeds(cfg.seed)

        self.env, self.obs_dim, self.act_dim = self._setup_environment()
        self._current_obs = None  # Track current observations across rollouts

        self.model = ActorCriticNet(self.obs_dim, self.act_dim, hidden=cfg.hidden).to(self.device)
        # Use RMSprop like SB3 (shared learning rate)
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=cfg.learning_rate,
            eps=1e-5,
            alpha=0.99
        )

        self.history = TrainingHistory()
        self.tag, self.log_path, self.tb = self._setup_logging()
        self._last_entropy_bonus: Optional[float] = None

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
            metrics = self._update(batch, update)

            ret_ma, stats = self._handle_logging(batch, metrics, update, ret_ma)
            best_ret_mean = max(best_ret_mean, stats["ret_mean"])

            # Perform evaluation
            if update % self.cfg.eval_freq == 0 or update == 1:
                eval_stats = self.evaluate_policy()
                self.history.eval_returns.append(eval_stats['eval_return_mean'])
                self.history.eval_updates.append(update)

                # Log evaluation results
                self.tb.log_scalar("eval/return_mean", eval_stats['eval_return_mean'], update)
                self.tb.log_scalar("eval/return_std", eval_stats['eval_return_std'], update)
                self.tb.log_scalar("eval/return_max", eval_stats['eval_return_max'], update)
                self.tb.log_scalar("eval/length_mean", eval_stats['eval_length_mean'], update)

            if update % 10 == 0 or update == 1:
                eval_info = ""
                if update % self.cfg.eval_freq == 0 or update == 1:
                    eval_info = f" | eval_ret {eval_stats['eval_return_mean']:7.2f}"

                msg = (
                    f"update {update:4d} | ret_mean {stats['ret_mean']:7.2f} | ret_ma {stats['ret_ma']:7.2f} | "
                    f"policy {metrics['policy_loss']:.4f} | value {metrics['value_loss']:.4f} | H {stats['avg_entropy']:.3f}{eval_info}"
                )
                print(msg)

                # Check if agent has learned CartPole (solved = 195+ average over 100 episodes)
                if len(self.history.eval_returns) >= 2:  # Need at least 2 eval points
                    recent_eval_mean = np.mean(self.history.eval_returns[-2:])
                    if recent_eval_mean >= 195.0:
                        print(f"\n🎉 CartPole-v1 SOLVED! Average return: {recent_eval_mean:.2f} >= 195.0")
                        print(f"Agent reached target performance at update {update}")

        # Final evaluation
        if self.cfg.max_updates % self.cfg.eval_freq != 0:
            eval_stats = self.evaluate_policy()
            self.history.eval_returns.append(eval_stats['eval_return_mean'])
            self.history.eval_updates.append(self.cfg.max_updates)

        summary = {
            "finished_at": iso_now(),
            "best_return_mean": float(best_ret_mean),
            "last_return_ma": float(ret_ma if ret_ma is not None else 0.0),
            "final_eval_return": float(self.history.eval_returns[-1]) if self.history.eval_returns else 0.0,
            "best_eval_return": float(max(self.history.eval_returns)) if self.history.eval_returns else 0.0,
            "updates": int(self.cfg.max_updates),
            "solved": float(max(self.history.eval_returns)) >= 195.0 if self.history.eval_returns else False,
        }
        write_json(self.log_path / "summary.json", summary)
        return self.history

    def _setup_environment(self) -> Tuple[gym.Env, int, int]:
        # Create vectorized environments
        def make_single_env(rank: int):
            def _init():
                return make_env(self.cfg.env_id, self.seed + rank, self.cfg.normalize_obs)
            return _init

        envs = [make_single_env(i) for i in range(self.cfg.num_envs)]
        env = SyncVectorEnv(envs)

        obs_shape = env.single_observation_space.shape
        if obs_shape is None or len(obs_shape) != 1:
            raise ValueError("A2CTrainer only supports flat observation spaces")
        obs_dim = obs_shape[0]
        if not hasattr(env.single_action_space, "n"):
            raise ValueError("A2CTrainer currently supports discrete action spaces")
        act_dim = env.single_action_space.n
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

    def _collect_rollout(self) -> RolloutBatch:
        # Initialize environments if needed
        if self._current_obs is None:
            self._current_obs, _ = self.env.reset()

        # Storage for rollout data
        obs_storage = np.zeros((self.cfg.rollout_length, self.cfg.num_envs, self.obs_dim), dtype=np.float32)
        actions_storage = np.zeros((self.cfg.rollout_length, self.cfg.num_envs), dtype=np.int64)
        rewards_storage = np.zeros((self.cfg.rollout_length, self.cfg.num_envs), dtype=np.float32)
        dones_storage = np.zeros((self.cfg.rollout_length, self.cfg.num_envs), dtype=np.float32)
        values_storage = np.zeros((self.cfg.rollout_length, self.cfg.num_envs), dtype=np.float32)
        logp_storage = np.zeros((self.cfg.rollout_length, self.cfg.num_envs), dtype=np.float32)
        entropy_storage = np.zeros((self.cfg.rollout_length, self.cfg.num_envs), dtype=np.float32)

        episode_rewards = np.zeros(self.cfg.num_envs, dtype=np.float32)
        episode_lengths = np.zeros(self.cfg.num_envs, dtype=np.int32)
        finished_episodes = []

        # Collect rollout_length steps
        for step in range(self.cfg.rollout_length):
            # Store current observations
            obs_storage[step] = self._current_obs

            # Get actions and values from policy
            obs_tensor = torch.as_tensor(self._current_obs, dtype=torch.float32, device=self.device)
            dist, values = self.model(obs_tensor)
            actions = dist.sample()

            # Store data (detach for storage, but keep gradients for training)
            with torch.no_grad():
                logp = dist.log_prob(actions)
                entropy = dist.entropy()
                actions_storage[step] = actions.cpu().numpy()
                values_storage[step] = values.cpu().numpy()
                logp_storage[step] = logp.cpu().numpy()
                entropy_storage[step] = entropy.cpu().numpy()

            # Take environment step
            next_obs, rewards, dones, truncateds, _ = self.env.step(actions.cpu().numpy())

            # Handle episode termination
            terminated = np.logical_or(dones, truncateds)

            # Store step data
            rewards_storage[step] = rewards
            dones_storage[step] = terminated.astype(np.float32)

            # Track episode statistics
            episode_rewards += rewards
            episode_lengths += 1

            # Record finished episodes
            for env_idx in range(self.cfg.num_envs):
                if terminated[env_idx]:
                    finished_episodes.append({
                        'reward': float(episode_rewards[env_idx]),
                        'length': int(episode_lengths[env_idx])
                    })
                    episode_rewards[env_idx] = 0.0
                    episode_lengths[env_idx] = 0

            self._current_obs = next_obs

        # Bootstrap values for the final step
        obs_tensor = torch.as_tensor(self._current_obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, bootstrap_values = self.model(obs_tensor)
        bootstrap_values = bootstrap_values.cpu().numpy()

        # Convert to tensors
        observations = torch.as_tensor(obs_storage, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions_storage, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards_storage, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones_storage, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(values_storage, dtype=torch.float32, device=self.device)
        log_probs = torch.as_tensor(logp_storage, dtype=torch.float32, device=self.device)
        entropies = torch.as_tensor(entropy_storage, dtype=torch.float32, device=self.device)
        bootstrap_values_tensor = torch.as_tensor(bootstrap_values, dtype=torch.float32, device=self.device)

        # Compute next values (values at t+1) for GAE
        next_values = torch.cat([values[1:], bootstrap_values_tensor.unsqueeze(0)], dim=0)

        # Create masks (all ones for fixed-length rollouts)
        masks = torch.ones_like(rewards)

        # Flatten batch dimensions for compatibility
        observations = observations.view(-1, self.obs_dim)
        actions = actions.view(-1)
        rewards = rewards.view(-1)
        dones = dones.view(-1)
        values = values.view(-1)
        next_values = next_values.view(-1)
        log_probs = log_probs.view(-1)
        entropies = entropies.view(-1)
        rtg = rewards  # Placeholder - we don't use RTG anymore
        masks = masks.view(-1)

        # Episode statistics
        raw_returns = torch.tensor([ep['reward'] for ep in finished_episodes] if finished_episodes else [0.0],
                                 dtype=torch.float32, device=self.device)
        lengths = torch.tensor([self.cfg.rollout_length * self.cfg.num_envs], dtype=torch.long, device=self.device)

        return RolloutBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            next_values=next_values,
            dones=dones,
            rtg=rewards,  # Placeholder - we don't use RTG anymore
            masks=masks,
            lengths=lengths,
            raw_returns=raw_returns,
        )

    def _compute_gae_advantages(self, rewards: Tensor, values: Tensor, next_values: Tensor, dones: Tensor) -> Tensor:
        """Compute Generalized Advantage Estimation (GAE) advantages."""
        # The tensors are flattened, so we need to work with the original 2D shape
        # Reshape to (T, N) for proper GAE computation
        T, N = self.cfg.rollout_length, self.cfg.num_envs

        rewards_2d = rewards.view(T, N)
        values_2d = values.view(T, N)
        next_values_2d = next_values.view(T, N)
        dones_2d = dones.view(T, N)

        advantages_2d = torch.zeros_like(rewards_2d)
        last_gae_lam = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)

        # Work backwards through time to compute GAE
        # CRITICAL: last_gae_lam should reset at episode boundaries!
        for step in reversed(range(T)):
            next_non_terminal = 1.0 - dones_2d[step]
            delta = rewards_2d[step] + self.cfg.gamma * next_values_2d[step] * next_non_terminal - values_2d[step]

            # FIXED: Reset last_gae_lam at episode boundaries (when next_non_terminal=0)
            last_gae_lam = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last_gae_lam
            advantages_2d[step] = last_gae_lam

            # CRITICAL FIX: Reset GAE accumulator when episode ends
            # This prevents advantages from leaking across episode boundaries
            last_gae_lam = last_gae_lam * next_non_terminal

        return advantages_2d.view(-1)  # Flatten back

    def evaluate_policy(self, n_episodes: int = None) -> Dict[str, float]:
        """Evaluate the current policy on a single environment."""
        n_episodes = n_episodes or self.cfg.n_eval_episodes

        # Create single evaluation environment
        from rl_common.env import make_env
        eval_env = make_env(self.cfg.env_id, self.seed + 999, self.cfg.normalize_obs)

        episode_returns = []
        episode_lengths = []

        self.model.eval()  # Set to evaluation mode

        try:
            for _ in range(n_episodes):
                obs, _ = eval_env.reset()
                episode_reward = 0.0
                episode_length = 0
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                    with torch.no_grad():
                        dist, _ = self.model(obs_tensor)
                        # Use deterministic policy for evaluation (no sampling)
                        action = dist.probs.argmax(dim=-1)

                    obs, reward, terminated, truncated, _ = eval_env.step(action.item())
                    episode_reward += reward
                    episode_length += 1

                episode_returns.append(episode_reward)
                episode_lengths.append(episode_length)

        finally:
            eval_env.close()
            self.model.train()  # Set back to training mode

        eval_stats = {
            'eval_return_mean': float(np.mean(episode_returns)),
            'eval_return_std': float(np.std(episode_returns)),
            'eval_return_min': float(np.min(episode_returns)),
            'eval_return_max': float(np.max(episode_returns)),
            'eval_length_mean': float(np.mean(episode_lengths)),
        }

        return eval_stats

    def _update(self, batch: RolloutBatch, _: int = 0) -> Dict[str, float]:
        """Update the actor-critic network using collected rollout data."""
        masks = batch.masks
        total_steps = masks.sum().clamp_min(1.0)

        # Recompute policy outputs for gradient computation
        dist, values_pred = self.model(batch.observations)
        log_probs = dist.log_prob(batch.actions)
        entropies = dist.entropy()

        # Always use GAE for advantage computation (simplified)
        advantages = self._compute_gae_advantages(batch.rewards, batch.values, batch.next_values, batch.dones)
        returns = advantages + batch.values  # Returns = advantages + values (SB3 style)
        advantage_policy = advantages.detach()

        if self.cfg.normalize_adv:
            advantage_policy = self._maybe_normalise_signal(advantage_policy, masks)

        # Actor loss: policy gradient with advantages
        policy_loss = -((log_probs * advantage_policy * masks).sum() / total_steps)

        # Critic loss: MSE between predicted values and returns
        value_loss = ((values_pred - returns.detach()).pow(2) * masks).sum() / total_steps
        value_loss = 0.5 * value_loss

        loss = policy_loss + self.cfg.value_loss_coef * value_loss
        loss = self._apply_entropy_bonus(loss, entropies, masks, total_steps)


        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._apply_grad_clip()
        self.optimizer.step()


        adv_mean, adv_std = self._masked_mean_std(advantage_policy, masks)
        td_error = advantage_policy
        td_mean, td_std = self._masked_mean_std(td_error, masks)

        metrics: Dict[str, float] = {
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "total_loss": float(loss.detach().cpu().item()),
            "policy_grad_norm": grad_norm,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "td_mean": td_mean,
            "td_std": td_std,
        }
        return metrics

    def _apply_entropy_bonus(
        self,
        loss: Tensor,
        entropies: Tensor,
        masks: Tensor,
        total_steps: Tensor,
    ) -> Tensor:
        if self.cfg.entropy_coef <= 0.0:
            self._last_entropy_bonus = None
            return loss
        bonus = -self.cfg.entropy_coef * (entropies * masks).sum() / total_steps
        self._last_entropy_bonus = float(bonus.detach().cpu().item())
        return loss + bonus

    def _apply_grad_clip(self) -> float:
        max_norm = self.cfg.grad_clip if self.cfg.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
        return float(grad_norm)

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
        self.tb.log_scalar("train/value_loss", metrics["value_loss"], update)
        self.tb.log_scalar("train/total_loss", metrics["total_loss"], update)
        self.tb.log_scalar("train/policy_grad_norm", metrics["policy_grad_norm"], update)
        self.tb.log_scalar("train/avg_entropy", stats["avg_entropy"], update)
        self.tb.log_scalar("train/avg_logp", stats["avg_logp"], update)
        self.tb.log_scalar("train/adv_mean", metrics["adv_mean"], update)
        self.tb.log_scalar("train/adv_std", metrics["adv_std"], update)
        self.tb.log_scalar("train/td_mean", metrics["td_mean"], update)
        self.tb.log_scalar("train/td_std", metrics["td_std"], update)
        self.tb.log_scalar("train/steps", stats["total_steps"], update)

        if self.cfg.entropy_coef > 0:
            bonus = getattr(self, "_last_entropy_bonus", None)
            if bonus is not None:
                self.tb.log_scalar("train/entropy_bonus", bonus, update)

    def _maybe_normalise_signal(self, signal: Tensor, masks: Tensor) -> Tensor:
        values = self._masked_values(signal, masks)
        if values.numel() <= 1:
            return signal
        mean = values.mean()
        std = values.std(unbiased=False)
        return (signal - mean) / (std + 1e-8)

    def _masked_values(self, tensor: Tensor, masks: Tensor) -> Tensor:
        flat = (tensor * masks).view(-1)
        mask = masks.view(-1) > 0
        return flat[mask]

    def _masked_mean_std(self, tensor: Tensor, masks: Tensor) -> Tuple[float, float]:
        values = self._masked_values(tensor, masks)
        if values.numel() == 0:
            return 0.0, 0.0
        mean = float(values.mean().detach().cpu().item())
        std = float(values.std(unbiased=False).detach().cpu().item()) if values.numel() > 1 else 0.0
        return mean, std
