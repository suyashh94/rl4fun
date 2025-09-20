from __future__ import annotations

import argparse
import time
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from reinforce.agents import CategoricalPolicy, ValueNet
from reinforce.algo.reinforce import (
    actor_loss,
    actor_loss_rtg,
    actor_critic_loss,
    collect_episodes,
    collect_episodes_steps,
)
from reinforce.utils.seeding import set_global_seeds
from reinforce.utils.tb import TbLogger
from reinforce.utils.metadata import get_git_info, get_versions, iso_now, write_json, append_jsonl


@dataclass
class Config:
    env_id: str = "CartPole-v1"
    seed: int = 1
    hidden: int = 128
    episodes_per_update: int = 10
    max_updates: int = 500
    gamma: float = 0.99
    lr: float = 3e-3
    log_dir: str = "reinforce/experiments/runs"
    tag: str | None = None
    normalize_obs: bool = False
    use_rtg: bool = False
    use_baseline: bool = False
    critic_lr: float = 1e-3


def make_env(env_id: str, seed: int, normalize_obs: bool) -> gym.Env:
    env = gym.make(env_id)
    # Seed via reset for gymnasium
    env.reset(seed=seed)
    if normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
    return env


def train(cfg: Config) -> None:
    from reinforce.utils.device import get_torch_device
    device = get_torch_device()
    seed = set_global_seeds(cfg.seed)

    env = make_env(cfg.env_id, seed, cfg.normalize_obs)
    obs_shape = env.observation_space.shape
    assert len(obs_shape) == 1, "Only flat observation spaces are supported in Phase 1"
    obs_dim = obs_shape[0]
    assert hasattr(env.action_space, "n"), "Only discrete action spaces supported in Phase 1"
    act_dim = env.action_space.n

    policy = CategoricalPolicy(obs_dim, act_dim, hidden=cfg.hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)

    value_net: ValueNet | None = None
    critic_optimizer: optim.Optimizer | None = None
    if cfg.use_baseline:
        value_net = ValueNet(obs_dim, hidden=cfg.hidden).to(device)
        critic_optimizer = optim.Adam(value_net.parameters(), lr=cfg.critic_lr)

    # Logging: ensure a unique run directory. If a tag is provided and already exists,
    # append a timestamp suffix to avoid collisions/overwrites.
    base_dir = Path(cfg.log_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg.tag or time.strftime("%Y%m%d_%H%M%S")
    log_path = base_dir / tag
    if log_path.exists() and any(log_path.iterdir()):
        suffix = time.strftime("%Y%m%d_%H%M%S")
        tag = f"{tag}_{suffix}"
        log_path = base_dir / tag
    tb = TbLogger(log_path)

    print(f"Seed: {seed}  Device: {device}")
    print(f"Logging to: {log_path}")

    # Save run metadata and config for reproducibility
    meta = {
        "created_at": iso_now(),
        "tag": tag,
        "seed": int(seed),
        "device": str(device),
        "env_id": cfg.env_id,
        "config": asdict(cfg),
        "versions": get_versions(),
        "git": get_git_info(),
        "tb_log_dir": str(log_path),
        "python": platform.python_version(),
    }
    write_json(log_path / "config.json", meta)
    # Also append to a global registry.jsonl for quick mapping later
    append_jsonl(Path(cfg.log_dir) / "registry.jsonl", meta)

    ret_ma = None  # moving average of return
    ma_alpha = 0.1

    best_ret_mean = -float("inf")
    for update in range(1, cfg.max_updates + 1):
        t0 = time.time()
        actor_loss_tensor: torch.Tensor
        critic_loss_tensor: torch.Tensor | None = None
        diag: Dict[str, float] = {}
        actor_grad_norm = 0.0
        critic_grad_norm = None

        if cfg.use_baseline:
            assert value_net is not None and critic_optimizer is not None
            all_logps, all_rewards, all_obs, stats = collect_episodes_steps(
                env=env, policy=policy, episodes=cfg.episodes_per_update, device=device
            )
            policy.train()
            value_net.train()
            optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            actor_loss_tensor, critic_loss_tensor, diag = actor_critic_loss(
                all_logps=all_logps,
                all_rewards=all_rewards,
                all_obs=all_obs,
                gamma=cfg.gamma,
                value_net=value_net,
                device=device,
            )
            actor_loss_tensor.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float("inf")).item()
            optimizer.step()

            critic_loss_tensor.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=float("inf")).item()
            critic_optimizer.step()
        elif cfg.use_rtg:
            all_logps, all_rewards, _all_obs, stats = collect_episodes_steps(
                env=env, policy=policy, episodes=cfg.episodes_per_update, device=device
            )
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            actor_loss_tensor, diag = actor_loss_rtg(all_logps, all_rewards, gamma=cfg.gamma, device=device)
            actor_loss_tensor.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float("inf")).item()
            optimizer.step()
        else:
            sum_logps, returns, stats = collect_episodes(
                env=env, policy=policy, episodes=cfg.episodes_per_update, device=device, gamma=cfg.gamma
            )
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            actor_loss_tensor = actor_loss(sum_logps, stats.returns, device=device)
            actor_loss_tensor.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float("inf")).item()
            optimizer.step()

        dt = time.time() - t0
        ret_mean = float(np.mean(stats.returns)) if stats.returns else 0.0
        ret_med = float(np.median(stats.returns)) if stats.returns else 0.0
        ret_max = float(np.max(stats.returns)) if stats.returns else 0.0
        ret_ma = ret_mean if ret_ma is None else (1 - ma_alpha) * ret_ma + ma_alpha * ret_mean
        best_ret_mean = max(best_ret_mean, ret_mean)

        # TB logs
        tb.log_scalar("train/return_mean", ret_mean, update)
        tb.log_scalar("train/return_median", ret_med, update)
        tb.log_scalar("train/return_max", ret_max, update)
        tb.log_scalar("train/return_ma", ret_ma, update)
        tb.log_scalar("train/actor_loss", float(actor_loss_tensor.detach().cpu().item()), update)
        tb.log_scalar("train/actor_grad_norm", actor_grad_norm, update)
        tb.log_scalar("train/grad_norm", actor_grad_norm, update)

        if cfg.use_baseline and critic_loss_tensor is not None:
            tb.log_scalar("train/critic_loss", float(critic_loss_tensor.detach().cpu().item()), update)
            tb.log_scalar("train/value_mae", diag.get("value_mae", 0.0), update)
            tb.log_scalar("train/adv_mean", diag.get("adv_mean", 0.0), update)
            tb.log_scalar("train/adv_std", diag.get("adv_std", 0.0), update)
            if critic_grad_norm is not None:
                tb.log_scalar("train/critic_grad_norm", critic_grad_norm, update)

        if cfg.use_baseline or cfg.use_rtg:
            tb.log_scalar("train/gt_mean", diag.get("gt_mean", 0.0), update)
            tb.log_scalar("train/gt_std", diag.get("gt_std", 0.0), update)
            tb.log_scalar("train/gt_min", diag.get("gt_min", 0.0), update)
            tb.log_scalar("train/gt_max", diag.get("gt_max", 0.0), update)
        tb.log_scalar("train/avg_entropy", stats.avg_entropy, update)
        tb.log_scalar("train/avg_logp", stats.avg_logp, update)
        tb.log_scalar("time/update_sec", dt, update)

        if update % 10 == 0 or update == 1:
            msg = (
                f"update {update:4d} | ret_mean {ret_mean:7.2f} | ret_ma {ret_ma:7.2f} | "
                f"actor {actor_loss_tensor.item():.4f} | H {stats.avg_entropy:.3f} | dt {dt:.2f}s"
            )
            if cfg.use_baseline and critic_loss_tensor is not None:
                msg += f" | critic {critic_loss_tensor.item():.4f}"
            print(msg)

    # Finalize and update a small summary file
    summary = {
        "finished_at": iso_now(),
        "best_return_mean": float(best_ret_mean),
        "last_return_ma": float(ret_ma if ret_ma is not None else 0.0),
        "updates": int(cfg.max_updates),
    }
    write_json(log_path / "summary.json", summary)
    tb.close()
    env.close()


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="REINFORCE training script (Phases 1-3)")
    p.add_argument("--env", dest="env_id", type=str, default="CartPole-v1")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--episodes-per-update", type=int, default=10)
    p.add_argument("--max-updates", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log-dir", type=str, default="reinforce/experiments/runs")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--normalize-obs", action="store_true", default=False, help="Normalize observations")
    p.add_argument("--use-rtg", action="store_true", default=False, help="Use reward-to-go objective (Phase 2)")
    p.add_argument("--use-baseline", action="store_true", default=False, help="Use learned value baseline (Phase 3)")
    p.add_argument("--critic-lr", type=float, default=1e-3, help="Learning rate for value network when baseline is enabled")
    args = p.parse_args()
    return Config(
        env_id=args.env_id,
        lr=args.lr,
        hidden=args.hidden,
        episodes_per_update=args.episodes_per_update,
        max_updates=args.max_updates,
        gamma=args.gamma,
        seed=args.seed,
        log_dir=args.log_dir,
        tag=args.tag,
        normalize_obs=args.normalize_obs,
        use_rtg=args.use_rtg,
        use_baseline=args.use_baseline,
        critic_lr=args.critic_lr,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
