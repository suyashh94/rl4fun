from __future__ import annotations

import argparse
import time
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from reinforce.agents.policy_categorical import CategoricalPolicy
from reinforce.algo.reinforce import actor_loss, collect_episodes
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

    # Logging
    tag = cfg.tag or time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(cfg.log_dir) / tag
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
        sum_logps, returns, stats = collect_episodes(
            env=env, policy=policy, episodes=cfg.episodes_per_update, device=device, gamma=cfg.gamma
        )
        
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss = actor_loss(sum_logps, returns, device=device)
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float("inf")).item()
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
        tb.log_scalar("train/actor_loss", float(loss.item()), update)
        tb.log_scalar("train/avg_entropy", stats.avg_entropy, update)
        tb.log_scalar("train/avg_logp", stats.avg_logp, update)
        tb.log_scalar("train/grad_norm", grad_norm, update)
        tb.log_scalar("time/update_sec", dt, update)

        if update % 10 == 0 or update == 1:
            print(
                f"update {update:4d} | ret_mean {ret_mean:7.2f} | ret_ma {ret_ma:7.2f} | "
                f"loss {loss.item():.4f} | H {stats.avg_entropy:.3f} | dt {dt:.2f}s"
            )

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
    p = argparse.ArgumentParser(description="Phase 1: Vanilla REINFORCE on CartPole")
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
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
