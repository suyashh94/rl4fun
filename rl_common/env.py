from __future__ import annotations

import gymnasium as gym


def make_env(env_id: str, seed: int, normalize_obs: bool) -> gym.Env:
    """Create and optionally wrap a Gymnasium environment with normalization."""
    env = gym.make(env_id)
    env.reset(seed=seed)
    if normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
    return env
