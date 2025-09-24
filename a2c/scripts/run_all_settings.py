#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

mpl_cache = PROJECT_ROOT / ".mpl-cache"
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
mpl_cache.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from a2c.trainer import A2CTrainer, TrainerConfig, TrainingHistory


SETTINGS: List[Tuple[str, Dict[str, object]]] = [
    ("td_base", {"normalize_adv": False, "grad_clip": 0.0}),
    ("td_norm", {"normalize_adv": True, "grad_clip": 0.0}),
    ("td_clip", {"normalize_adv": False, "grad_clip": 1.0}),
    ("td_norm_clip", {"normalize_adv": True, "grad_clip": 1.0}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A2C TD settings and plot metrics")
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium environment id")
    parser.add_argument("--seed", type=int, default=42, help="Seed shared across settings")
    parser.add_argument("--episodes-per-update", type=int, default=10)
    parser.add_argument("--max-updates", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--actor-lr", type=float, default=3e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log-dir", default="a2c/experiments/runs", help="Base directory for run artefacts")
    parser.add_argument("--output-dir", default="a2c/experiments/comparisons", help="Where to write the plots")
    parser.add_argument("--normalize-obs", action="store_true", default=True)
    parser.add_argument("--no-normalize-obs", action="store_false", dest="normalize_obs")
    return parser.parse_args()


def run_setting(base_cfg: TrainerConfig, label: str, overrides: Dict[str, object]) -> Tuple[str, TrainingHistory]:
    tag = f"{label}_{time.strftime('%Y%m%d_%H%M%S')}"
    cfg = replace(base_cfg, tag=tag)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.tag = tag

    print(f"\n[setting] {label} -> tag={cfg.tag}")
    trainer = A2CTrainer(cfg)
    try:
        history = trainer.train()
    finally:
        trainer.close()
    return label, history


def plot_metric(output_dir: Path, histories: List[Tuple[str, TrainingHistory]], metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots()
    for label, history in histories:
        updates = history.updates
        values = getattr(history, metric)
        ax.plot(updates, values, label=label)
    ax.set_xlabel("Update")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} across settings")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = TrainerConfig(
        env_id=args.env,
        seed=args.seed,
        hidden=args.hidden,
        episodes_per_update=args.episodes_per_update,
        max_updates=args.max_updates,
        gamma=args.gamma,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        value_loss_coef=args.value_loss_coef,
        log_dir=args.log_dir,
        tag=None,
        normalize_obs=args.normalize_obs,
        normalize_adv=False,
        grad_clip=0.0,
    )

    histories: List[Tuple[str, TrainingHistory]] = []
    for label, overrides in SETTINGS:
        histories.append(run_setting(base_cfg, label, overrides))

    plot_metric(output_dir, histories, "avg_logp", "Average log probability")
    plot_metric(output_dir, histories, "return_ma", "Return moving average")
    plot_metric(output_dir, histories, "avg_entropy", "Average entropy")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
