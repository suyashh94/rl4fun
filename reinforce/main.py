from __future__ import annotations

import argparse
import platform

from reinforce.trainer import ReinforceTrainer, TrainerConfig


def parse_args() -> TrainerConfig:
    parser = argparse.ArgumentParser(description="REINFORCE training script (Phases 1-4)")
    parser.add_argument("--env", dest="env_id", type=str, default="CartPole-v1")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--episodes-per-update", type=int, default=10)
    parser.add_argument("--max-updates", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="reinforce/experiments/runs")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--normalize-obs", action="store_true", default=False, help="Normalize observations")
    parser.add_argument("--use-rtg", action="store_true", default=False, help="Use reward-to-go objective (Phase 2)")
    parser.add_argument("--use-baseline", action="store_true", default=False, help="Use learned value baseline (Phase 3)")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="Learning rate for the baseline network")
    parser.add_argument(
        "--normalize-adv",
        action="store_true",
        default=False,
        help="Normalize returns or advantages before updating the policy",
    )
    parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy bonus weight (Phase 4)")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Max gradient norm for clipping (0 disables)")
    args = parser.parse_args()

    return TrainerConfig(
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
        normalize_adv=args.normalize_adv,
        entropy_coef=args.entropy_coef,
        grad_clip=args.grad_clip,
    )


def main() -> None:
    cfg = parse_args()
    trainer = ReinforceTrainer(cfg)
    try:
        print(f"Seed: {trainer.seed}  Device: {trainer.device}")
        print(f"Logging to: {trainer.log_path}  (Python {platform.python_version()})")
        trainer.train()
    finally:
        trainer.close()
if __name__ == "__main__":
    main()
