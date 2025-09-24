from __future__ import annotations

import argparse
import platform

from a2c.trainer import A2CTrainer, TrainerConfig


def parse_args() -> TrainerConfig:
    parser = argparse.ArgumentParser(description="A2C training script (single-step TD)")
    parser.add_argument("--env", dest="env_id", type=str, default="CartPole-v1")
    parser.add_argument("--learning-rate", type=float, default=7e-4, help="Learning rate for both actor and critic")
    # Backward compatibility
    parser.add_argument("--actor-lr", type=float, help="Deprecated: use --learning-rate")
    parser.add_argument("--critic-lr", type=float, help="Deprecated: use --learning-rate")
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--rollout-length", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-updates", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="a2c/experiments/runs")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--normalize-obs", action="store_true", default=True, help="Normalize observations (default on)")
    parser.add_argument("--no-normalize-obs", action="store_false", dest="normalize_obs")
    parser.add_argument("--normalize-adv", action="store_true", default=False, help="Normalize advantages before policy update")
    parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy bonus weight")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Max gradient norm for clipping (0 disables)")
    # use_td removed - always use GAE now
    parser.add_argument("--use-gae", action="store_true", default=False, help="Use GAE advantages (placeholder)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--eval-freq", type=int, default=100, help="Evaluate policy every N updates")
    parser.add_argument("--n-eval-episodes", type=int, default=10, help="Number of episodes for evaluation")
    args = parser.parse_args()

    return TrainerConfig(
        env_id=args.env_id,
        learning_rate=args.learning_rate if args.learning_rate != 7e-4 else (args.actor_lr if args.actor_lr else 7e-4),
        value_loss_coef=args.value_loss_coef,
        hidden=args.hidden,
        rollout_length=args.rollout_length,
        num_envs=args.num_envs,
        max_updates=args.max_updates,
        gamma=args.gamma,
        seed=args.seed,
        log_dir=args.log_dir,
        tag=args.tag,
        normalize_obs=args.normalize_obs,
        normalize_adv=args.normalize_adv,
        entropy_coef=args.entropy_coef,
        grad_clip=args.grad_clip,
        # use_td removed - always use GAE
        use_gae=args.use_gae,
        gae_lambda=args.gae_lambda,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
    )


def main() -> None:
    cfg = parse_args()
    trainer = A2CTrainer(cfg)
    try:
        print(f"Seed: {trainer.seed}  Device: {trainer.device}")
        print(f"Logging to: {trainer.log_path}  (Python {platform.python_version()})")
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
