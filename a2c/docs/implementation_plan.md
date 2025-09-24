# A2C Implementation Plan (Single-Step TD)

The goal is to mirror the REINFORCE package structure with an Advantage Actor-Critic (A2C) variant that uses one-step temporal-difference targets. This plan covers the initial TD implementation while laying the groundwork for future GAE support.

## 1. Repository layout

Replicate the REINFORCE layout under a new `a2c/` package, reusing modules from `rl_common/` whenever possible.

```
a2c/
  __init__.py
  trainer.py          # main training loop
  main.py             # CLI entrypoint (mirrors reinforce/main.py)
  agents/
    __init__.py
    actor_critic.py   # shared backbone with separate policy/value heads
  cfgs/
    cartpole_td.json  # sample config(s)
  scripts/
    __init__.py
    run_from_config.sh
    run_all_settings.py
  docs/
    implementation_plan.md (this file)
```

Key principles:
- Use `rl_common` for environment setup, logging, seeding, metadata, returns, etc.
- Maintain parity with REINFORCE filenames/CLI options where semantics align (e.g. `--normalize-obs`, `--normalize-adv`, `--entropy-coef`, `--grad-clip`).
- Avoid REINFORCE-specific flags (`--use-rtg`, `--use-baseline`); introduce `--use-td` (default true) and `--use-gae` (future toggle, default false).

## 2. Configuration & CLI

`TrainerConfig` (and CLI flags) should include:
- Environment / seed / hidden size / episodes-per-update / max-updates / gamma
- Learning rates: `actor_lr`, `critic_lr` (allow tying them together but keep independent fields)
- `normalize_obs`, `normalize_adv`, `entropy_coef`, `grad_clip`
- `use_td` (bool, default true) — when false and `use_gae` false -> fall back to Monte Carlo advantages (mainly for testing)
- `use_gae` (bool, default false) with `gae_lambda` placeholder (not used until Phase 2)
- Logging directory + tag

## 3. Networks

Create `a2c/agents/actor_critic.py` with:
- Shared backbone MLP producing feature vector
- Two heads: policy logits (categorical) and value scalar
- Forward methods returning distribution + value (and optionally features for logging)

## 4. Rollout collection

Follow the batched collector from `ReinforceTrainer` with adjustments:
- For each step record `obs`, `action`, `log_prob`, `entropy`, `reward`, `done` mask, and `value` prediction.
- Keep `next_obs` to compute TD targets.
- Pad episodes to the max length; produce tensors `[B, T, ...]` with masks identical to REINFORCE.
- Store `bootstrap_value` for the final state (0 when terminal, V(next_state) when truncated or continuing).

## 5. Losses

Inside `A2CTrainer`:
- Policy loss (TD variant): `-(log_pi * advantage)` where `advantage = reward + gamma * V(next) * (1 - done) - V(current)`.
- Value loss: MSE of TD target vs. V(current).
- Optional: when `use_gae` is enabled (future), compute GAE λ advantages before reusing the same code path.
- Entropy bonus and gradient clipping identical to REINFORCE helpers.
- Advantage normalisation applies to the computed advantages before policy loss.

## 6. Training loop structure

Mirror `ReinforceTrainer`:
- `_setup_environment`, `_setup_logging`, `_collect_rollout`, `_update`, `_handle_logging`, etc., reusing helper methods via inheritance or composition where sensible.
- Maintain `TrainingHistory` with `return_mean`, `return_ma`, `avg_entropy`, `avg_logp`, plus additional critic metrics (e.g. value loss, TD error stats).
- Use `rl_common.discount_cumsum` for optional Monte Carlo baselines or diagnostics (even if TD is default).

## 7. Scripts & configs

- Port `run_from_config.sh` into `a2c/scripts/` with appropriate flags.
- Provide `run_all_settings.py` (analogue of `run_all_phases.py`) to compare TD with / without normalisation & grad clipping (and later GAE).
- Sample config `cartpole_td.json` enabling `normalize_obs` by default, toggling `normalize_adv`/`grad_clip` combinations.

## 8. Documentation

- Write `a2c/docs/README.md` or extend this plan with derivation notes (TD target, advantage computation).
- Document feature toggles, CLI usage, and plotting script referencing `a2c/scripts/` locations.
- Note shared dependency on `rl_common` to encourage reuse by other algorithms.

## 9. Validation strategy

- Smoke test via `python -m a2c.main --max-updates 1` using CartPole to verify shapes.
- Use `run_all_settings.py` for short runs capturing metrics (log prob, entropy, returns).
- Monitor parity with REINFORCE logging fields to ease dashboard comparisons.

## 10. Future extensions

- Add Phase 2 with GAE: compute λ-returns, mix with TD target, expose `--use-gae/--gae-lambda`.
- Integrate shared plotting utilities (possibly move to `rl_common/plots.py`).
- Provide cross-algorithm experiment scripts comparing REINFORCE vs. A2C once both stabilise.
