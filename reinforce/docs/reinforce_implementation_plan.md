# Implementation Plan: REINFORCE on CartPole

## Tech Stack & Baseline
- Python ≥ 3.10 (using 3.12 locally/devcontainer)
- Libraries: torch, gymnasium[classic_control], numpy, tensorboard, tqdm, matplotlib
- Environment: `CartPole-v1`
- Seeding: seed NumPy, PyTorch, and Gym; log the seed

## Repository Layout
```
reinforce/
  main.py
  cfgs/
    cartpole.yaml
  agents/
    policy_categorical.py
    value_net.py
  algo/
    reinforce.py
    returns.py
    meters.py
    eval.py
  utils/
    seeding.py
    tb.py
    ckpt.py
  experiments/
    runs/               # tensorboard
    checkpoints/
```

---

## Phase 1 — Vanilla REINFORCE (episode return, no baseline)
Goal: working agent; minimal features to validate pipeline.

Core loop
- Collect one (or K) full episodes with current policy.
- Compute total return R(τ) (undiscounted or discounted).
- Loss: `L_actor = - Σ_t logπ(a_t|s_t) * R_total`.
- Optimizer: Adam(lr=3e-3); no entropy bonus, no clipping.

Tracking (TensorBoard)
- Per update: episode return (mean/median/max), update time.
- Policy stats: average logπ, policy entropy.
- Grad norm of actor.

CLI / config flags (start simple)
- `--env CartPole-v1 --lr 3e-3 --hidden 128 --episodes-per-update 10 --max-updates 500 --gamma 0.99 --seed 1`

Exit criterion
- Moving-average return > 150 within ~200–300 updates.

---

## Phase 2 — Reward-to-Go (variance reduction)
Add
- Replace `R_total` with `G_t` (discounted return from step t).
- Loss: `L_actor = - mean_t( logπ_t * G_t )`.
- Optional batching across episodes before one backward.

New tracking
- Distribution of `G_t`: mean/std/min/max (per batch).
- Compare learning curves vs Phase 1 (run both with `--use_rtg {0,1}`).

Flags
- `--use_rtg {0,1}`

Ablation
- Run both settings with same seeds; plot returns side-by-side.

---

## Phase 3 — Baseline (value network) → Advantages
Add
- Value net `V(s)` (MLP). Critic loss: `0.5 * (G_t - V(s_t))^2`.
- Advantage: `A_t = (G_t - V(s_t)).detach()`.
- Policy loss: `-(logπ_t * A_t).mean()`.

New tracking
- Critic loss, value MAE `|G - V|`.
- Advantage stats (mean≈0 if normalized, std).
- Actor/critic grad norms (separate).

Flags
- `--use_baseline {0,1}` (auto-enables value net if 1)
- `--critic_lr 1e-3`

---

## Phase 4 — Stabilizers & Toggles (compare settings)
Add
- Advantage normalization (per batch): `(A - mean)/std`.
- Entropy bonus: `-β * entropy` in actor loss.
- Gradient clipping: `clip_grad_norm_(params, max_norm)`.

New tracking
- Entropy, β used, grad norms (post-clip), % grads clipped.
- Advantage norm on/off, its std.

Flags
- `--normalize_adv {0,1}`
- `--entropy_coef 0.0..0.01`
- `--grad_clip 0|0.5|1.0|2.0`

Ablation matrix (CartPole)

| Feature         | Off | On    |
|-----------------|-----|-------|
| Reward-to-Go    | ✓   | ✓     |
| Baseline        | ✓   | ✓     |
| Adv norm        | ✓   | ✓     |
| Entropy bonus   | 0.0 | 0.003 |
| Grad clip       | 0   | 1.0   |

Run key pairs with 3–5 seeds; log mean±std.

---

## Phase 5 — Batch & Optimizer Hygiene
Add
- Episodes-per-update sweep: 5, 10, 20.
- LR sweep (actor & critic): 1e-2, 3e-3, 1e-3.
- Optional LR scheduler (cosine/step).

Tracking
- KL between consecutive policies (optional).
- Wall-clock vs return curve.

Flags
- `--episodes_per_update 10`
- `--actor_lr 3e-3 --critic_lr 1e-3`
- `--lr_sched {none,cosine,step}`

---

## Phase 6 — Evaluation & Reproducibility
Add
- Eval mode: every N updates, run 10 eval episodes with argmax(logits) (or sample w/o grad); record mean/max.
- Checkpoints: save best policy by eval return; periodic autosave.
- Run metadata: dump config + git commit + seed to JSON.

Flags
- `--eval_every 10 --eval_episodes 10`
- `--save_dir ./experiments/checkpoints --tag cartpole_rtg_baseline_clip`

Outputs
- `experiments/runs/<tag>` (TensorBoard)
- `experiments/checkpoints/<tag>/best.pt` (+ `config.json`)

---

## Phase 7 — Report & Plots
- Plot training return (smoothed), eval return, entropy, value MAE, grad norms.
- Summary table:
  - Best avg eval return
  - Updates to reach 200
  - Area under curve (AUC)
  - Stability (std across seeds)

---

## Minimal API (what each file does)

`agents/policy_categorical.py`
```python
class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        ...
    def forward(self, obs):  # returns dist
        logits = net(obs)
        return torch.distributions.Categorical(logits=logits)
```

`agents/value_net.py`
```python
class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=128): ...
    def forward(self, obs):  # returns V(s)
        return net(obs).squeeze(-1)
```

`algo/returns.py`
```python
def discount_cumsum(rewards, gamma):
    # vectorized reward-to-go
```

`algo/reinforce.py`
```python
def collect_episodes(env, policy, episodes, gamma):
    # returns lists: states, actions, logps, rewards, episode_indices

def compute_objectives(logps, rewards, states, value_net=None, gamma=0.99,
                       use_rtg=True, use_baseline=True, normalize_adv=True):
    # returns actor_loss, critic_loss (or 0), diagnostics
```

`utils/tb.py`
```python
class TbLogger:
    def log_scalar(self, name, value, step): ...
```

`main.py`
- Parse args → seed → create env → init nets/optims.
- Training loop using `collect_episodes` + `compute_objectives`.
- Apply grad clip, step optimizers, log TB, evaluate, save checkpoints.

---

## Default Configs to Start

Baseline “strong” config
```
--use_rtg 1
--use_baseline 1
--normalize_adv 1
--entropy_coef 0.003
--grad_clip 1.0
--episodes_per_update 10
--actor_lr 3e-3
--critic_lr 1e-3
--gamma 0.99
--seed 1
```

Vanilla “control” config
```
--use_rtg 0
--use_baseline 0
--normalize_adv 0
--entropy_coef 0.0
--grad_clip 0
--episodes_per_update 10
--actor_lr 3e-3
--gamma 0.99
--seed 1
```

---

## What to Compare (quick wins)
- RTG vs full return (Phase 1 vs 2).
- Baseline on/off (Phase 2 vs 3).
- Advantage norm on/off (Phase 3 vs 4).
- Grad clip on/off (Phase 4).
- Entropy bonus 0 vs 0.003 (Phase 4).

Track: sample efficiency (updates-to-200), final performance, stability across seeds.
