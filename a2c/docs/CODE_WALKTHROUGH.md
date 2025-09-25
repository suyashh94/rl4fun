# A2C Implementation: Complete Code Walkthrough

This document provides a detailed, line-by-line walkthrough of the A2C implementation, explaining how all components work together.

## 📋 Table of Contents

1. [Neural Network Architecture](#neural-network-architecture)
2. [Configuration and Data Structures](#configuration-and-data-structures)
3. [Environment Setup](#environment-setup)
4. [Rollout Collection](#rollout-collection)
5. [GAE Advantage Computation](#gae-advantage-computation)
6. [Network Updates](#network-updates)
7. [Evaluation System](#evaluation-system)
8. [Training Loop Integration](#training-loop-integration)

---

## 1. Neural Network Architecture

### File: `a2c/agents/actor_critic.py`

```python
class ActorCriticNet(nn.Module):
    """Shared MLP backbone with separate policy and value heads."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),     # 4 → 128 for CartPole
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),      # 128 → 128
            nn.ReLU(inplace=True),
        )
        # Separate heads for policy and value
        self.policy_head = nn.Linear(hidden, act_dim)  # 128 → 2 for CartPole
        self.value_head = nn.Linear(hidden, 1)         # 128 → 1

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        features = self.backbone(obs)                   # Extract features
        logits = self.policy_head(features)            # Policy logits
        value = self.value_head(features).squeeze(-1)  # State value
        dist = Categorical(logits=logits)              # Action distribution
        return dist, value
```

**Key Points:**
- **Shared backbone**: Both actor and critic use same feature representation
- **Categorical distribution**: For discrete action spaces (CartPole has 2 actions)
- **Value squeeze**: Removes last dimension to match expected shape

---

## 2. Configuration and Data Structures

### File: `a2c/trainer.py` (Lines 22-44)

```python
@dataclass
class TrainerConfig:
    """Complete configuration for A2C training."""
    # Environment
    env_id: str = "CartPole-v1"
    seed: int = 1
    normalize_obs: bool = True

    # Network architecture
    hidden: int = 128

    # Training parameters
    learning_rate: float = 7e-4          # Unified LR like SB3
    rollout_length: int = 20             # Steps per environment per update
    num_envs: int = 8                    # Parallel environments
    max_updates: int = 500               # Total training updates

    # Algorithm parameters
    gamma: float = 0.99                  # Discount factor
    gae_lambda: float = 0.95             # GAE bias-variance tradeoff
    value_loss_coef: float = 0.5         # Value loss weight
    entropy_coef: float = 0.0            # Exploration bonus
    grad_clip: float = 0.0               # Gradient norm clipping
    normalize_adv: bool = False          # Advantage normalization

    # Evaluation
    eval_freq: int = 100                 # Evaluate every N updates
    n_eval_episodes: int = 10            # Episodes per evaluation

    # Logging
    log_dir: str = "a2c/experiments/runs"
    tag: Optional[str] = None
```

### Data Structure: `RolloutBatch`

```python
@dataclass
class RolloutBatch:
    """Container for a batch of rollout data."""
    observations: Tensor    # Shape: (total_steps, obs_dim)
    actions: Tensor        # Shape: (total_steps,)
    rewards: Tensor        # Shape: (total_steps,)
    log_probs: Tensor      # Shape: (total_steps,) - stored for reference
    entropies: Tensor      # Shape: (total_steps,) - stored for reference
    values: Tensor         # Shape: (total_steps,) - value predictions
    next_values: Tensor    # Shape: (total_steps,) - for GAE computation
    dones: Tensor          # Shape: (total_steps,) - episode terminations
    rtg: Tensor           # Shape: (total_steps,) - placeholder (unused)
    masks: Tensor         # Shape: (total_steps,) - valid step mask
    lengths: Tensor       # Shape: (num_sequences,) - sequence lengths
    raw_returns: Tensor   # Shape: (num_episodes,) - episode returns
```

**Tensor Shapes Explained:**
- `total_steps = rollout_length × num_envs` (e.g., 5 × 4 = 20)
- All tensors are flattened for batch processing
- `next_values` computed as `[values[1:], bootstrap_value]`

---

## 3. Environment Setup

### File: `a2c/trainer.py` (Lines 145-165)

```python
def _setup_environment(self) -> Tuple[gym.Env, int, int]:
    """Create vectorized environments for parallel data collection."""

    # Create environment factory functions
    def make_single_env(rank: int):
        def _init():
            return make_env(self.cfg.env_id, self.seed + rank, self.cfg.normalize_obs)
        return _init

    # Create num_envs parallel environments
    envs = [make_single_env(i) for i in range(self.cfg.num_envs)]
    env = SyncVectorEnv(envs)  # Synchronous vectorization

    # Extract space information
    obs_shape = env.single_observation_space.shape  # (4,) for CartPole
    act_dim = env.single_action_space.n              # 2 for CartPole
    obs_dim = obs_shape[0]                           # 4 for CartPole

    return env, obs_dim, act_dim
```

**Key Points:**
- **Different seeds**: Each environment gets `seed + rank` for diversity
- **SyncVectorEnv**: Runs environments sequentially (simpler than async)
- **Space introspection**: Extracts dimensions for network initialization

---

## 4. Rollout Collection

### File: `a2c/trainer.py` (Lines 179-289)

```python
def _collect_rollout(self) -> RolloutBatch:
    """Collect rollout_length steps from num_envs parallel environments."""

    # Initialize or use existing environment states
    if self._current_obs is None:
        self._current_obs, _ = self.env.reset()

    # Pre-allocate storage arrays
    T, N = self.cfg.rollout_length, self.cfg.num_envs
    obs_storage = np.zeros((T, N, self.obs_dim), dtype=np.float32)
    actions_storage = np.zeros((T, N), dtype=np.int64)
    rewards_storage = np.zeros((T, N), dtype=np.float32)
    dones_storage = np.zeros((T, N), dtype=np.float32)
    values_storage = np.zeros((T, N), dtype=np.float32)
    logp_storage = np.zeros((T, N), dtype=np.float32)
    entropy_storage = np.zeros((T, N), dtype=np.float32)

    # Episode tracking
    episode_rewards = np.zeros(N, dtype=np.float32)
    episode_lengths = np.zeros(N, dtype=np.int32)
    finished_episodes = []

    # Collect rollout_length steps
    for step in range(T):
        # Store current observations
        obs_storage[step] = self._current_obs

        # Get policy outputs
        obs_tensor = torch.as_tensor(self._current_obs, dtype=torch.float32, device=self.device)
        dist, values = self.model(obs_tensor)
        actions = dist.sample()                    # Sample actions

        # Store policy outputs (detached for storage)
        with torch.no_grad():
            logp = dist.log_prob(actions)
            entropy = dist.entropy()
            actions_storage[step] = actions.cpu().numpy()
            values_storage[step] = values.cpu().numpy()
            logp_storage[step] = logp.cpu().numpy()
            entropy_storage[step] = entropy.cpu().numpy()

        # Execute actions in environment
        next_obs, rewards, dones, truncateds, _ = self.env.step(actions.cpu().numpy())
        terminated = np.logical_or(dones, truncateds)

        # Store environment outputs
        rewards_storage[step] = rewards
        dones_storage[step] = terminated.astype(np.float32)

        # Track episode statistics
        episode_rewards += rewards
        episode_lengths += 1

        # Record finished episodes
        for env_idx in range(N):
            if terminated[env_idx]:
                finished_episodes.append({
                    'reward': float(episode_rewards[env_idx]),
                    'length': int(episode_lengths[env_idx])
                })
                episode_rewards[env_idx] = 0.0
                episode_lengths[env_idx] = 0

        # Update current observations
        self._current_obs = next_obs

    # Bootstrap final values
    obs_tensor = torch.as_tensor(self._current_obs, dtype=torch.float32, device=self.device)
    with torch.no_grad():
        _, bootstrap_values = self.model(obs_tensor)
    bootstrap_values = bootstrap_values.cpu().numpy()

    # Convert to tensors and compute next_values
    values = torch.as_tensor(values_storage, dtype=torch.float32, device=self.device)
    bootstrap_values_tensor = torch.as_tensor(bootstrap_values, dtype=torch.float32, device=self.device)
    next_values = torch.cat([values[1:], bootstrap_values_tensor.unsqueeze(0)], dim=0)

    # Flatten all tensors for batch processing
    # Shape: (T, N) → (T*N,)
    observations = torch.as_tensor(obs_storage, dtype=torch.float32, device=self.device).view(-1, self.obs_dim)
    actions = torch.as_tensor(actions_storage, dtype=torch.long, device=self.device).view(-1)
    rewards = torch.as_tensor(rewards_storage, dtype=torch.float32, device=self.device).view(-1)
    # ... (similar for other tensors)

    return RolloutBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        # ... (other fields)
        raw_returns=torch.tensor([ep['reward'] for ep in finished_episodes])
    )
```

**Critical Details:**
- **Stateful collection**: `self._current_obs` persists across rollouts
- **Bootstrap values**: Final state values for GAE computation
- **Episode boundaries**: Tracked within rollouts, don't reset environments
- **Flattening**: 2D tensors (T, N) flattened to 1D (T*N) for batch processing

### **Episode Termination in Vectorized Environments: Detailed Example**

**Key Question**: When one environment terminates, do we reset everything?
**Answer**: No! Each environment runs independently, and the vectorized environment automatically handles resets.

#### **Example Scenario:**
```
Configuration: rollout_length=5, num_envs=4 (Envs A, B, C, D)

Initial State:
- Env A: CartPole episode ongoing (step 15 of current episode)
- Env B: CartPole episode ongoing (step 3 of current episode)
- Env C: CartPole episode ongoing (step 8 of current episode)
- Env D: CartPole episode ongoing (step 1 of current episode)
```

#### **Step-by-Step Rollout Collection:**

**Rollout Step 0:**
```python
# Current observations from all 4 environments
current_obs = [obs_A, obs_B, obs_C, obs_D]  # Shape: (4, 4)

# Policy forward pass on all observations simultaneously
actions = model(current_obs).sample()  # Shape: (4,) e.g., [1, 0, 1, 0]

# Environment step - EACH ENV PROCESSES ITS OWN ACTION
next_obs, rewards, dones, truncated, _ = env.step(actions)

# Results:
next_obs = [obs_A', obs_B', obs_C', obs_D']  # Shape: (4, 4)
rewards = [1.0, 1.0, 1.0, 1.0]              # All envs still running
dones = [False, False, False, False]          # No terminations yet
```

**Rollout Step 1:**
```python
current_obs = next_obs  # From previous step
actions = model(current_obs).sample()  # e.g., [0, 1, 1, 0]

# Environment step
next_obs, rewards, dones, truncated, _ = env.step(actions)

# Results - ENV B TERMINATES:
next_obs = [obs_A'', new_obs_B, obs_C'', obs_D'']  # B got reset obs!
rewards = [1.0, 1.0, 1.0, 1.0]
dones = [False, True, False, False]    # Env B terminated
truncated = [False, False, False, False]

# What happened to Env B:
# 1. CartPole episode ended (pole fell over)
# 2. SyncVectorEnv automatically called env_B.reset()
# 3. new_obs_B is the initial observation of a fresh episode
# 4. We record that this step had done=True for env B
```

**Rollout Steps 2-4:**
```python
# Env B continues from its NEW episode while others continue their original episodes
# Each environment runs completely independently
```

#### **Critical Points:**

1. **No Manual Resets**: We never call `env.reset()` during rollout collection. The `SyncVectorEnv` handles this automatically.

2. **Independent Episodes**: Each environment can be at different stages of different episodes within the same rollout.

3. **Done Flags**: The `dones` array tells us which environments terminated at each step, crucial for GAE computation.

4. **Automatic Reset**: When `env.step()` returns `done=True` for an environment, that environment's `next_obs` is already the initial observation of a new episode.

#### **Data Storage:**

```python
# What gets stored for our 5-step rollout:
obs_storage[0] = [obs_A, obs_B, obs_C, obs_D]           # Step 0 states
obs_storage[1] = [obs_A', obs_B', obs_C', obs_D']       # Step 1 states
obs_storage[2] = [obs_A'', new_obs_B, obs_C'', obs_D''] # Step 2: B restarted

dones_storage[0] = [0, 0, 0, 0]  # No terminations
dones_storage[1] = [0, 1, 0, 0]  # Env B terminated
dones_storage[2] = [0, 0, 0, 0]  # Env B is in new episode
```

---

## 5. GAE Advantage Computation

### File: `a2c/trainer.py` (Lines 291-338)

```python
def _compute_gae_advantages(self, rewards: Tensor, values: Tensor, next_values: Tensor, dones: Tensor) -> Tensor:
    """Compute Generalized Advantage Estimation with proper episode boundary handling."""

    # Reshape flattened tensors back to 2D for proper iteration
    T, N = self.cfg.rollout_length, self.cfg.num_envs
    rewards_2d = rewards.view(T, N)
    values_2d = values.view(T, N)
    next_values_2d = next_values.view(T, N)
    dones_2d = dones.view(T, N)

    # Initialize advantage storage and GAE accumulator
    advantages_2d = torch.zeros_like(rewards_2d)
    last_gae_lam = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)

    # Work backwards through time steps
    for step in reversed(range(T)):
        # Compute TD error: δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)
        next_non_terminal = 1.0 - dones_2d[step]  # 0 if episode ended
        delta = rewards_2d[step] + self.cfg.gamma * next_values_2d[step] * next_non_terminal - values_2d[step]

        # GAE recursion: Âₜ = δₜ + γλ(1-dₜ)Âₜ₊₁
        last_gae_lam = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last_gae_lam
        advantages_2d[step] = last_gae_lam

        # CRITICAL: Reset GAE accumulator at episode boundaries
        # This prevents advantage leakage across episodes
        last_gae_lam = last_gae_lam * next_non_terminal

    return advantages_2d.view(-1)  # Flatten back to 1D
```

**GAE Mathematics:**
- **TD Error**: `δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)`
- **GAE Formula**: `Âₜ = δₜ + γλ(1-dₜ)Âₜ₊₁`
- **Episode Reset**: `last_gae_lam *= (1 - done)` prevents cross-episode contamination

### **GAE Computation with Episode Boundaries: Concrete Example**

Let's trace through GAE computation for our example where Env B terminates:

#### **Data Setup:**
```python
# Our rollout data (5 steps, 4 envs):
rewards_2d = [
    [1.0, 1.0, 1.0, 1.0],  # Step 0: all envs running
    [1.0, 1.0, 1.0, 1.0],  # Step 1: Env B terminates after this reward
    [1.0, 1.0, 1.0, 1.0],  # Step 2: Env B is in new episode
    [1.0, 1.0, 1.0, 1.0],  # Step 3
    [1.0, 1.0, 1.0, 1.0],  # Step 4
]

dones_2d = [
    [0, 0, 0, 0],  # Step 0: no terminations
    [0, 1, 0, 0],  # Step 1: Env B terminates
    [0, 0, 0, 0],  # Step 2: Env B is in new episode
    [0, 0, 0, 0],  # Step 3
    [0, 0, 0, 0],  # Step 4
]

# Values predicted by critic at each step
values_2d = [
    [15.0, 3.0, 8.0, 1.0],  # Step 0: higher values = expect longer episodes
    [16.0, 2.0, 9.0, 2.0],  # Step 1
    [17.0, 5.0, 10.0, 3.0], # Step 2: Env B has reset, new value estimate
    [18.0, 4.0, 11.0, 4.0], # Step 3
    [19.0, 3.0, 12.0, 5.0], # Step 4
]

# Next values (shifted + bootstrap)
next_values_2d = [
    [16.0, 2.0, 9.0, 2.0],   # Step 0 → Step 1 values
    [17.0, 5.0, 10.0, 3.0],  # Step 1 → Step 2 values (B restarted!)
    [18.0, 4.0, 11.0, 4.0],  # Step 2 → Step 3 values
    [19.0, 3.0, 12.0, 5.0],  # Step 3 → Step 4 values
    [20.0, 6.0, 13.0, 6.0],  # Step 4 → Bootstrap values
]
```

#### **GAE Computation (Working Backwards):**

```python
# Initialize GAE accumulator for each environment
last_gae_lam = [0.0, 0.0, 0.0, 0.0]  # One per environment

# Step 4 (t=4):
next_non_terminal = [1.0, 1.0, 1.0, 1.0]  # All continuing
delta = [1.0 + 0.99*20.0 - 19.0, 1.0 + 0.99*6.0 - 3.0, ...] = [1.8, 3.94, 1.87, 1.94]
last_gae_lam = delta + 0.99*1.0*[1,1,1,1]*last_gae_lam = [1.8, 3.94, 1.87, 1.94]
advantages[4] = [1.8, 3.94, 1.87, 1.94]

# Step 3 (t=3):
next_non_terminal = [1.0, 1.0, 1.0, 1.0]  # All continuing
delta = [1.0 + 0.99*19.0 - 18.0, 1.0 + 0.99*3.0 - 4.0, ...] = [1.81, -0.03, 1.89, 1.95]
last_gae_lam = [1.81, -0.03, 1.89, 1.95] + 0.99*1.0*[1.8, 3.94, 1.87, 1.94]
             = [3.59, 3.87, 3.74, 3.87]
advantages[3] = [3.59, 3.87, 3.74, 3.87]

# Step 2 (t=2):
next_non_terminal = [1.0, 1.0, 1.0, 1.0]  # All continuing
delta = [...] # Similar calculation
last_gae_lam = [...] # Accumulated advantages

# Step 1 (t=1) - THE CRITICAL STEP:
next_non_terminal = [1.0, 0.0, 1.0, 1.0]  # Env B terminated!
delta = [1.0 + 0.99*17.0 - 16.0, 1.0 + 0.99*5.0 - 2.0, ...] = [1.83, 3.95, ...]

# For Env B: next_non_terminal = 0.0, so:
last_gae_lam[B] = delta[B] + 0.99*1.0*0.0*previous_gae = 3.95 + 0 = 3.95
# Env B's GAE advantage is just its TD error - no future influence!

# For other envs: next_non_terminal = 1.0, so they accumulate normally
advantages[1] = [..., 3.95, ...] # Env B gets clean advantage

# CRITICAL RESET:
last_gae_lam = last_gae_lam * next_non_terminal
             = [..., 3.95*0.0, ...] = [..., 0.0, ...]
# Env B's accumulator is reset to 0!

# Step 0 (t=0):
# Now Env B starts fresh with last_gae_lam[B] = 0.0
# This prevents Episode 1 advantages from contaminating Episode 2
```

#### **Why Episode Boundaries Matter:**

**Without Reset:**
```
Episode 1: [s₁, s₂, s₃] → done=1, advantages = [A₁, A₂, A₃]
Episode 2: [s₄, s₅] → ongoing, advantages = [A₄+leak, A₅+leak]
```
- Future rewards from Episode 2 would incorrectly influence Episode 1 advantages
- Episode 1 advantages would be inflated by Episode 2's expected returns

**With Reset (Our Implementation):**
```
Episode 1: [s₁, s₂, s₃] → done=1, advantages = [A₁, A₂, A₃] ✓ Clean
Episode 2: [s₄, s₅] → ongoing, advantages = [A₄, A₅] ✓ Independent
```
- Each episode gets its own clean advantage computation
- No cross-episode contamination

---

## 6. Network Updates

### File: `a2c/trainer.py` (Lines 403-448)

```python
def _update(self, batch: RolloutBatch, _: int = 0) -> Dict[str, float]:
    """Update actor-critic network using collected rollout data."""

    # Prepare batch processing
    masks = batch.masks                          # Valid step mask
    total_steps = masks.sum().clamp_min(1.0)    # Avoid division by zero

    # Recompute policy outputs (required for gradient computation)
    dist, values_pred = self.model(batch.observations)
    log_probs = dist.log_prob(batch.actions)    # Fresh log probabilities
    entropies = dist.entropy()                   # Fresh entropies

    # Compute advantages using GAE
    advantages = self._compute_gae_advantages(
        batch.rewards, batch.values, batch.next_values, batch.dones
    )
    returns = advantages + batch.values          # Returns = Advantages + Values (SB3 style)
    advantage_policy = advantages.detach()       # Detach for policy gradient

    # Optional advantage normalization
    if self.cfg.normalize_adv:
        advantage_policy = self._maybe_normalise_signal(advantage_policy, masks)

    # Compute losses

    # Actor loss: Policy gradient with advantages
    policy_loss = -((log_probs * advantage_policy * masks).sum() / total_steps)

    # Critic loss: MSE between predicted values and returns
    value_loss = ((values_pred - returns.detach()).pow(2) * masks).sum() / total_steps
    value_loss = 0.5 * value_loss

    # Total loss
    loss = policy_loss + self.cfg.value_loss_coef * value_loss
    loss = self._apply_entropy_bonus(loss, entropies, masks, total_steps)

    # Optimization step
    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = self._apply_grad_clip()
    self.optimizer.step()

    # Return metrics
    return {
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "total_loss": float(loss.detach().cpu().item()),
        "policy_grad_norm": grad_norm,
        # ... (additional metrics)
    }
```

**Loss Functions Explained:**

1. **Policy Loss** (Actor):
   ```
   L_π = -𝔼[log π(aₜ|sₜ) × Âₜ]
   ```
   - Increases probability of actions with positive advantages
   - Decreases probability of actions with negative advantages

2. **Value Loss** (Critic):
   ```
   L_V = 𝔼[(V(sₜ) - Rₜ)²]
   ```
   - MSE between predicted values and actual returns
   - Teaches critic to predict future rewards accurately

3. **Total Loss**:
   ```
   L = L_π + c₁ × L_V + c₂ × H[π]
   ```
   - `c₁ = value_loss_coef` (default: 0.5)
   - `c₂ = entropy_coef` (default: 0.0)

---

## 7. Evaluation System

### File: `a2c/trainer.py` (Lines 349-420)

```python
def evaluate_policy(self, n_episodes: int = None) -> Dict[str, float]:
    """Evaluate current policy using deterministic actions."""

    n_episodes = n_episodes or self.cfg.n_eval_episodes

    # Create separate evaluation environment
    eval_env = make_env(self.cfg.env_id, self.seed + 999, self.cfg.normalize_obs)

    episode_returns = []
    episode_lengths = []

    # Switch to evaluation mode
    self.model.eval()

    try:
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            terminated = False
            truncated = False

            # Run full episode
            while not (terminated or truncated):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    dist, _ = self.model(obs_tensor)
                    # DETERMINISTIC: Use argmax instead of sampling
                    action = dist.probs.argmax(dim=-1)

                obs, reward, terminated, truncated, _ = eval_env.step(action.item())
                episode_reward += reward
                episode_length += 1

            episode_returns.append(episode_reward)
            episode_lengths.append(episode_length)

    finally:
        eval_env.close()
        self.model.train()  # Switch back to training mode

    # Compute statistics
    return {
        'eval_return_mean': float(np.mean(episode_returns)),
        'eval_return_std': float(np.std(episode_returns)),
        'eval_return_min': float(np.min(episode_returns)),
        'eval_return_max': float(np.max(episode_returns)),
        'eval_length_mean': float(np.mean(episode_lengths)),
    }
```

**Why Deterministic Evaluation?**
- **Consistent measurement**: Removes sampling noise
- **True policy assessment**: Shows what policy "really" learned
- **Reproducible**: Same policy state → same evaluation results

**Training vs Evaluation:**
- **Training**: Stochastic policy (sample from distribution)
- **Evaluation**: Deterministic policy (argmax selection)

---

## 8. Training Loop Integration

### File: `a2c/trainer.py` (Lines 117-149)

```python
def train(self) -> TrainingHistory:
    """Main training loop integrating all components."""

    ret_ma: Optional[float] = None
    best_ret_mean = -float("inf")

    for update in range(1, self.cfg.max_updates + 1):
        # 1. COLLECT: Gather experience from environments
        batch = self._collect_rollout()

        # 2. UPDATE: Train actor-critic network
        metrics = self._update(batch, update)

        # 3. LOG: Record training statistics
        ret_ma, stats = self._handle_logging(batch, metrics, update, ret_ma)
        best_ret_mean = max(best_ret_mean, stats["ret_mean"])

        # 4. EVALUATE: Assess policy performance
        if update % self.cfg.eval_freq == 0 or update == 1:
            eval_stats = self.evaluate_policy()
            self.history.eval_returns.append(eval_stats['eval_return_mean'])
            self.history.eval_updates.append(update)

            # Log to TensorBoard
            self.tb.log_scalar("eval/return_mean", eval_stats['eval_return_mean'], update)
            # ... (other eval metrics)

        # 5. PRINT: Display progress
        if update % 10 == 0 or update == 1:
            eval_info = ""
            if update % self.cfg.eval_freq == 0 or update == 1:
                eval_info = f" | eval_ret {eval_stats['eval_return_mean']:7.2f}"

            msg = (
                f"update {update:4d} | ret_mean {stats['ret_mean']:7.2f} | "
                f"ret_ma {stats['ret_ma']:7.2f} | policy {metrics['policy_loss']:.4f} | "
                f"value {metrics['value_loss']:.4f} | H {stats['avg_entropy']:.3f}{eval_info}"
            )
            print(msg)

            # 6. CHECK SUCCESS: CartPole-v1 solved criterion
            if len(self.history.eval_returns) >= 2:
                recent_eval_mean = np.mean(self.history.eval_returns[-2:])
                if recent_eval_mean >= 195.0:
                    print(f"\\n🎉 CartPole-v1 SOLVED! Average return: {recent_eval_mean:.2f} >= 195.0")
                    print(f"Agent reached target performance at update {update}")

    return self.history
```

**Training Loop Flow:**
```
Initialize → [Collect → Update → Evaluate → Log] × max_updates → Save Results
     ↓              ↓        ↓        ↓       ↓
Environment  → Experience → Network → Policy → Metrics
   Setup         Batch      Update    Assessment  Storage
```

**Key Integration Points:**
- **State persistence**: `self._current_obs` carries between rollouts
- **Evaluation timing**: Every `eval_freq` updates + first update
- **Success detection**: Automatic CartPole-v1 solving detection
- **Logging coordination**: TensorBoard + console + history tracking

---

## 🔍 Data Flow Summary

### Complete Pipeline:
```
1. Environment Reset → Initial observations
2. Policy Forward Pass → Actions, log_probs, values, entropies
3. Environment Step → Next observations, rewards, dones
4. Bootstrap Computation → Final state values
5. GAE Computation → Advantages with episode boundaries
6. Loss Calculation → Policy loss + Value loss + Entropy bonus
7. Backpropagation → Gradient computation and parameter update
8. Evaluation → Deterministic policy assessment
```

### Tensor Transformations:
```
Collection:    (rollout_length, num_envs) → Flattened (total_steps,)
GAE:           Flattened → Reshaped → GAE → Flattened
Network:       Flattened tensors → Forward pass → Loss computation
Evaluation:    Single environment → Full episodes → Statistics
```

## 🎯 **Episode Termination Summary**

### **Key Takeaways:**

1. **Automatic Environment Management**:
   - `SyncVectorEnv` automatically resets terminated environments
   - No manual `env.reset()` calls needed during rollout collection
   - Each environment operates independently

2. **Episode Boundary Tracking**:
   - `dones` array tracks which environments terminated at each step
   - Episode statistics are recorded but don't interrupt data collection
   - Rollouts can span multiple episode boundaries per environment

3. **GAE with Episode Boundaries**:
   - GAE accumulator (`last_gae_lam`) is reset when `done=True`
   - Prevents advantage leakage across episode boundaries
   - Critical for correct advantage computation in short episodes

4. **Data Consistency**:
   - All rollout data is collected regardless of episode terminations
   - Episode boundaries are handled in post-processing (GAE computation)
   - This design enables efficient vectorized training

### **Common Misconceptions:**

❌ **Wrong**: "When one environment terminates, we reset all environments"
✅ **Correct**: Each environment runs independently, only the terminated one resets

❌ **Wrong**: "Episode boundaries break the rollout"
✅ **Correct**: Rollouts continue uninterrupted, episode boundaries are handled in GAE

❌ **Wrong**: "We need to wait for all environments to terminate"
✅ **Correct**: We collect fixed-length rollouts regardless of episode states

This walkthrough covers the complete A2C implementation, showing how each component integrates to create a working reinforcement learning agent. The key innovations in this implementation are proper GAE episode boundary handling and deterministic evaluation for robust performance assessment.