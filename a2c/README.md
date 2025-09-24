# A2C (Advantage Actor-Critic) Implementation

A clean, well-documented implementation of the Advantage Actor-Critic (A2C) algorithm for reinforcement learning, optimized for CartPole-v1 and other discrete action environments.

## 🎯 Features

- **Pure A2C Implementation**: Follows Stable Baselines3 design patterns
- **GAE (Generalized Advantage Estimation)**: Proper episode boundary handling
- **Vectorized Environments**: Parallel data collection from multiple environments
- **Deterministic Evaluation**: Separate evaluation with deterministic policy
- **TensorBoard Logging**: Comprehensive metrics and visualization
- **Configurable**: JSON-based configuration with bash script runner

## 🏗️ Code Architecture

### Core Components

```
a2c/
├── agents/
│   ├── __init__.py          # Agent exports
│   └── actor_critic.py      # Neural network architecture
├── cfgs/
│   └── cartpole_td.json     # Configuration file
├── scripts/
│   └── run_from_config.sh   # Bash script runner
├── main.py                  # Entry point and argument parsing
├── trainer.py               # Main training logic
└── README.md               # This file
```

## 🔧 Key Classes and Functions

### 1. `TrainerConfig`
Configuration dataclass containing all hyperparameters:
- **Environment**: `env_id`, `normalize_obs`
- **Training**: `learning_rate`, `max_updates`, `rollout_length`, `num_envs`
- **Algorithm**: `gamma`, `gae_lambda`, `value_loss_coef`, `entropy_coef`
- **Evaluation**: `eval_freq`, `n_eval_episodes`

### 2. `ActorCriticNet`
Neural network with shared backbone:
- **Shared layers**: 2-layer MLP (obs_dim → 128 → 128)
- **Policy head**: Linear layer outputting action logits
- **Value head**: Linear layer outputting state value
- **Output**: Categorical distribution and value prediction

### 3. `A2CTrainer`
Main training class handling:
- **Environment setup**: Vectorized environment creation
- **Rollout collection**: Step-based data gathering
- **GAE computation**: Advantage estimation with episode boundaries
- **Network updates**: Actor-critic loss computation and optimization
- **Evaluation**: Deterministic policy assessment

## 🧠 Algorithm Flow

### Training Loop (Simplified)
```python
for update in range(max_updates):
    # 1. Collect rollouts from parallel environments
    batch = collect_rollout()  # Shape: (rollout_length × num_envs, ...)

    # 2. Compute GAE advantages
    advantages = compute_gae_advantages(rewards, values, next_values, dones)
    returns = advantages + values

    # 3. Update policy and value function
    policy_loss = -(log_probs * advantages).mean()
    value_loss = ((predicted_values - returns)**2).mean()
    total_loss = policy_loss + value_loss_coef * value_loss

    # 4. Evaluate policy periodically
    if update % eval_freq == 0:
        eval_return = evaluate_policy()
```

### GAE Computation (Critical)
```python
def compute_gae_advantages(rewards, values, next_values, dones):
    advantages = torch.zeros_like(rewards)
    last_gae_lam = torch.zeros(num_envs)

    # Work backwards through time
    for step in reversed(range(rollout_length)):
        # TD error
        delta = rewards[step] + gamma * next_values[step] * (1 - dones[step]) - values[step]

        # GAE accumulation
        last_gae_lam = delta + gamma * gae_lambda * (1 - dones[step]) * last_gae_lam
        advantages[step] = last_gae_lam

        # CRITICAL: Reset at episode boundaries
        last_gae_lam = last_gae_lam * (1 - dones[step])

    return advantages
```

## 📊 Data Flow

### 1. Rollout Collection
- **Input**: Current observations from `num_envs` parallel environments
- **Process**: Execute policy for `rollout_length` steps
- **Output**: `RolloutBatch` with observations, actions, rewards, values, dones

### 2. Advantage Computation
- **Input**: Rollout batch data
- **Process**: GAE computation with proper episode boundary handling
- **Output**: Advantages and returns for policy/value updates

### 3. Network Update
- **Input**: Rollout batch and computed advantages
- **Process**: Forward pass, loss computation, backpropagation
- **Output**: Updated policy and value function parameters

### 4. Evaluation
- **Input**: Current policy parameters
- **Process**: Deterministic rollouts on single environment
- **Output**: Average episodic returns and statistics

## ⚙️ Configuration

### Key Hyperparameters (CartPole-v1)
```json
{
  "env": "CartPole-v1",
  "learning_rate": 0.0007,
  "rollout_length": 5,
  "num_envs": 4,
  "gae_lambda": 1.0,
  "value_loss_coef": 0.5,
  "eval_freq": 50
}
```

### Parameter Explanations
- **`rollout_length`**: Steps collected per environment before update (5 for CartPole)
- **`num_envs`**: Parallel environments for data collection (4 for CartPole)
- **`gae_lambda`**: GAE bias-variance tradeoff (1.0 = Monte Carlo)
- **`learning_rate`**: Shared RMSprop learning rate (0.0007 like SB3)

## 🚀 Usage

### Quick Start
```bash
# Run with default CartPole configuration
./a2c/scripts/run_from_config.sh a2c/cfgs/cartpole_td.json

# Run with custom parameters
python -m a2c.main --learning-rate 0.001 --max-updates 5000
```

### Success Criteria (CartPole-v1)
- **Target**: 195+ average return over 100 consecutive episodes
- **Typical training**: 1000-5000 updates to reach target
- **Current performance**: ~10-20 steps (early learning phase)

## 🐛 Common Issues and Solutions

### 1. **Poor Learning Performance**
- **Cause**: Episode boundary handling in GAE
- **Solution**: Ensure GAE resets at episode termination

### 2. **High Value Loss**
- **Cause**: Scale mismatch between predictions and targets
- **Solution**: Verify GAE computation and value target scaling

### 3. **Training Instability**
- **Cause**: Wrong hyperparameters or optimizer
- **Solution**: Use RMSprop with lr=0.0007, proper gradient clipping

## 📈 Monitoring Training

### Key Metrics
- **`ret_mean`**: Average return from training rollouts
- **`eval_ret`**: Average return from deterministic evaluation
- **`policy_loss`**: Actor loss magnitude
- **`value_loss`**: Critic loss magnitude
- **`H` (entropy)**: Policy randomness (decreases as learning progresses)

### Expected Learning Curve (CartPole)
```
Updates 1-100:    eval_ret ~10-20   (basic control)
Updates 100-500:  eval_ret ~50-100  (skill development)
Updates 500-2000: eval_ret ~195+    (target performance)
```

## 🔍 Implementation Details

### Critical Design Decisions
1. **Always use GAE**: Simplified from TD/MC choice for better performance
2. **Unified learning rate**: Single RMSprop optimizer like SB3
3. **Deterministic evaluation**: Argmax policy for consistent assessment
4. **Episode boundary handling**: Proper GAE reset at termination
5. **Vectorized rollouts**: Parallel data collection for efficiency

### Removed/Simplified Components
- ❌ Separate actor/critic learning rates → Unified learning rate
- ❌ TD vs GAE choice → Always GAE
- ❌ Returns-to-go computation → Only GAE+values
- ❌ Manual episode handling → Vectorized environment management