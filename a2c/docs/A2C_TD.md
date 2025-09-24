# From REINFORCE (with Baseline) to Advantage Actor–Critic (A2C, TD-based)

> **Prerequisite**: You’ve read the REINFORCE docs. This file starts with a recap of the basic foundations — trajectories as random variables and the overall objective of policy gradient methods — and then builds step by step into A2C (TD-based).

---

## 0) Recap: Trajectories and the policy gradient objective

A **trajectory** is a sequence of states, actions, and rewards:  

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T).
$$

Because the environment is stochastic and the policy $\pi_\theta(a|s)$ is stochastic, $\tau$ is a **random variable** with probability:  

$$
p_\theta(\tau) = p(s_0)\prod_{t=0}^{T-1} \pi_\theta(a_t|s_t)\,p(s_{t+1}\mid s_t,a_t).
$$

The **objective** in policy gradient methods is to maximize the expected return:  

$$
J(\theta) = \mathbb{E}_{\tau\sim p_\theta}[R(\tau)] \quad\text{where}\quad R(\tau)=\sum_{t=0}^{T-1} \gamma^t r_t.
$$

- **In words:** We want to tune policy parameters $\theta$ so that on average, sampled trajectories yield as much reward as possible.

Using the log-derivative trick, the **policy gradient theorem** gives:  

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim p_\theta}\Bigg[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\,A_t\Bigg],
$$

where $A_t$ is an **advantage signal** that tells us how good or bad an action was compared to expectation.

---

## 1) What changes conceptually? (Monte-Carlo → Bootstrapped TD)

- **REINFORCE + baseline** uses **Monte-Carlo** returns:  
  $\hat A_t = G_t - V(s_t)$, where $G_t = r_t + \gamma r_{t+1} + \cdots$.  
  - *Meaning*: “Judge an action by everything that happened **after** it.” Unbiased, but noisy.

- **A2C** swaps $G_t$ for a **1-step bootstrapped target**:  
  $\hat A_t = \underbrace{r_t + \gamma V(s_{t+1})}_{\text{one-step target}} - V(s_t)$.  
  - *Meaning*: “Judge an action by **what happened next** plus my *current* estimate of the future.” Lower variance; can update every step.

- **On-policy** stays the same: data must come from the **current policy**.

---

## 2) Starting point: REINFORCE with a baseline

From REINFORCE we had the policy gradient with a baseline:  

$$
\nabla_\theta J(\theta)
= \mathbb{E}\Bigg[\sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)\;\underbrace{(G_t - b(s_t))}_{\text{advantage}}\Bigg].
$$

- If we pick $b(s_t)=V^\pi(s_t)$, then **advantage** is $A^\pi(s_t,a_t)=G_t-V^\pi(s_t)$.  
- *In words*: compare outcome to “what was expected in this state.”

---

## 3) The A2C replacement: one-step TD advantage

We **approximate** the Monte-Carlo return with a **one-step** target:  

$$
G_t \approx r_t + \gamma V(s_{t+1}).
$$

Plugging this into the advantage gives the **TD error**:  

$$
\boxed{\;\hat A_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\;}
$$

> **Plain English**: If “reward + predicted next value” is bigger than “predicted current value,” then this action did **better than expected** (positive $\delta_t$); otherwise worse (negative $\delta_t$).

The actor update becomes:  

$$
\nabla_\theta J(\theta) \;\approx\; \mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t)\,\delta_t].
$$

---

## 4) Training the critic (value function)

We fit $V_\phi$ with **TD(0)** regression:  

$$
\boxed{\;L_{\text{critic}}(\phi) = \tfrac{1}{2}\big(r_t + \gamma (1-d_{t+1})V_\phi(s_{t+1}) - V_\phi(s_t)\big)^2\;}
$$

- $d_{t+1}$ is 1 if next state is a **true terminal**, else 0 (mask bootstrapping at terminals).  
- *In words*: Make the **current** value prediction match “reward + discounted next value.”

> **Coupling**: The actor’s signal $\delta_t$ depends on the critic. The critic is trained from the same batch. Both are updated **together** each iteration.

---

## 5) Bias–variance: why A2C is different

- **REINFORCE + baseline** (with Monte-Carlo $G_t$): **unbiased**, **high variance**.  
- **A2C** (with one-step TD): **lower variance** (relies on immediate outcomes + current value), but **introduces bias** if the critic is imperfect.  
- In practice, this variance reduction yields **faster & more stable** learning; bias shrinks as the critic improves.

---

## 6) A2C objective (actor + critic + entropy)

**Actor loss** (minimization form used in code):  

$$
L_{\text{actor}}(\theta) = -\,\mathbb{E}[\log \pi_\theta(a_t\mid s_t)\,\underbrace{\delta_t}_{\text{advantage}}]\; -\; c_e\,\mathbb{E}[H(\pi_\theta(\cdot\mid s_t))].
$$

**Critic loss**:  

$$
L_{\text{critic}}(\phi) = \tfrac{1}{2}\,\mathbb{E}[\big(r_t + \gamma (1-d_{t+1})V_\phi(s_{t+1}) - V_\phi(s_t)\big)^2].
$$

**Total loss** (to minimize):  

$$
L(\theta,\phi) = L_{\text{actor}} + c_v\,L_{\text{critic}}.
$$

- **Entropy** term (coef $c_e$) keeps the policy stochastic → better exploration, less premature collapse.  
- **Value coef** $c_v$ balances actor vs critic updates.

---

## 7) How A2C is run in practice (no code, just the flow)

1. **Roll out** with current policy for a short horizon $T$ in $N$ parallel envs → batch size $N\times T$.  
2. **Compute** $\delta_t$ for each step using current $V_\phi$.  
3. **Actor update**: maximize $\log \pi\,\delta_t$ (minimize its negative).  
4. **Critic update**: regress to the TD target.  
5. **Repeat** with freshly collected on-policy data.

> **Why short horizons?** With TD targets, we don’t need full episodes; we can update every few steps.

---

## 8) Implementation details that matter

- **On-policy requirement**: reuse of the same batch for many epochs is *not* safe (that’s PPO’s trick). Use each batch once.  
- **Shared backbone**: commonly share feature layers between actor and critic with two heads.  
- **Terminals vs time limits**: mask bootstrapping at true terminals; for time-limit truncation you may bootstrap (partial episode bootstrapping).  
- **Normalizing advantages**: optional but helpful—center to 0 mean, scale to unit std per batch.  
- **Gradient clipping**: clip global grad norm (e.g., 0.5–1.0) to stabilize.  
- **Entropy coef**: small (e.g., 1e-3 to 3e-3).  
- **Learning rates**: actor and critic can share or be separate; often Adam with 3e-4 to 3e-3.

---

## 9) What *exactly* changed vs REINFORCE + baseline

| Aspect | REINFORCE + baseline | A2C (TD-based) |
|---|---|---|
| Advantage | $G_t - V(s_t)$ | $r_t + \gamma V(s_{t+1}) - V(s_t)$ |
| Target type | Monte-Carlo (long horizon) | 1-step bootstrap |
| Variance | High | Lower |
| Bias | None (in expectation) | Possible (if critic imperfect) |
| Update timing | End of episode (or after full rollout) | Every step / short rollouts |
| Batch reuse | No (one pass) | No (one pass) |

> **Mental model:** *Same formula*, different **target**: replace $G_t$ by $r_t + \gamma V(s_{t+1})$.

---

## 10) Minimal pseudocode (conceptual)

```text
loop over updates:
  # rollout short segments (on-policy)
  collect (s_t, a_t, r_t, s_{t+1}, done) for t=0..T-1 across N envs

  # critic predictions
  V_t   = Vφ(s_t)
  V_tp1 = Vφ(s_{t+1}) * (1 - done)        # no bootstrap across true terminals

  # TD-error advantage
  delta_t = r_t + γ * V_tp1 - V_t

  # ACTOR: maximize logπ * delta  (minimize negative)
  L_actor  = -(logπθ(a_t|s_t) * delta_t).mean() - c_e * entropy(πθ(.|s_t)).mean()

  # CRITIC: TD regression
  L_critic = 0.5 * (r_t + γ * V_tp1 - V_t).pow(2).mean()

  loss = L_actor + c_v * L_critic
  backprop(loss), step optimizers
```

(*Note:* In code, detach $\delta_t$ from critic when used in the actor loss, so actor grads don’t flow into the critic.)

---

## 11) Sanity checks & diagnostics

- **Learning curves**: faster rise vs REINFORCE on the same env.  
- **Value MAE**: |TD target − V| should shrink.  
- **Entropy**: should decrease gradually (not collapse immediately).  
- **Grad norms**: stay bounded with clipping.  
- **On-policy**: confirm each batch is used once per update.

---

## 12) Takeaway

- A2C is **REINFORCE + baseline** where the long-horizon return $G_t$ is replaced by a **1-step TD target**.  
- This reduces variance, enables **per-step** updates, and typically learns faster—at the cost of bias when the critic is imperfect.  
- It’s the standard stepping stone to **GAE** (weighted sums of many n-step TD errors) and then **PPO** (safe batch reuse with a clipped surrogate).
