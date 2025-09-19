# REINFORCE Algorithm: From Basics to Implementation

Reinforcement learning (RL) problems involve an agent interacting with an environment over a sequence of **states** (s₁,s₂,…), taking **actions** (a₁,a₂,…) and receiving **rewards** (r₁,r₂,…).  The goal of the agent is to learn a **policy** π_θ(a|s) parameterised by θ that maximises the expected sum of rewards.  In policy‑gradient methods like **REINFORCE** we do not derive a value function directly; instead we optimise the parameters of a stochastic policy using gradient ascent on the expected return.

## Trajectories and returns

An entire episode of interaction is called a **trajectory** τ=(s₁,a₁,s₂,a₂,…,s_T,a_T).  The total return of a trajectory, for finite‑horizon tasks, is often written as

`R(τ) = r₁ + r₂ + … + r_T.`

The agent’s objective can be written as an expectation over all possible trajectories that the current policy could produce:

`J(θ) = E_{τ∼π_θ}[ R(τ) ],`

where the expectation is taken with respect to the distribution over trajectories induced by the policy.  The Spinning Up notes clarify that we wish to maximise **expected return** and that the policy gradient provides a way to differentiate this objective:contentReference[oaicite:0]{index=0}.

To make this more concrete, denote by p_θ(τ) the probability of a trajectory under the policy.  It can be factored as

`p_θ(τ) = p(s₁) ⋅ ∏_{t=1}^{T} π_θ(a_t|s_t) ⋅ p(s_{t+1}|s_t,a_t),`

where p(s₁) is the initial state distribution and p(s_{t+1}|s_t,a_t) are the environment’s transition dynamics:contentReference[oaicite:1]{index=1}.  **Importantly, the dynamics and the initial‑state distribution do not depend on the policy parameters**:contentReference[oaicite:2]{index=2}, which means that the only terms containing θ in p_θ(τ) come from the policy itself.

Because the return R(τ) is a function of the state and reward sequence, it is fixed once a trajectory is given and does not explicitly depend on the policy.  We therefore write the objective as the integral over all trajectories:

`J(θ) = ∫ p_θ(τ) R(τ) dτ.`

This formulation expresses the expected return as an integral over the trajectory distribution:contentReference[oaicite:3]{index=3}.  It is the starting point for deriving the policy gradient.

## Deriving the REINFORCE gradient

To optimise J(θ) we need its gradient with respect to θ.  The derivation proceeds in a few algebraic steps, each of which has a clear intuitive interpretation.

### 1. Differentiate under the integral sign

The gradient of the objective can be expressed as

`∇_θ J(θ) = ∇_θ ∫ p_θ(τ) R(τ) dτ = ∫ ∇_θ p_θ(τ) R(τ) dτ.`

**In words:** we bring the gradient inside the integral since the return does not depend on θ.  The integrand now involves the gradient of the trajectory probability.

### 2. Apply the log‑derivative trick

Directly differentiating p_θ(τ) is inconvenient because it is a product of many probabilities.  The **log‑derivative trick** (or likelihood ratio trick) rewrites this derivative as

`∇_θ p_θ(τ) = p_θ(τ) ⋅ ∇_θ log p_θ(τ).`

Substituting into the integral yields

`∇_θ J(θ) = ∫ p_θ(τ) ∇_θ log p_θ(τ) R(τ) dτ = E_{τ∼p_θ}[ ∇_θ log p_θ(τ) ⋅ R(τ) ].`

**In words:** we have replaced the gradient of a probability with the probability times the gradient of its log.  This is useful because we can estimate expectations of ∇_θ log p_θ(τ) with samples.

### 3. Remove environment terms

The log probability of a trajectory can be expanded using the factorisation of p_θ(τ):

`log p_θ(τ) = log p(s₁) + ∑_{t=1}^{T} [ log p(s_{t+1}|s_t,a_t) + log π_θ(a_t|s_t) ].`

Because the initial state distribution and transition dynamics do **not** depend on θ, their gradients are zero:contentReference[oaicite:4]{index=4}.  Consequently

`∇_θ log p_θ(τ) = ∑_{t=1}^{T} ∇_θ log π_θ(a_t|s_t).`

**In words:** only the policy terms contribute to the gradient; the environment’s randomness does not.  This step is crucial because it removes unknown dynamics from the gradient expression:contentReference[oaicite:5]{index=5}.

### 4. Final policy gradient expression

Plugging the simplified gradient back into the expectation gives the **REINFORCE gradient**:

`∇_θ J(θ) = E_{τ∼π_θ}[ ∑_{t=1}^{T} ∇_θ log π_θ(a_t|s_t) ⋅ R(τ) ].`

This result shows that the gradient of expected return can be estimated by summing, over each time step, the gradient of the log policy multiplied by the **same total return** R(τ).  Spinning Up’s derivation arrives at the same formula:contentReference[oaicite:6]{index=6}.  It has two important properties:

* **Unbiased estimate.**  When we sample trajectories under π_θ and compute the sum above, the expectation equals the true gradient.  Thus we can perform stochastic gradient ascent to improve the policy.
* **High variance.**  Using the full return R(τ) for every time step often leads to noisy gradient estimates, especially for long trajectories.  Later sections address variance reduction techniques such as reward‑to‑go and baselines.

### 5. Sample‑based estimation

In practice we do not compute the expectation exactly; instead we sample N trajectories {τᵢ} by running the policy in the environment and compute the empirical mean

`ĝ = (1/N) ∑_{i=1}^{N} ∑_{t=1}^{T_i} ∇_θ log π_θ(a_t^i|s_t^i) ⋅ R(τᵢ),`

where T_i is the length of trajectory i.  This quantity is an unbiased estimator of the true gradient:contentReference[oaicite:7]{index=7}.

## Reward‑to‑go and baseline as variance reduction

The simple REINFORCE gradient uses the same total return R(τ) at every time step, giving each action credit or blame for the entire episode.  Two standard modifications reduce variance while keeping the gradient unbiased.

### Reward‑to‑go (step‑wise returns)

Instead of using the full return, we can replace R(τ) by the **reward‑to‑go** G_t = r_t + r_{t+1} + ⋯ + r_T, the sum of rewards from the current time step onward.  Because future rewards after time t do not depend on earlier actions, using G_t still gives an unbiased gradient estimator.  It reduces variance by giving each action credit only for what happens after it.

### Adding a baseline

The **expected grad–log–prob (EGLP) lemma** states that for any function b(s_t) depending only on the state, the expectation of ∇_θ log π_θ(a_t|s_t) ⋅ b(s_t) under the policy is zero:contentReference[oaicite:8]{index=8}.  Therefore we can subtract b(s_t) from the return without changing the expected value of the gradient:

`∇_θ J(θ) = E_{τ∼π_θ}[ ∑_{t=1}^{T} ∇_θ log π_θ(a_t|s_t) ⋅ (G_t – b(s_t))  ].`

Any such b is called a **baseline**:contentReference[oaicite:9]{index=9}.  The most common choice is the **on‑policy state value function** V^π(s_t), which measures the expected return from s_t under the current policy.  Spinning Up notes that using this baseline reduces the variance of the gradient estimator and leads to faster, more stable learning:contentReference[oaicite:10]{index=10}.  In practice V^π(s_t) is approximated by a neural network V_φ(s_t); its parameters are updated by minimising a mean squared error between predicted values and empirical returns:contentReference[oaicite:11]{index=11}.  When we subtract V(s_t) from the return, the term G_t – V(s_t) is called the **advantage** A(s_t,a_t), indicating how much better or worse the action was compared with the average outcome in that state.

## Pseudocode for REINFORCE without baseline

The following pseudocode outlines the REINFORCE algorithm in its simplest form (no reward‑to‑go or baseline), and maps each operation back to the mathematical concepts above.  It assumes access to an environment that can be reset and stepped, and a policy network that maps states to a probability distribution over actions.

1. **Initialise policy parameters** θ (for example, neural network weights).
   *Corresponds to starting with some policy π_θ we wish to improve.*
2. **For each update** (iteration of gradient ascent):
   1. **Collect trajectories:** run the current policy in the environment for K episodes.  For each episode record (s_t,a_t,r_t, log π_θ(a_t|s_t)).
      *This corresponds to sampling trajectories from p_θ(τ) to approximate the expectation E_{τ∼π_θ}[…].*
   2. **Compute returns:** for each trajectory, compute the total return R(τ) = r₁ + ⋯ + r_T.
      *This implements the return R(τ) used in the gradient formula.*
   3. **Compute the policy loss:** for each trajectory, sum the log probabilities of actions and multiply by the return: `Loss = -∑_t log π_θ(a_t|s_t) ⋅ R(τ)`.
      *The negative sign reflects performing gradient descent on the loss to achieve gradient ascent on J(θ); this term matches the gradient expression ∑ ∇_θ log π_θ(a_t|s_t) R(τ).*
   4. **Backpropagate and update:** compute the gradient of the loss with respect to θ and take a small step in the direction of the gradient.
      *This approximates updating θ ← θ + α ∇_θ J(θ).*

This version of REINFORCE is easy to implement but suffers from high variance because it assigns credit for the entire episode’s return to every action.

## Pseudocode for REINFORCE with reward‑to‑go and baseline

The variance of policy gradient estimates can be significantly reduced by using reward‑to‑go and a baseline.  The following pseudocode incorporates these modifications:

1. **Initialise policy parameters** θ and value function parameters φ.  The value function approximator V_φ(s) will be used as the baseline.
2. **For each update:**
   1. **Collect K trajectories** by running π_θ.  For each time step store (s_t,a_t,r_t, log π_θ(a_t|s_t)).
   2. **Compute reward‑to‑go:** for each trajectory and each time step compute G_t = r_t + r_{t+1} + … + r_T.
      *This replaces the full return with step‑wise returns, reducing variance.*
   3. **Evaluate baseline:** compute V_φ(s_t) for each recorded state.
      *These are the baseline estimates b(s_t) used to subtract from the returns.*
   4. **Compute advantages:** A_t = G_t – V_φ(s_t).  Optionally normalise advantages over the batch.
      *The advantage is the return‑to‑go minus the baseline; it measures how much better an action was compared with the expected outcome.*
   5. **Policy loss:** sum `-log π_θ(a_t|s_t) ⋅ A_t` over all time steps and episodes.  Add an entropy bonus term (weighted by β) if desired to encourage exploration.
   6. **Update policy parameters:** backpropagate the policy loss to update θ.
   7. **Value loss:** compute the mean squared error between G_t and V_φ(s_t) and update φ by gradient descent.
      *This trains the value network to approximate V^π(s), the average return from state s under the current policy:contentReference[oaicite:12]{index=12}.*

This enhanced version implements the unbiased gradient estimator with baseline:contentReference[oaicite:13]{index=13} and reduces variance.  In practice it is referred to as the **vanilla policy gradient** or **REINFORCE with baseline** and forms the foundation for more advanced actor–critic methods.

## Summary

The REINFORCE algorithm is a simple Monte‑Carlo policy gradient method: it estimates the gradient of the expected return by sampling trajectories and weighting the log‑probabilities of actions by the observed returns.  The derivation leverages the log‑derivative trick and the independence of environment dynamics:contentReference[oaicite:14]{index=14}, leading to an elegant expression for the policy gradient:contentReference[oaicite:15]{index=15}.  Reward‑to‑go and baselines are optional modifications that retain unbiasedness while reducing variance.  Using an on‑policy value function as a baseline is particularly effective:contentReference[oaicite:16]{index=16} and is standard in modern policy gradient implementations.
