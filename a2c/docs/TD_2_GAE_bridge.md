# Bridging One-Step TD and Generalized Advantage Estimation (GAE)

This note connects the single-step temporal-difference (TD) update used in the current A2C trainer with the more general **Generalized Advantage Estimation (GAE)** that we plan to support. The goal is to provide both mathematical definitions and intuition so the transition from TD to GAE feels natural.

## 1. Recap: One-Step TD Advantage

In A2C with one-step TD, we estimate the advantage at time step *t* as

$$
A_t^{\text{TD}} = r_t + \gamma \, V_\phi(s_{t+1}) - V_\phi(s_t).
$$

- **Immediate reward (`r_t`)** captures what we just observed.
- **Bootstrapped value (`V_\phi(s_{t+1})`)** injects our critic’s estimate of the future.
- **Current value (`V_\phi(s_t)`)** serves as the baseline we’re comparing against.

This is a bias/variance compromise: a single bootstrap step keeps variance low but introduces bias if `V` is imperfect.

## 2. Beyond One Step: n-Step TD

We can extend TD to look ahead *n* steps before bootstrapping:

$$
G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k \, r_{t+k} + \gamma^{n} \, V_\phi(s_{t+n})
$$

The corresponding advantage estimate is $$`A_t^{(n)} = G_t^{(n)} - V_\phi(s_t)`$$.

- When `n = 1`, we recover the one-step TD formula.
- When `n` equals the remaining episode length, the bootstrap term vanishes and we obtain a Monte Carlo return with minimal bias but maximal variance.

Choosing `n` is essentially balancing bias (longer horizon reduces bias) vs. variance (longer horizon increases variance).

## 3. From n-Step TD to GAE

**Generalized Advantage Estimation** (GAE) averages these n-step advantage estimates with exponentially decaying weights:

$$
A_t^{\text{GAE}(\lambda)} = \sum_{n=1}^{\infty} (\gamma \lambda)^{n-1} \, \delta_t^{(n)}
$$

where the *n*-step TD error is defined recursively as

with the TD residuals

$$
\delta_t = r_t + \gamma \, V_\phi(s_{t+1}) - V_\phi(s_t),
\qquad
\delta_t^{(n)} = \sum_{k=0}^{n-1} (\gamma \lambda)^k \, \delta_{t+k}.
$$

In practice we compute GAE with a simple backward pass over the trajectory:

```
\text{for final step } T:\quad A_T = 0
\text{for } t < T:\quad A_t = \delta_t + \gamma \lambda (1 - \text{done}_t) \; A_{t+1}
```

This recursion accumulates (discounted) TD residuals until the end of the episode or truncation.

### Parameter intuition

- `λ = 0` collapses the recursion to the one-step TD error. So **GAE(γ, 0) = TD(1-step)**.
- `λ = 1` recovers a Monte Carlo-style estimator (sum of discounted rewards minus value), assuming full episode trajectories.
- Intermediate values interpolate smoothly between the two extremes, trading off bias and variance.

## 4. Bias–Variance Interpretation

| Setting        | Bias                    | Variance                 | Comment                             |
|----------------|-------------------------|--------------------------|-------------------------------------|
| TD (`λ=0`)     | High (uses bootstrap)   | Low (single future estimate) | Stable but may underfit long horizons |
| Monte Carlo (`λ=1`) | Low (true returns)      | High (no bootstrapping)        | Unstable but unbiased              |
| GAE (`0<λ<1`)  | Controlled (interpolates) | Moderated (averages TD errors) | Practical sweet spot for many tasks |

GAE is therefore a principled way to tune the bias/variance balance with a single hyperparameter λ.

## 5. Implementation Outline

To extend the current A2C trainer:

1. **Collect log-probs, rewards, values, and done flags** exactly as in the TD code. No extra data is needed.
2. **Compute TD residuals** `δ_t = r_t + γ V(s_{t+1}) - V(s_t)` (already done for single-step TD).
3. **Run the backward GAE recursion** per trajectory to produce `A_t^{GAE(λ)}`.
4. **Use these advantages for the policy loss**, optionally normalising them.
5. **Define the value target** as `A_t^{GAE} + V(s_t)` (or equivalently the λ-return) for the critic loss.

If `use_td=True` we stick with one-step TD. When `use_gae=True` we replace the advantage (and the value targets) with the λ-based versions. The same trainer infrastructure (masking, batching, logging) can be reused.

## 6. Limiting Cases & Sanity Checks

- `λ → 0`: advantages reduce to the current TD residuals, so behaviour should match the existing implementation.
- `λ → 1`: advantages converge toward Monte Carlo returns minus the baseline; we should observe higher variance but potentially lower bias.
- If `γλ` is small (e.g. λ = 0.95, γ = 0.99), contributions decay quickly—effectively weighting the most recent few steps more than distant ones.

## 7. Practical Tips

- **Normalisation**: Applying advantage normalisation is still helpful; it stabilises gradient magnitudes whether using TD or GAE.
- **Value targets**: When using GAE, ensure the critic is trained against the λ-returns (`A + V`). Otherwise, the policy and value networks optimise inconsistent objectives.
- **Diagnostics**: Log both `adv_mean/adv_std` and `td_mean/td_std` to understand how λ affects the spread and bias of the estimates.

## 8. Summary

GAE is an elegant generalisation of TD learning that averages multi-step advantages with exponentially decaying weights. It interpolates between the low-variance/high-bias one-step TD estimator and the high-variance/low-bias Monte Carlo estimator. By exposing a `λ` hyperparameter, we can smoothly tune this balance without rewriting the entire training loop.

This conceptual bridge should clarify how the existing TD implementation fits inside a broader family of estimators, and how small changes (adding the λ-recursion) will give us GAE in Phase 2.
