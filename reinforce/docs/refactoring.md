# Reinforce Refactor Plan

## Goal
Refactor the REINFORCE implementation to use a unified `ReinforceTrainer` class that supports all Phase 1–4 features (reward-to-go, baseline, advantage normalisation, entropy bonus, gradient clipping) with batched experience handling, while standardising terminology ("REINFORCE with baseline" vs "actor-critic"). Provide automation to run and visualise all four phases sequentially.

## Tasks

1. **Audit current code and docs**
   - Locate functions, variables, and doc strings that reference "actor-critic" and determine appropriate replacements (e.g. `reinforce_with_baseline`).
   - Catalogue data collection utilities (`collect_episodes`, `collect_episodes_steps`) and identify overlapping logic for consolidation.

2. **Design `ReinforceTrainer` API**
   - Define configuration inputs (env factory, policy/value nets, batching flags, Phase feature toggles).
   - Plan internal workflow: rollout collection → tensor batching with padding + masks → loss computation per phase → optimisation + logging hooks.
   - Decide on dependency injection for logger/metrics to keep CLI and script integration straightforward.

3. **Implement unified data collection & batching**
   - Create a single environment interaction method that records observations, actions, rewards, log-probs, and entropies for each episode.
   - Convert variable-length trajectories into padded tensors (max length = longest episode in batch) and build masks to ignore padded steps in loss/metrics.
   - Ensure observations and value targets are batched for baseline usage; maintain reward-to-go cache for reuse across losses.

4. **Refactor losses into trainer methods**
   - Reimplement vanilla, reward-to-go, and baseline pathways inside `ReinforceTrainer`, reusing batched tensors.
   - Integrate advantage normalisation, entropy bonus, and gradient clipping without duplicating code paths.
   - Remove / deprecate standalone functions from `reinforce/algo/reinforce.py` once functionality is encapsulated.

5. **Update CLI entry point (`reinforce/main.py`)**
   - Replace the free-function training loop with instantiation and invocation of `ReinforceTrainer.train()`.
   - Simplify argument parsing if possible, ensuring config compatibility and preserving metadata logging.

6. **Adapt scripts & configs**
   - Ensure `scripts/run_from_config.sh` forwards all required flags to the new trainer interface.
   - Review existing JSON configs, adjusting keys as needed after renaming.

7. **Develop phase comparison runner**
   - Implement a script that launches the four phases sequentially (pure REINFORCE, reward-to-go, baseline, baseline + Phase 4 stabilisers) using consistent seeds.
   - Aggregate metrics (avg logp, return moving average, entropy) and produce Matplotlib plots: one figure per metric comparing the four runs.
   - Provide CLI options for output directories and whether to reuse existing run data.

8. **Documentation updates**
   - Refresh `reinforce/docs/REINFORCE.md` (and other affected docs) to align terminology and describe the trainer architecture and batching approach.
   - Add instructions for the new comparison script and interpretation of plots.

9. **Validation & cleanup**
   - Run targeted smoke tests (short runs per phase) to verify numerical parity and stability.
   - Remove or archive obsolete helpers after the refactor and ensure imports are updated.

