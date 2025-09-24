#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 path/to/config.json [--dry-run]" >&2
  exit 1
fi

CONFIG_PATH="$1"
DRY_RUN="false"
if [[ ${2-} == "--dry-run" ]]; then
  DRY_RUN="true"
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "[!] jq is required. Install with: brew install jq" >&2
  exit 1
fi

read_json_default() {
  local key="$1"; shift
  local default_val="${1-}"
  local val
  val=$(jq -er ".${key} // empty" "$CONFIG_PATH" || true)
  if [[ -z "$val" ]]; then
    echo "$default_val"
  else
    echo "$val"
  fi
}

ENV_ID=$(jq -er '.env // .env_id // empty' "$CONFIG_PATH" || true)
LEARNING_RATE=$(read_json_default learning_rate "7e-4")
# Backward compatibility
ACTOR_LR=$(read_json_default actor_lr "")
CRITIC_LR=$(read_json_default critic_lr "")
VALUE_LOSS_COEF=$(read_json_default value_loss_coef "0.5")
HIDDEN=$(read_json_default hidden 128)
ROLLOUT_LENGTH=$(read_json_default rollout_length 20)
NUM_ENVS=$(read_json_default num_envs 8)
MAX_UPDATES=$(read_json_default max_updates 500)
GAMMA=$(read_json_default gamma 0.99)
SEED=$(read_json_default seed 1)
TAG=$(read_json_default tag "")
LOG_DIR=$(read_json_default log_dir "a2c/experiments/runs")
NORMALIZE_OBS=$(jq -er '.normalize_obs // true' "$CONFIG_PATH" 2>/dev/null || echo true)
NORMALIZE_ADV=$(jq -er '.normalize_adv // false' "$CONFIG_PATH" 2>/dev/null || echo false)
ENTROPY_COEF=$(read_json_default entropy_coef "0.0")
GRAD_CLIP=$(read_json_default grad_clip "0.0")
USE_TD=$(jq -er '.use_td // true' "$CONFIG_PATH" 2>/dev/null || echo true)
USE_GAE=$(jq -er '.use_gae // false' "$CONFIG_PATH" 2>/dev/null || echo false)
GAE_LAMBDA=$(read_json_default gae_lambda "0.95")
EVAL_FREQ=$(read_json_default eval_freq "100")
N_EVAL_EPISODES=$(read_json_default n_eval_episodes "10")

if [[ -z "$ENV_ID" ]]; then
  echo "[!] Missing 'env' (or 'env_id') in $CONFIG_PATH" >&2
  exit 1
fi

CMD=(python -m a2c.main --env "$ENV_ID")

# Add learning rate (unified or backward compatible)
if [[ -n "$LEARNING_RATE" && "$LEARNING_RATE" != "null" ]]; then
  CMD+=(--learning-rate "$LEARNING_RATE")
elif [[ -n "$ACTOR_LR" && "$ACTOR_LR" != "null" ]]; then
  CMD+=(--actor-lr "$ACTOR_LR")
  [[ -n "$CRITIC_LR" && "$CRITIC_LR" != "null" ]] && CMD+=(--critic-lr "$CRITIC_LR")
fi

CMD+=(\
  --value-loss-coef "$VALUE_LOSS_COEF" \
  --hidden "$HIDDEN" \
  --rollout-length "$ROLLOUT_LENGTH" \
  --num-envs "$NUM_ENVS" \
  --max-updates "$MAX_UPDATES" \
  --gamma "$GAMMA" \
  --seed "$SEED" \
  --log-dir "$LOG_DIR" \
  --entropy-coef "$ENTROPY_COEF" \
  --grad-clip "$GRAD_CLIP" \
  --gae-lambda "$GAE_LAMBDA" \
  --eval-freq "$EVAL_FREQ" \
  --n-eval-episodes "$N_EVAL_EPISODES")

if [[ -n "$TAG" && "$TAG" != "null" ]]; then
  CMD+=(--tag "$TAG")
fi

if [[ "$NORMALIZE_OBS" == "true" ]]; then
  CMD+=(--normalize-obs)
else
  CMD+=(--no-normalize-obs)
fi

if [[ "$NORMALIZE_ADV" == "true" ]]; then
  CMD+=(--normalize-adv)
fi

if [[ "$USE_TD" != "true" ]]; then
  CMD+=(--no-td)
fi

if [[ "$USE_GAE" == "true" ]]; then
  CMD+=(--use-gae)
fi

echo "[run] ${CMD[*]}"
if [[ "$DRY_RUN" == "false" ]]; then
  exec "${CMD[@]}"
fi
