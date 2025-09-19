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

# Support both env and env_id keys
ENV_ID=$(jq -er '.env // .env_id // empty' "$CONFIG_PATH" || true)
LR=$(read_json_default lr "3e-3")
HIDDEN=$(read_json_default hidden 128)
EPISODES_PER_UPDATE=$(read_json_default episodes_per_update 10)
MAX_UPDATES=$(read_json_default max_updates 500)
GAMMA=$(read_json_default gamma 0.99)
SEED=$(read_json_default seed 1)
TAG=$(read_json_default tag "")
LOG_DIR=$(read_json_default log_dir "reinforce/experiments/runs")
NORMALIZE_OBS=$(jq -er '.normalize_obs // false' "$CONFIG_PATH" 2>/dev/null || echo false)
USE_RTG=$(jq -er '.use_rtg // false' "$CONFIG_PATH" 2>/dev/null || echo false)

if [[ -z "$ENV_ID" ]]; then
  echo "[!] Missing 'env' (or 'env_id') in $CONFIG_PATH" >&2
  exit 1
fi

CMD=(python -m reinforce.main --env "$ENV_ID" \
  --lr "$LR" \
  --hidden "$HIDDEN" \
  --episodes-per-update "$EPISODES_PER_UPDATE" \
  --max-updates "$MAX_UPDATES" \
  --gamma "$GAMMA" \
  --seed "$SEED" \
  --log-dir "$LOG_DIR")

if [[ -n "$TAG" && "$TAG" != "null" ]]; then
  CMD+=(--tag "$TAG")
fi

if [[ "$NORMALIZE_OBS" == "true" ]]; then
  CMD+=(--normalize-obs)
fi

if [[ "$USE_RTG" == "true" ]]; then
  CMD+=(--use-rtg)
fi

echo "[run] ${CMD[*]}"
if [[ "$DRY_RUN" == "false" ]]; then
  exec "${CMD[@]}"
fi
