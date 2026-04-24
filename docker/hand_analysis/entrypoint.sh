#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  exec python /app/runner.py "$@"
fi

if [[ -n "${INPUT_ARTIFACTS_PATH:-}" && -n "${RESULT_OUTPUT_PATH:-}" && -n "${STAGE_MANIFEST_PATH:-}" && -n "${OUTPUT_DIR:-}" ]]; then
  exec python /app/platform_runner.py
fi

MODE="${MODE:-video}"
BACKEND="${BACKEND:-${HAND_BACKEND:-yolo}}"
CONF="${CONF:-0.3}"
STEP="${STEP:-1}"
RESIZE="${RESIZE:-720}"
PREVIEW="${PREVIEW:-false}"
NO_BATCH="${NO_BATCH:-false}"
MAX_WORKERS="${MAX_WORKERS:-4}"

normalize_bool() {
  local value
  value="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "$value" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

case "$MODE" in
  video)
    : "${VIDEO_PATH:?VIDEO_PATH is required when MODE=video}"
    set -- --video "$VIDEO_PATH"
    if [[ -n "${OUTPUT_PATH:-}" ]]; then
      set -- "$@" --output "$OUTPUT_PATH"
    fi
    ;;
  episode)
    : "${EPISODE_DIR:?EPISODE_DIR is required when MODE=episode}"
    set -- --episode "$EPISODE_DIR"
    ;;
  *)
    echo "Unsupported MODE: $MODE (expected: video or episode)" >&2
    exit 2
    ;;
esac

set -- "$@" --backend "$BACKEND" --frame-step "$STEP"

if [[ "$BACKEND" == "yolo" ]]; then
  set -- "$@" --conf "$CONF" --resize "$RESIZE"
fi

if normalize_bool "$PREVIEW"; then
  set -- "$@" --preview
fi

if [[ "$BACKEND" == "vlm" ]]; then
  if [[ -n "${VLM_API_PROVIDER:-}" ]]; then
    set -- "$@" --vlm-provider "$VLM_API_PROVIDER"
  fi
  if [[ -n "${VLM_API_KEY:-}" ]]; then
    set -- "$@" --vlm-api-key "$VLM_API_KEY"
  fi
  if [[ -n "${DASHSCOPE_API_KEY:-}" ]]; then
    set -- "$@" --dashscope-api-key "$DASHSCOPE_API_KEY"
  fi
  if [[ -n "${ARK_API_KEY:-}" ]]; then
    set -- "$@" --ark-api-key "$ARK_API_KEY"
  fi
  if [[ -n "${VLM_MODEL:-}" ]]; then
    set -- "$@" --vlm-model "$VLM_MODEL"
  elif [[ -n "${VLM_HAND_MODEL:-}" ]]; then
    set -- "$@" --vlm-model "$VLM_HAND_MODEL"
  elif [[ -n "${VLM_DEFAULT_MODEL:-}" ]]; then
    set -- "$@" --vlm-model "$VLM_DEFAULT_MODEL"
  fi
  if normalize_bool "$NO_BATCH"; then
    set -- "$@" --no-batch
  fi
  set -- "$@" --max-workers "$MAX_WORKERS"
fi

exec python /app/runner.py "$@"
