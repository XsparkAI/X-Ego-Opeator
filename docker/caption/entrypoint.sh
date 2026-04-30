#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  exec python /app/runner.py "$@"
fi

if [[ -n "${INPUT_ARTIFACTS_PATH:-}" && -n "${RESULT_OUTPUT_PATH:-}" && -n "${STAGE_MANIFEST_PATH:-}" && -n "${OUTPUT_DIR:-}" ]]; then
  exec python /app/platform_runner.py
fi

MODE="${MODE:-video}"
METHOD="${METHOD:-task}"
PREVIEW="${PREVIEW:-false}"
SEGMENT_CUT="${SEGMENT_CUT:-false}"
SEGMENT_GRANULARITY="${SEGMENT_GRANULARITY:-task}"
NO_BATCH="${NO_BATCH:-false}"
MAX_WORKERS="${MAX_WORKERS:-8}"

WINDOW_SEC="${WINDOW_SEC:-10.0}"
STEP_SEC="${STEP_SEC:-5.0}"
FRAMES_PER_WINDOW="${FRAMES_PER_WINDOW:-12}"
TASK_NAME="${TASK_NAME:-}"

TASK_WINDOW_SEC="${TASK_WINDOW_SEC:-12.0}"
TASK_STEP_SEC="${TASK_STEP_SEC:-6.0}"
TASK_FRAMES_PER_WINDOW="${TASK_FRAMES_PER_WINDOW:-12}"
ACTION_WINDOW_SEC="${ACTION_WINDOW_SEC:-6.0}"
ACTION_STEP_SEC="${ACTION_STEP_SEC:-3.0}"
ACTION_FRAMES_PER_WINDOW="${ACTION_FRAMES_PER_WINDOW:-8}"

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

set -- "$@" --method "$METHOD" --max-workers "$MAX_WORKERS"

if normalize_bool "$PREVIEW"; then
  set -- "$@" --preview
fi

if normalize_bool "$SEGMENT_CUT"; then
  set -- "$@" \
    --segment-cut \
    --segment-granularity "$SEGMENT_GRANULARITY"
  if [[ -n "${SEGMENT_OUTPUT_DIR:-}" ]]; then
    set -- "$@" --segment-output-dir "$SEGMENT_OUTPUT_DIR"
  fi
fi

if normalize_bool "$NO_BATCH"; then
  set -- "$@" --no-batch
fi

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
fi

case "$METHOD" in
  task|segment_v2t)
    set -- "$@" \
      --window-sec "$WINDOW_SEC" \
      --step-sec "$STEP_SEC" \
      --frames-per-window "$FRAMES_PER_WINDOW"
    if [[ -n "$TASK_NAME" ]]; then
      set -- "$@" --task-name "$TASK_NAME"
    fi
    ;;
  atomic_action|task_action_v2t)
    set -- "$@" \
      --task-window-sec "$TASK_WINDOW_SEC" \
      --task-step-sec "$TASK_STEP_SEC" \
      --task-frames-per-window "$TASK_FRAMES_PER_WINDOW" \
      --action-window-sec "$ACTION_WINDOW_SEC" \
      --action-step-sec "$ACTION_STEP_SEC" \
      --action-frames-per-window "$ACTION_FRAMES_PER_WINDOW"
    ;;
  *)
    echo "Unsupported METHOD: $METHOD (expected: task or atomic_action)" >&2
    exit 2
    ;;
esac

exec python /app/runner.py "$@"
