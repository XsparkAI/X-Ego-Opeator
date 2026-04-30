#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  exec python /app/runner.py "$@"
fi

if [[ -n "${INPUT_ARTIFACTS_PATH:-}" && -n "${RESULT_OUTPUT_PATH:-}" && -n "${STAGE_MANIFEST_PATH:-}" && -n "${OUTPUT_DIR:-}" ]]; then
  exec python /app/platform_runner.py
fi

MODE="${MODE:-video}"
DETECTION_MODE="${DETECTION_MODE:-sampling_expand}"
FRAME_SAMPLING_STEP="${FRAME_SAMPLING_STEP:-${STEP:-30}}"
SCALE="${SCALE:-1.0}"
BLUR_TARGETS="${BLUR_TARGETS:-face}"
PREVIEW="${PREVIEW:-false}"
USE_FRAME_CACHE="${USE_FRAME_CACHE:-true}"
FRAME_CACHE_NUM_WORKERS="${FRAME_CACHE_NUM_WORKERS:-1}"
YOLO_CONF_THRESH="${YOLO_CONF_THRESH:-0.25}"
YOLO_INPUT_SIZE="${YOLO_INPUT_SIZE:-960}"

DETECTION_MODE="$(printf '%s' "$DETECTION_MODE" | tr '[:upper:]' '[:lower:]')"

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
    if [[ -n "${OUTPUT_PATH:-}" ]]; then
      set -- "$@" --output "$OUTPUT_PATH"
    fi
    ;;
  *)
    echo "Unsupported MODE: $MODE (expected: video or episode)" >&2
    exit 2
    ;;
esac

set -- "$@" \
  --detection-mode "$DETECTION_MODE" \
  --frame-sampling-step "$FRAME_SAMPLING_STEP" \
  --scale "$SCALE" \
  --blur-targets "$BLUR_TARGETS"

if [[ -n "${FACE_THRESH:-}" ]]; then
  set -- "$@" --face-thresh "$FACE_THRESH"
fi
if [[ -n "${LP_THRESH:-}" ]]; then
  set -- "$@" --lp-thresh "$LP_THRESH"
fi

set -- "$@" --yolo-conf-thresh "$YOLO_CONF_THRESH" --yolo-input-size "$YOLO_INPUT_SIZE"
if [[ -n "${YOLO_FACE_MODEL_PATH:-}" ]]; then
  set -- "$@" --yolo-face-model-path "$YOLO_FACE_MODEL_PATH"
fi
if [[ -n "${YOLO_LP_MODEL_PATH:-}" ]]; then
  set -- "$@" --yolo-lp-model-path "$YOLO_LP_MODEL_PATH"
fi

if normalize_bool "$PREVIEW"; then
  set -- "$@" --preview
fi
if normalize_bool "$USE_FRAME_CACHE"; then
  set -- "$@" --use-frame-cache --frame-cache-num-workers "$FRAME_CACHE_NUM_WORKERS"
fi

exec python /app/runner.py "$@"
