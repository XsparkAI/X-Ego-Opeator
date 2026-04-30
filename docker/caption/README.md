# Caption Docker

This directory packages the `operators/caption` video segmentation operator as a Docker image.

Supported methods:

- `task`: single-task captioning with sliding-window segmentation
- `atomic_action`: scene -> tasks -> atomic actions

Legacy method values `segment_v2t` and `task_action_v2t` are still accepted as aliases.

The image supports two startup modes:

- local mode: explicit CLI args or env-driven `MODE=video|episode`
- platform mode: automatic Drobodata-compatible execution when `INPUT_ARTIFACTS_PATH`, `OUTPUT_DIR`, `RESULT_OUTPUT_PATH`, and `STAGE_MANIFEST_PATH` are injected

## Build

From the repository root:

```bash
docker build -f docker/caption/Dockerfile -t egox-caption-cpu .
```

## Run With Explicit CLI Args

Single video with `task`:

```bash
docker run --rm \
  -e DASHSCOPE_API_KEY=your_key \
  -v "$(pwd)/test_video:/data" \
  egox-caption-cpu \
  --method task \
  --video /data/test_10s.mp4 \
  --output /data/caption.json
```

Single video with `atomic_action`:

```bash
docker run --rm \
  -e DASHSCOPE_API_KEY=your_key \
  -v "$(pwd)/test_video:/data" \
  egox-caption-cpu \
  --method atomic_action \
  --video /data/test_10s.mp4 \
  --output /data/caption.json \
  --no-batch \
  --max-workers 4
```

Episode directory:

```bash
docker run --rm \
  -e DASHSCOPE_API_KEY=your_key \
  -v "$(pwd)/tmp/test_episode_10s:/data" \
  egox-caption-cpu \
  --method task \
  --episode /data
```

In episode mode, the operator writes `caption.json` into the mounted episode directory unless `--output` is set explicitly.

Optional caption-driven segment cutting:

```bash
docker run --rm \
  -e DASHSCOPE_API_KEY=your_key \
  -v "$(pwd)/tmp/test_episode_10s:/data" \
  egox-caption-cpu \
  --method atomic_action \
  --episode /data \
  --no-batch \
  --segment-cut \
  --segment-granularity atomic_action
```

When enabled, the runner first writes `caption.json`, then cuts clips from that
caption into `segments/seg_*/rgb.mp4` and writes `segments/segments_manifest.json`.
The segment cutter re-encodes at exact caption boundaries while preserving the
source video's codec family, frame rate, pixel format, bitrate when available,
and audio stream settings instead of forcing one fixed ffmpeg profile.

## Caption JSON Schema

`method=task` writes task-level segments without `atomic_actions`:

```json
{
  "instruction": "wipe the table",
  "scene": "kitchen",
  "tasks": [
    {"caption": "pick up the cloth", "frame_interval": [0, 120]},
    {"caption": "wipe the table surface", "frame_interval": [120, 260]}
  ]
}
```

`method=atomic_action` writes task-level captions plus nested atomic actions:

```json
{
  "instruction": "pick up the cloth; wipe the table surface",
  "scene": "kitchen",
  "tasks": [
    {
      "caption": "wipe the table surface",
      "frame_interval": [120, 260],
      "atomic_actions": [
        {"caption": "move the cloth across the table", "frame_interval": [120, 180]}
      ]
    }
  ]
}
```

## Run With Environment Variables

`task` video mode:

```bash
docker run --rm \
  -e MODE=video \
  -e METHOD=task \
  -e VIDEO_PATH=/data/test_10s.mp4 \
  -e OUTPUT_PATH=/data/caption.json \
  -e WINDOW_SEC=10 \
  -e STEP_SEC=5 \
  -e FRAMES_PER_WINDOW=12 \
  -e DASHSCOPE_API_KEY=your_key \
  -v "$(pwd)/test_video:/data" \
  egox-caption-cpu
```

`atomic_action` episode mode:

```bash
docker run --rm \
  -e MODE=episode \
  -e METHOD=atomic_action \
  -e EPISODE_DIR=/data \
  -e EGOX_INPUT_VIDEO_PATH=rgb.mp4 \
  -e NO_BATCH=true \
  -e SEGMENT_CUT=true \
  -e SEGMENT_GRANULARITY=task \
  -e MAX_WORKERS=4 \
  -e TASK_WINDOW_SEC=12 \
  -e TASK_STEP_SEC=6 \
  -e TASK_FRAMES_PER_WINDOW=12 \
  -e ACTION_WINDOW_SEC=6 \
  -e ACTION_STEP_SEC=3 \
  -e ACTION_FRAMES_PER_WINDOW=8 \
  -e DASHSCOPE_API_KEY=your_key \
  -v "$(pwd)/tmp/test_episode_10s:/data" \
  egox-caption-cpu
```

## Run In Platform Mode

When the platform injects runtime variables, the same image switches to a compliant custom-operator adapter automatically.

It will:

- read upstream artifacts from `INPUT_ARTIFACTS_PATH`
- read hyperparameters from `NODE_DATA_JSON.hyperparams`
- resolve a local video file or local episode directory from the input artifact
- download BOS video inputs into `EGOX_BOS_CACHE_DIR` using the same shared cache root as `docker/hand_analysis`
- create a caption-specific runtime directory under that cache root so `.frame_cache` stays isolated per downloaded video
- write the business JSON into `OUTPUT_DIR`
- write `RESULT_OUTPUT_PATH`
- write `STAGE_MANIFEST_PATH`
- print `DROBOTICFLOW_RESULT_PATH=...`

The output artifact follows the platform large-payload convention:

- `artifact.path` points to the full caption JSON in `OUTPUT_DIR`
- `artifact.data` contains only a small summary (`scene`, task/action counts, method, and `payloadPath`)
- `artifact.dataRef.payloadPath` also points to the full caption JSON for downstream operators that prefer external payloads
- `artifact.portId` is `out-1` and `artifact.portName` is `result`
- when `segment_cut=true`, the same caption node also emits `out-2` / `segments`
  as a JSON artifact pointing to `segments_manifest.json`; each manifest entry
  points to the actual cut video and `segment_info.json`
- each successfully cut clip is also emitted directly as a `video` artifact on
  `out-3` / `segment_video`, so platform consumers do not need to parse the
  manifest before seeing the clipped videos

Recommended hyperparams:

Platform hyperparams use one canonical snake_case name per behavior. The
matching uppercase environment variable is only for Docker local/env mode.

- `method`: `task` or `atomic_action`
- `no_batch`; Docker platform mode defaults to direct requests (`no_batch=true`) for every provider
- set `no_batch=false` explicitly only when the selected provider supports async batch submission and the workflow is intended to wait for batch completion
- `volcengine_ark` does not support async batch submission in this project, so platform mode forces direct requests (`no_batch=true`) for that provider
- `max_workers`
- `window_sec`, `step_sec`, `frames_per_window`
- `task_window_sec`, `task_step_sec`, `task_frames_per_window`
- `action_window_sec`, `action_step_sec`, `action_frames_per_window`
- `preview`
- `segment_cut`: optional, defaults to `false`
- `segment_granularity`: `task` or `atomic_action`, defaults to `task`; `atomic_action` requires caption JSON with nested `atomic_actions`
- `vlm_api_provider`, `vlm_api_key`, `vlm_model`

## Cache And Path Conventions

- Local episode inputs continue to use the existing `.frame_cache` directory next to the source video, so repeated caption runs reuse extracted frames.
- BOS downloads continue to use `EGOX_BOS_CACHE_DIR` or `/tmp/egox-bos-cache` by default, matching the existing hand-analysis docker convention.
- Caption then builds a per-video runtime directory under that same cache root, which lets frame-cache files stay deterministic without re-downloading the source video.
