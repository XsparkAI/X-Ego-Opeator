# Hand Analysis Docker

This directory packages the hand analysis operator as a CPU-only Docker image.

Supported backends:

- `yolo`: local hand detector using `weights/detector.pt`
- `vlm`: remote VLM API inference via DashScope or Volcengine Ark

The image now supports two startup modes:

- local mode: explicit CLI args or env-driven `MODE=video|episode`
- platform mode: automatic Drobodata-compatible execution when `INPUT_ARTIFACTS_PATH`, `OUTPUT_DIR`, `RESULT_OUTPUT_PATH`, and `STAGE_MANIFEST_PATH` are injected

## Build

From the repository root:

```bash
docker build -f docker/hand_analysis/Dockerfile -t egox-hand-cpu .
```

The image embeds the current hand detector weights from `weights/detector.pt`.

Important: the container can only read files that are inside mounted volumes. A host path like `./tmp/test_episode_10s/rgb.mp4` must be mounted into the container first, then referenced by the in-container path.

## Run With Explicit CLI Args

YOLO mode on a single video:

```bash
docker run --rm \
  -v "$(pwd)/test_video:/data" \
  egox-hand-cpu \
  --backend yolo \
  --video /data/test_10s.mp4 \
  --output /data/hand_analysis.json \
  --conf 0.3 \
  --frame-step 1 \
  --resize 720
```

Equivalent command for `tmp/test_episode_10s/rgb.mp4` from the repository root:

```bash
docker run --rm \
  -v "$(pwd)/tmp:/data" \
  egox-hand-cpu \
  --backend yolo \
  --video /data/test_episode_10s/rgb.mp4 \
  --output /data/test_episode_10s/hand_analysis.json
```

YOLO mode with preview:

```bash
docker run --rm \
  -v "$(pwd)/test_video:/data" \
  egox-hand-cpu \
  --backend yolo \
  --video /data/test_10s.mp4 \
  --output /data/hand_analysis.json \
  --preview
```

VLM mode on a single video:

```bash
docker run --rm \
  -e VLM_API_PROVIDER=dashscope \
  -e DASHSCOPE_API_KEY=your_key \
  -e VLM_MODEL=qwen3.5-flash \
  -v "$(pwd)/test_video:/data" \
  egox-hand-cpu \
  --backend vlm \
  --video /data/test_10s.mp4 \
  --output /data/hand_analysis.json \
  --frame-step 30
```

VLM mode with direct requests instead of batch:

```bash
docker run --rm \
  -e VLM_API_PROVIDER=volcengine_ark \
  -e ARK_API_KEY=your_key \
  -e VLM_MODEL=doubao-seed-2-0-lite-260215 \
  -v "$(pwd)/test_video:/data" \
  egox-hand-cpu \
  --backend vlm \
  --video /data/test_10s.mp4 \
  --output /data/hand_analysis.json \
  --frame-step 30 \
  --no-batch \
  --max-workers 4
```

## Run With Environment Variables

The entrypoint also supports a simpler env-driven mode.

YOLO video mode:

```bash
docker run --rm \
  -e MODE=video \
  -e BACKEND=yolo \
  -e VIDEO_PATH=/data/test_10s.mp4 \
  -e OUTPUT_PATH=/data/hand_analysis.json \
  -e CONF=0.3 \
  -e STEP=1 \
  -e RESIZE=720 \
  -v "$(pwd)/test_video:/data" \
  egox-hand-cpu
```

YOLO episode mode:

```bash
docker run --rm \
  -e MODE=episode \
  -e BACKEND=yolo \
  -e EPISODE_DIR=/data \
  -e EGOX_INPUT_VIDEO_PATH=rgb.mp4 \
  -e CONF=0.3 \
  -e STEP=1 \
  -e RESIZE=720 \
  -v "$(pwd)/tmp/test_episode_10s:/data" \
  egox-hand-cpu
```

VLM video mode:

```bash
docker run --rm \
  -e MODE=video \
  -e BACKEND=vlm \
  -e VIDEO_PATH=/data/test_10s.mp4 \
  -e OUTPUT_PATH=/data/hand_analysis.json \
  -e STEP=30 \
  -e VLM_API_PROVIDER=dashscope \
  -e DASHSCOPE_API_KEY=your_key \
  -e VLM_MODEL=qwen3.5-flash \
  -v "$(pwd)/test_video:/data" \
  egox-hand-cpu
```

VLM episode mode:

```bash
docker run --rm \
  -e MODE=episode \
  -e BACKEND=vlm \
  -e EPISODE_DIR=/data \
  -e EGOX_INPUT_VIDEO_PATH=rgb.mp4 \
  -e STEP=30 \
  -e VLM_API_PROVIDER=volcengine_ark \
  -e ARK_API_KEY=your_key \
  -e VLM_MODEL=doubao-seed-2-0-lite-260215 \
  -e NO_BATCH=true \
  -e MAX_WORKERS=4 \
  -v "$(pwd)/tmp/test_episode_10s:/data" \
  egox-hand-cpu
```

In episode mode, the operator writes `hand_analysis.json` into the mounted episode directory.

Both backends now write the same top-level JSON shape:

- `backend`
- `summary`
- `frameResults`
- `rawSummary`

And `summary` uses one shared camelCase schema across YOLO and VLM:

- `totalFrames`
- `sampledFrames`
- `validFrames`
- `failedFrames`
- `avgEgoHandCount`
- `framesWithAtLeastOneHand`
- `atLeastOneHandRatio`
- `framesWithBothHands`
- `bothHandsRatio`
- `framesWithNoHands`
- `noHandsRatio`
- `activeManipulationRatio`
- `singlePersonOperationRatio`

## Run In Platform Mode

When the platform injects runtime variables, the same image switches to a compliant custom-operator adapter automatically.

It will:

- read upstream artifacts from `INPUT_ARTIFACTS_PATH`
- read hyperparameters from `NODE_DATA_JSON.hyperparams`
- resolve a local video file or local episode directory from the input artifact
- write the business JSON into `OUTPUT_DIR`
- write `RESULT_OUTPUT_PATH`
- write `STAGE_MANIFEST_PATH`
- print `DROBOTICFLOW_RESULT_PATH=...`

Current input resolution priority:

1. local `artifact.path`
2. local path fields inside `artifact.data`
3. local paths found via `dataRef.payloadPath`, `manifestPath`, or `partitionDir`

Current limitation:

- if the input only exposes remote `bos://...` references, the container now tries to download the video locally first
- this requires valid BOS credentials via `Endpoint` / `AccessKey` / `SecretKey`
- platform-style aliases `BOS_ENDPOINT` / `BOS_ACCESS_KEY_ID` / `BOS_SECRET_ACCESS_KEY` and `accessKeyId` / `secretAccessKey` are also supported
- the adapter writes both `credentials` and `config` for `bcecmd`, so non-default BOS domains can be resolved correctly
- when only a BOS prefix is available, the adapter lists objects under that prefix and prefers `rgb.mp4`, then `cam_head.mp4`, then other video files
- VLM model env/config now uses `VLM_MODEL`; old `VLM_HAND_MODEL` is still accepted as a compatibility alias
- if no VLM model is specified, the provider-specific fallback is `qwen3.5-flash` for `dashscope` and `doubao-seed-2-0-lite-260215` for `volcengine_ark`

Example local simulation:

```bash
mkdir -p /tmp/egox-hand-platform/outputs
```

Create `/tmp/egox-hand-platform/input-artifacts.json`:

```json
[
  {
    "name": "input-video",
    "type": "video",
    "portId": "in-1",
    "portName": "input",
    "path": "/workspace/test_10s.mp4"
  }
]
```

Run:

```bash
docker run --rm \
  -v "$(pwd)/test_video/test_10s.mp4:/workspace/test_10s.mp4:ro" \
  -v /tmp/egox-hand-platform:/workspace/out \
  -e INPUT_ARTIFACTS_PATH=/workspace/out/input-artifacts.json \
  -e OUTPUT_DIR=/workspace/out/outputs \
  -e RESULT_OUTPUT_PATH=/workspace/out/result.json \
  -e STAGE_MANIFEST_PATH=/workspace/out/stage-manifest.json \
  -e NODE_DATA_JSON='{"hyperparams":{"method":"yolo","yolo_frame_step":1,"conf_thresh":0.3,"input_height":720}}' \
  egox-hand-cpu
```

In that example, put `input-artifacts.json` under `/tmp/egox-hand-platform/` before running.

## Notes

- This image is CPU-only.
- `vlm` mode requires outbound network access and a valid API key.
- DashScope supports batch mode; Ark should usually be run with `NO_BATCH=true`.
- The build context is reduced by the repository-root `.dockerignore`.
