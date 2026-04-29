# Privacy Blur Docker

This directory packages the privacy blur operator as a CPU-only Docker image.

Supported backend:

- `yolo`: dedicated YOLO face and license-plate models using `/app/weights/yolo26s_face.pt` and `/app/weights/yolo_lp.pt`

The default target is face-only. Set `blur_targets=lp` or `blur_targets=both` to enable license-plate detection.

The image supports:

- local mode: explicit CLI args or env-driven `MODE=video|episode`
- platform mode: automatic Drobodata-compatible execution when `INPUT_ARTIFACTS_PATH`, `OUTPUT_DIR`, `RESULT_OUTPUT_PATH`, and `STAGE_MANIFEST_PATH` are injected

## Build

From the repository root:

```bash
docker build -f docker/privacy_blur/Dockerfile -t egox-privacy-blur-cpu .
```

The image embeds the current face / license-plate detection code plus `weights/yolo26s_face.pt` and `weights/yolo_lp.pt`.

## Run With Explicit CLI Args

Single video:

```bash
docker run --rm \
  -v "$(pwd)/test_video:/data" \
  egox-privacy-blur-cpu \
  --video /data/test_10s.mp4 \
  --output /data/test_10s_blurred.mp4 \
  --frame-sampling-step 30
```

Episode mode:

```bash
docker run --rm \
  -v "$(pwd)/tmp/test_episode_10s:/data" \
  egox-privacy-blur-cpu \
  --episode /data \
  --frame-sampling-step 30
```

With explicit YOLO options:

```bash
docker run --rm \
  -v "$(pwd)/test_video:/data" \
  egox-privacy-blur-cpu \
  --video /data/test_10s.mp4 \
  --output /data/test_10s_blurred.mp4 \
  --yolo-face-model-path /app/weights/yolo26s_face.pt \
  --yolo-lp-model-path /app/weights/yolo_lp.pt \
  --blur-targets both \
  --yolo-conf-thresh 0.25 \
  --yolo-input-size 960
```

## Run With Environment Variables

```bash
docker run --rm \
  -e MODE=video \
  -e VIDEO_PATH=/data/test_10s.mp4 \
  -e OUTPUT_PATH=/data/test_10s_blurred.mp4 \
  -e FRAME_SAMPLING_STEP=30 \
  -v "$(pwd)/test_video:/data" \
  egox-privacy-blur-cpu
```

For episode mode, mount an episode directory and set `MODE=episode`, `EPISODE_DIR=/data`.

## Platform Mode

When the platform injects runtime variables, the image writes:

- blurred video artifact on `out-1`
- summary JSON artifact on `out-2`
- optional preview video on `out-3` when `preview=true`
- `RESULT_OUTPUT_PATH`
- `STAGE_MANIFEST_PATH`
- stdout line `DROBOTICFLOW_RESULT_PATH=...`

For `x-platform-clips` inputs, the adapter reads local video paths from `artifact.data`,
`dataRef.payloadPath`, `dataRef.manifestPath`, and `dataRef.partitionDir`. Each input clip
is processed independently and produces a matching `<clip-stem>_blurred.mp4` artifact.

Example local simulation:

```bash
mkdir -p /tmp/egox-privacy-platform/outputs
```

Create `/tmp/egox-privacy-platform/input-artifacts.json`:

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
  -v /tmp/egox-privacy-platform:/workspace/out \
  -e INPUT_ARTIFACTS_PATH=/workspace/out/input-artifacts.json \
  -e OUTPUT_DIR=/workspace/out/outputs \
  -e RESULT_OUTPUT_PATH=/workspace/out/result.json \
  -e STAGE_MANIFEST_PATH=/workspace/out/stage-manifest.json \
  -e NODE_DATA_JSON='{"hyperparams":{"frame_sampling_step":30}}' \
  egox-privacy-blur-cpu
```

Platform hyperparameters include:

- `detection_mode`: `sampling_expand` or `legacy_per_frame`
- `frame_sampling_step`: default `30`
- `scale`
- `blur_targets`: `face` (default), `lp`, or `both`
- `face_thresh`, `lp_thresh`
- `use_frame_cache`, `frame_cache_num_workers`
- `yolo_face_model_path`, `yolo_lp_model_path`, `yolo_conf_thresh`, `yolo_input_size`
- `output_name`

## Push

After logging in to CCR:

```bash
docker/privacy_blur/upload.sh
```
