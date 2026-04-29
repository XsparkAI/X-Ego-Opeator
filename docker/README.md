# Docker Operators

Each operator should live in its own subdirectory under `docker/`.

Current layout:

- `docker/hand_analysis/`: CPU-only Docker packaging for the YOLO hand detection operator
- `docker/caption/`: CPU-only Docker packaging for the VLM caption / video segmentation operator
- `docker/custom_dataset_operator/`: minimal Drobodata-compatible custom operator image for dataset-style JSON inputs

Build the hand detection image from the repository root:

```bash
docker build -f docker/hand_analysis/Dockerfile -t egox-hand-cpu .
```

Build the caption image from the repository root:

```bash
docker build -f docker/caption/Dockerfile -t egox-caption-cpu .
```
