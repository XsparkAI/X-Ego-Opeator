# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ego-X_Operator is a video segmentation pipeline for egocentric manual task videos (FusionX dataset). It uses a VLM (Qwen via DashScope) with a sliding-window approach to detect task step transitions, then refines cut points using IMU accelerometer data.

## Running

```bash
# Process all episodes
python caption/segment_v2t.py

# Filter by task or episode
python caption/segment_v2t.py --task waterpour
python caption/segment_v2t.py --episode waterpour1

# Preview and dry-run
python caption/segment_v2t.py --preview
python caption/segment_v2t.py --dry-run
```

The `segment_v2t_desc_only.py` variant includes SOP step descriptions in the VLM prompt and adds per-phase benchmark metrics.

## Dependencies

Python with: `cv2` (OpenCV), `numpy`, `pandas`, `scipy`, `dashscope` (Alibaba Cloud VLM API), optional `Pillow` for better subtitle rendering. Requires `ffprobe` on PATH for video rotation detection.

## Architecture

### Core Pipeline (5 stages)

1. **Windowing** — Split video into overlapping windows (default 10s window, 5s step), sample 12 frames per window
2. **VLM Analysis** — Send frames to Qwen VLM to detect step transitions per window (parallel via ThreadPoolExecutor)
3. **Cut Clustering** — Merge nearby transitions across windows using Hanning-weighted voting (cluster radius = 2s)
4. **IMU Snap** — Snap cut points to nearest IMU acceleration valley (from `frames.parquet`, radius = 45 frames)
5. **Output** — Write `caption_v2t.json` per episode with `atomic_action` segments

### Data Layout

- `caption/FusionX-Multimodal-Sample-Data-V2/<episode>/` — episode directories containing `rgb.mp4` and `frames.parquet`
- `caption/sop/` — SOP JSON files per task type (box_cut, drill, screwunscrew, syringe, waterpour)
- Output: `caption_v2t.json` (or `caption_v2t_desc.json`) written into each episode directory

### VLM Configuration

Uses DashScope API with model `qwen3.5-plus`. API key sourced from `DASHSCOPE_API_KEY` env var.

### Key Tuning Parameters

All configurable via CLI flags or module-level constants:
- `WINDOW_SEC` / `STEP_SEC` — window duration and overlap
- `FRAMES_PER_WINDOW` — frames sampled per window (affects VLM cost)
- `SNAP_RADIUS_FRAMES` — how far a cut can snap to an IMU valley
- `MIN_SEGMENT_SEC` / `CLUSTER_SEC` — minimum segment length and merge radius
