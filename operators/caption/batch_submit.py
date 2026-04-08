#!/usr/bin/env python3
"""
Batch submit: prepare VLM window-analysis requests as JSONL -> submit to DashScope Batch API.

Replaces the real-time VLM calls in segment_v2t.py / segment_v2t_desc_only.py with a
50%-cheaper asynchronous batch job.

Usage:
  python batch_submit.py                                # all episodes, v2t prompt
  python batch_submit.py --variant v2t_desc             # with SOP descriptions
  python batch_submit.py --task waterpour               # filter by task
  python batch_submit.py --episode waterpour1           # single episode
  python batch_submit.py --dry-run                      # write JSONL only, don't submit
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    from ..video_path import resolve_episode_video_path
except ImportError:
    from video_path import resolve_episode_video_path

import segment_v2t as _sv
from segment_v2t import (
    API_KEY, MODEL, WINDOW_SEC, STEP_SEC, FRAMES_PER_WINDOW,
    SCRIPT_DIR, SOP_DIR, TASK_SOP_MAP,
    build_windows, extract_frames_b64, find_imu_valleys,
    discover_episodes, get_task_type,
)

PROJECT_ROOT = SCRIPT_DIR.parent.parent
BATCH_TMP_DIR = PROJECT_ROOT / "tmp" / "batch"
BATCH_STATE_FILE = BATCH_TMP_DIR / "batch_state.json"
# Default data root — overridable via --data-root
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "test_video" / "FusionX-Multimodal-Sample-Data-V2"


# ── Prompt builders (both variants) ─────────────────────────────────────────

def _build_prompt_v2t(sop, n_images, window, fps):
    """Pure visual analysis prompt (no SOP info)."""
    t_start = window.start_frame / fps
    t_end = window.end_frame / fps
    return f"""\
You are analyzing a segment of an egocentric video of a manual task.
This segment covers time {t_start:.1f}s to {t_end:.1f}s of the full video.
You are given {n_images} frames sampled evenly from this segment.

Analyze the {n_images} frames carefully. Identify any task step transitions that occur within this segment.
A transition means the person has COMPLETED one action/step and STARTED a different action/step.
For example: picking up a tool -> using the tool, or pouring water -> putting down the bottle.

IMPORTANT:
- Frame indices are 0 to {n_images - 1} (referring to the {n_images} images you see).
- Only report transitions you are confident about based on visual evidence.
- If no transition occurs in this segment, return an empty transitions array.
- Each instruction should be a brief description of what happens in that segment.

Output ONLY valid JSON in this format:
{{
  "thought": "Brief frame-by-frame analysis...",
  "transitions": [<frame_indices_where_step_changes>],
  "instructions": ["description of segment before first transition", "description after first transition", ...]
}}

Note: len(instructions) should be len(transitions) + 1.
If transitions is empty, instructions should have exactly 1 element describing the whole segment."""


def _build_prompt_v2t_desc(sop, n_images, window, fps):
    """Prompt with SOP step descriptions."""
    steps_text = "\n".join(
        f"  Step {s['step']}: {s['description']}"
        for s in sop["steps"]
    )
    t_start = window.start_frame / fps
    t_end = window.end_frame / fps
    return f"""\
You are analyzing a segment of an egocentric video of a manual task.
This segment covers time {t_start:.1f}s to {t_end:.1f}s of the full video.
You are given {n_images} frames sampled evenly from this segment.

The task follows this SOP (Standard Operating Procedure):
Task: {sop['task_name']}
Steps:
{steps_text}

Analyze the {n_images} frames carefully. Identify any SOP step transitions that occur within this segment.
A transition means the person has COMPLETED one step and STARTED the next step.

IMPORTANT:
- Frame indices are 0 to {n_images - 1} (referring to the {n_images} images you see).
- Only report transitions you are confident about based on visual evidence.
- If no transition occurs in this segment, return an empty transitions array.
- Each instruction should be a brief description of what happens in that segment.

Output ONLY valid JSON in this format:
{{
  "thought": "Brief frame-by-frame analysis...",
  "transitions": [<frame_indices_where_step_changes>],
  "instructions": ["description of segment before first transition", "description after first transition", ...]
}}

Note: len(instructions) should be len(transitions) + 1.
If transitions is empty, instructions should have exactly 1 element describing the whole segment."""


def main():
    p = argparse.ArgumentParser(description="Submit VLM batch job to DashScope")
    p.add_argument("--variant", choices=["v2t", "v2t_desc"], default="v2t",
                   help="Prompt variant: v2t (no SOP) or v2t_desc (with SOP descriptions)")
    p.add_argument("--task", type=str, default=None,
                   help="Filter by task type")
    p.add_argument("--episode", type=str, default=None,
                   help="Process single episode by name")
    p.add_argument("--model", type=str, default="qwen3.5-plus",
                   help="Model name for batch (default: qwen3.5-plus)")
    p.add_argument("--data-root", type=str, default=None,
                   help=f"Data directory (default: {DEFAULT_DATA_ROOT})")
    p.add_argument("--dry-run", action="store_true",
                   help="Write JSONL but don't upload/submit")
    p.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    p.add_argument("--step-sec", type=float, default=STEP_SEC)
    p.add_argument("--frames-per-window", type=int, default=FRAMES_PER_WINDOW)
    args = p.parse_args()

    # Override globals
    _sv.WINDOW_SEC = args.window_sec
    _sv.STEP_SEC = args.step_sec
    _sv.FRAMES_PER_WINDOW = args.frames_per_window
    _sv.DATA_ROOT = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT

    prompt_fn = _build_prompt_v2t if args.variant == "v2t" else _build_prompt_v2t_desc

    # Discover episodes
    episodes = discover_episodes(task_filter=args.task, episode_filter=args.episode)
    if not episodes:
        log.error("No episodes found. Check --task / --episode filter.")
        return

    log.info(f"Found {len(episodes)} episode(s)")

    # ── Build JSONL and state ────────────────────────────────────────────────
    BATCH_TMP_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = BATCH_TMP_DIR / "batch_requests.jsonl"
    state = {
        "variant": args.variant,
        "model": args.model,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": {},
    }

    total_requests = 0

    with open(jsonl_path, "w") as f:
        for ep_dir, sop in episodes:
            name = ep_dir.name
            rgb_path = str(resolve_episode_video_path(ep_dir))

            cap = cv2.VideoCapture(rgb_path)
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

            windows = build_windows(fps, nframes)
            valleys = find_imu_valleys(ep_dir)

            state["episodes"][name] = {
                "fps": fps,
                "nframes": nframes,
                "task_type": get_task_type(name),
                "windows": [
                    {"window_id": w.window_id, "start_frame": w.start_frame,
                     "end_frame": w.end_frame, "frame_ids": w.frame_ids}
                    for w in windows
                ],
                "imu_valleys": valleys.tolist() if valleys is not None else None,
            }

            log.info(f"[{name}] {nframes} frames, {fps:.0f} fps, {len(windows)} windows — extracting frames...")

            for w in windows:
                frames_b64 = extract_frames_b64(rgb_path, w.frame_ids)
                prompt = prompt_fn(sop, len(frames_b64), w, fps)

                # OpenAI-compatible multimodal message
                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    for b64 in frames_b64
                ]
                content.append({"type": "text", "text": prompt})

                request = {
                    "custom_id": f"{name}__win{w.window_id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": [{"role": "user", "content": content}],
                        
                    },
                }
                f.write(json.dumps(request, ensure_ascii=False) + "\n")
                total_requests += 1

            log.info(f"[{name}] {len(windows)} requests written")

    jsonl_mb = jsonl_path.stat().st_size / (1024 * 1024)
    log.info(f"JSONL ready: {jsonl_path} ({jsonl_mb:.1f} MB, {total_requests} requests)")

    if args.dry_run:
        log.info("Dry-run mode — skipping upload")
        state["batch_id"] = None
        state["input_file_id"] = None
        state["status"] = "dry_run"
        BATCH_STATE_FILE.write_text(json.dumps(state, indent=2))
        log.info(f"State: {BATCH_STATE_FILE}")
        return

    # ── Upload & submit ──────────────────────────────────────────────────────
    from openai import OpenAI

    client = OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    log.info("Uploading JSONL file...")
    file_obj = client.files.create(file=jsonl_path, purpose="batch")
    log.info(f"File uploaded: {file_obj.id}")

    log.info("Creating batch job...")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    log.info(f"Batch created: {batch.id} (status: {batch.status})")

    state["batch_id"] = batch.id
    state["input_file_id"] = file_obj.id
    state["status"] = batch.status
    BATCH_STATE_FILE.write_text(json.dumps(state, indent=2))

    print(f"\n  Batch job submitted: {batch.id}")
    print(f"  Requests: {total_requests}")
    print(f"  JSONL size: {jsonl_mb:.1f} MB")
    print(f"  Run `python batch_collect.py` to poll and collect results.\n")


if __name__ == "__main__":
    main()
