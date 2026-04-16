#!/usr/bin/env python3
"""
Single-task caption generation via VLM sliding-window analysis.

Core algorithm:
  1. Split video into overlapping windows, sample N frames per window
  2. Send frames to VLM → detect transitions per window
  3. Cluster transitions across windows using Hanning-weighted voting
  4. Assemble a unified caption with one task and multiple atomic actions

Public API:
  from operators.caption.segment_v2t import segment
  result = segment("video.mp4")

CLI usage:
  python segment_v2t.py video.mp4                  # Process a single video
  python segment_v2t.py a.mp4 b.mp4                # Process multiple videos
  python segment_v2t.py video.mp4 --preview        # Generate preview video
  python segment_v2t.py video.mp4 --dry-run        # Show plan without VLM calls
"""

import base64
import json
import logging
import os
import re
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ..vlm_limit import vlm_api_slot
try:
    from .vlm_api import (
        build_multimodal_message,
        collect_batch_chat_requests,
        submit_batch_chat_requests,
        submit_batch_chat_requests_async,
        submit_direct_chat_requests,
    )
except ImportError:
    from vlm_api import (
        build_multimodal_message,
        collect_batch_chat_requests,
        submit_batch_chat_requests,
        submit_batch_chat_requests_async,
        submit_direct_chat_requests,
    )
try:
    from .scene_classifier import (
        classify_video_scene,
        classify_video_scene_direct,
        collect_scene_classification,
        submit_scene_classification,
    )
except ImportError:
    from scene_classifier import (
        classify_video_scene,
        classify_video_scene_direct,
        collect_scene_classification,
        submit_scene_classification,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("dashscope").setLevel(logging.CRITICAL)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent

# ── VLM Config ───────────────────────────────────────────────────────────────
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MODEL = "qwen3.5-plus"
THINKING_BUDGET = 1024
EXTRA_BODY = {"enable_thinking": True, "thinking_budget": THINKING_BUDGET}

# ── Windowing Config ─────────────────────────────────────────────────────────
WINDOW_SEC = 10.0          # Window duration in seconds
STEP_SEC = 5.0             # Step between windows (overlap = window - step)
FRAMES_PER_WINDOW = 12     # Number of frames sampled per window
TARGET_W = 640             # Resize width for VLM
TARGET_H = 480             # Resize height for VLM

# ── Segmentation Config ─────────────────────────────────────────────────────
MIN_SEGMENT_SEC = 0.8      # Minimum segment duration
CLUSTER_SEC = 2.0          # Cluster radius for merging nearby cuts

# ── Token tracking ───────────────────────────────────────────────────────────
_token_stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

OUTPUT_SUFFIX = "v2t"


# ═══════════════════════════════════════════════════════════════════════════════
# Windowing
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Window:
    window_id: int
    start_frame: int
    end_frame: int
    frame_ids: list  # Sampled frame indices within this window


def build_windows(fps: float, nframes: int) -> list[Window]:
    """Build overlapping windows across the video."""
    window_frames = int(WINDOW_SEC * fps)
    step_frames = int(STEP_SEC * fps)

    if nframes <= window_frames:
        frame_ids = np.linspace(0, nframes - 1, FRAMES_PER_WINDOW, dtype=int).tolist()
        return [Window(0, 0, nframes - 1, frame_ids)]

    windows = []
    wid = 0
    start = 0
    while start < nframes:
        end = min(start + window_frames - 1, nframes - 1)
        actual_len = end - start + 1
        n_samples = min(FRAMES_PER_WINDOW, actual_len)
        frame_ids = np.linspace(start, end, n_samples, dtype=int).tolist()
        windows.append(Window(wid, start, end, frame_ids))
        wid += 1
        start += step_frames
        if end == nframes - 1:
            break

    return windows


def extract_frames_b64(video_path: str, frame_ids: list[int]) -> list[str]:
    """Extract specific frames from video as base64-encoded JPG strings."""
    from ..frame_cache.cache_utils import ensure_cached_frame_b64
    from ..video_utils import get_manual_rotation, apply_rotation

    cached = ensure_cached_frame_b64(Path(video_path).parent, frame_ids)
    if cached:
        return cached

    rotation = get_manual_rotation(video_path)
    cap = cv2.VideoCapture(video_path)
    results = []
    for fid in sorted(set(frame_ids)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = apply_rotation(frame, rotation)
        frame = cv2.resize(frame, (TARGET_W, TARGET_H))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        results.append(base64.b64encode(buf).decode("ascii"))
    cap.release()
    return results


def save_frames_as_tmp_jpg(video_path: str, frame_ids: list[int], tmp_dir: str) -> list[str]:
    """Extract frames and save as temporary JPG files. Returns file paths."""
    from ..frame_cache.cache_utils import ensure_cached_frame_paths
    from ..video_utils import get_manual_rotation, apply_rotation

    cached = ensure_cached_frame_paths(Path(video_path).parent, frame_ids)
    if cached:
        return cached

    rotation = get_manual_rotation(video_path)
    cap = cv2.VideoCapture(video_path)
    paths = []
    for i, fid in enumerate(sorted(set(frame_ids))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = apply_rotation(frame, rotation)
        frame = cv2.resize(frame, (TARGET_W, TARGET_H))
        path = os.path.join(tmp_dir, f"frame_{i:03d}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(path)
    cap.release()
    return paths



# ═══════════════════════════════════════════════════════════════════════════════
# VLM Prompt & Inference
# ═══════════════════════════════════════════════════════════════════════════════

def build_window_prompt(n_images: int, window: Window, fps: float, task_name: str | None = None) -> str:
    """Build prompt for a single window."""
    t_start = window.start_frame / fps
    t_end = window.end_frame / fps

    task_desc = f" of a manual task: {task_name}" if task_name else " of a manual task"

    return f"""\
You are analyzing a segment of an egocentric video{task_desc}.
This segment covers time {t_start:.1f}s to {t_end:.1f}s of the full video.
You are given {n_images} frames sampled evenly from this segment.

Analyze the {n_images} frames carefully. Identify any task step transitions that occur within this segment.
A transition means the person has COMPLETED one action/step and STARTED a different action/step.
For example: picking up a tool → using the tool, or pouring water → putting down the bottle.

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


def _parse_vlm_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        return None
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return None


def vlm_analyze_windows_batch(video_path: str, windows: list[Window], fps: float, task_name: str | None = None) -> list[dict | None]:
    requests = build_window_requests(video_path, windows, fps, task_name)
    if not requests:
        return [None] * len(windows)

    with vlm_api_slot():
        responses = submit_batch_chat_requests(requests, model=MODEL, extra_body=EXTRA_BODY)

    return parse_window_batch_responses(windows, requests, responses)


def vlm_analyze_windows_direct(
    video_path: str,
    windows: list[Window],
    fps: float,
    task_name: str | None = None,
    max_workers: int = 8,
) -> list[dict | None]:
    requests = build_window_requests(video_path, windows, fps, task_name)
    if not requests:
        return [None] * len(windows)

    with vlm_api_slot():
        responses = submit_direct_chat_requests(
            requests,
            model=MODEL,
            extra_body=EXTRA_BODY,
            max_workers=max_workers,
        )
    return parse_window_batch_responses(windows, requests, responses)


def build_window_requests(
    video_path: str,
    windows: list[Window],
    fps: float,
    task_name: str | None = None,
) -> list[dict]:
    requests: list[dict] = []
    for window in windows:
        frames_b64 = extract_frames_b64(video_path, window.frame_ids)
        if not frames_b64:
            continue
        prompt = build_window_prompt(len(frames_b64), window, fps, task_name)
        requests.append(
            {
                "custom_id": f"win_{window.window_id}",
                "model": MODEL,
                "messages": build_multimodal_message(frames_b64, prompt),
            }
        )
    return requests


def parse_window_batch_responses(
    windows: list[Window],
    requests: list[dict],
    responses: dict[str, dict],
) -> list[dict | None]:
    by_id = {req["custom_id"]: _parse_vlm_json(responses.get(req["custom_id"], {}).get("text")) for req in requests}
    return [by_id.get(f"win_{window.window_id}") for window in windows]


def parse_window_results_map(
    windows: list[Window],
    responses: dict[str, dict],
) -> list[dict | None]:
    return [_parse_vlm_json(responses.get(f"win_{window.window_id}", {}).get("text")) for window in windows]


# ═══════════════════════════════════════════════════════════════════════════════
# Cut Clustering
# ═══════════════════════════════════════════════════════════════════════════════

def build_segments_via_cuts(
    windows: list[Window],
    window_results: list[dict | None],
    fps: float,
    nframes: int,
    task_name: str | None = None,
) -> list[dict]:
    """Cluster per-window transitions into final segments using Hanning weighting."""
    n_windows = len(windows)

    # Step 1: Collect weighted cuts from all windows
    raw_cuts = []  # list of (global_frame, weight)
    window_instructions = []  # list of (global_frame_start, global_frame_end, instruction)

    hanning = np.hanning(n_windows + 2)[1:-1] if n_windows > 1 else np.array([1.0])

    for i, (win, result) in enumerate(zip(windows, window_results)):
        if result is None:
            continue

        transitions = result.get("transitions", [])
        instructions = result.get("instructions", [])
        n_sampled = len(win.frame_ids)

        # Map local frame indices to global frames
        for t_idx in transitions:
            if 0 <= t_idx < n_sampled:
                global_frame = win.frame_ids[min(t_idx, len(win.frame_ids) - 1)]
                raw_cuts.append((global_frame, hanning[i]))

        # Map instructions to global frame ranges
        if instructions:
            # Build segment boundaries within this window
            seg_bounds = [0] + [
                min(t, n_sampled - 1) for t in transitions if 0 <= t < n_sampled
            ] + [n_sampled - 1]

            for j in range(len(seg_bounds) - 1):
                if j < len(instructions):
                    s_local = seg_bounds[j]
                    e_local = seg_bounds[j + 1]
                    s_global = win.frame_ids[min(s_local, len(win.frame_ids) - 1)]
                    e_global = win.frame_ids[min(e_local, len(win.frame_ids) - 1)]
                    window_instructions.append((s_global, e_global, instructions[j]))

    if not raw_cuts:
        # No transitions detected → single segment
        instruction = task_name or "Full video"
        if window_instructions:
            instruction = window_instructions[0][2]
        return [{
            "step": 1,
            "frame_interval": [0, nframes],
            "instruction": instruction,
            "confidence": 1.0,
        }]

    # Step 2: Cluster nearby cuts
    cluster_radius = int(CLUSTER_SEC * fps)
    raw_cuts.sort(key=lambda x: x[0])

    clusters = []
    current_cluster = [raw_cuts[0]]
    for frame, weight in raw_cuts[1:]:
        if frame - current_cluster[0][0] <= cluster_radius:
            current_cluster.append((frame, weight))
        else:
            clusters.append(current_cluster)
            current_cluster = [(frame, weight)]
    clusters.append(current_cluster)

    # Weighted average for each cluster → final cut points
    cut_points = []
    for cluster in clusters:
        total_w = sum(w for _, w in cluster)
        avg_frame = int(sum(f * w for f, w in cluster) / total_w)
        cut_points.append(avg_frame)

    # Deduplicate and sort
    cut_points = sorted(set(cut_points))

    # Step 3: Filter out cuts that create too-short segments
    min_frames = int(MIN_SEGMENT_SEC * fps)
    filtered = []
    prev = 0
    for cp in cut_points:
        if cp - prev >= min_frames:
            filtered.append(cp)
            prev = cp
    if nframes - prev < min_frames and filtered:
        filtered.pop()
    cut_points = filtered

    # Step 4: Build segment list with instructions
    boundaries = [0] + cut_points + [nframes]
    segments = []

    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        seg_mid = (seg_start + seg_end) // 2

        # Find best instruction: most common in center 20% of segment
        center_start = seg_start + int(0.4 * (seg_end - seg_start))
        center_end = seg_start + int(0.6 * (seg_end - seg_start))

        matching_instructions = []
        for s, e, instr in window_instructions:
            if s <= center_end and e >= center_start:
                matching_instructions.append(instr)

        if matching_instructions:
            instruction = Counter(matching_instructions).most_common(1)[0][0]
        else:
            instruction = f"Step {i + 1}"

        segments.append({
            "step": i,
            "frame_interval": [seg_start, seg_end],
            "instruction": instruction,
            "confidence": min(1.0, len([c for c in clusters if i < len(clusters)]) / n_windows) if clusters else 0.5,
        })

    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# Video Processing
# ═══════════════════════════════════════════════════════════════════════════════

def segment(
    video_path: str | Path,
    *,
    task_name: str | None = None,
    window_sec: float = WINDOW_SEC,
    step_sec: float = STEP_SEC,
    frames_per_window: int = FRAMES_PER_WINDOW,
    max_workers: int = 8,
    batch_enabled: bool = True,
) -> dict:
    """Generate a unified caption with one task and multiple atomic actions.

    Args:
        video_path: Path to the mp4 video file.
        task_name: Optional task description to provide context to the VLM
                   (e.g. "water pouring", "screw assembly").
        window_sec: Sliding window duration in seconds.
        step_sec: Step between windows in seconds.
        frames_per_window: Number of frames sampled per window.

    Returns:
        dict with keys:
          - scene: scene label
          - tasks: a single task covering the whole video
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    name = video_path.stem

    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    log.info(f"[{name}] {nframes} frames, {fps:.0f} fps, {nframes / fps:.1f}s")

    # Build windows
    windows = build_windows(fps, nframes)
    log.info(f"[{name}] {len(windows)} windows (window={window_sec}s, step={step_sec}s, {frames_per_window} frames/win)")

    mode_desc = "batch API" if batch_enabled else "direct API"
    log.info(f"[{name}] Submitting scene + {len(windows)} windows via {mode_desc}...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        if batch_enabled:
            future_scene = pool.submit(classify_video_scene, video_path, fps=fps, nframes=nframes)
            future_windows = pool.submit(vlm_analyze_windows_batch, str(video_path), windows, fps, task_name)
        else:
            future_scene = pool.submit(classify_video_scene_direct, video_path, fps=fps, nframes=nframes)
            future_windows = pool.submit(
                vlm_analyze_windows_direct,
                str(video_path),
                windows,
                fps,
                task_name,
                max_workers,
            )
        scene = future_scene.result()
        window_results = future_windows.result()
    log.info(f"[{name}] Scene: {scene}")

    # Build segments via cut clustering
    segments = build_segments_via_cuts(windows, window_results, fps, nframes, task_name)
    task_caption = task_name or (segments[0]["instruction"] if segments else "perform the current task")

    return {
        "scene": scene,
        "tasks": [
            {
                "instruction": task_caption,
                "frame_interval": [0, nframes],
                "atomic_actions": [
                    {
                        "frame_interval": seg["frame_interval"],
                        "caption": seg["instruction"],
                    }
                    for seg in segments
                ],
            }
        ],
    }


def submit_segment_job(
    video_path: str | Path,
    *,
    task_name: str | None = None,
    window_sec: float = WINDOW_SEC,
    step_sec: float = STEP_SEC,
    frames_per_window: int = FRAMES_PER_WINDOW,
) -> dict:
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    windows = build_windows(fps, nframes)
    requests = build_window_requests(str(video_path), windows, fps, task_name)

    log.info(f"[{video_path.stem}] Submitting scene + {len(windows)} windows via batch API...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_scene = pool.submit(submit_scene_classification, video_path, fps=fps, nframes=nframes)
        future_windows = pool.submit(
            lambda: (
                {"batch_id": None, "request_count": 0}
                if not requests
                else submit_batch_chat_requests_async(requests, model=MODEL, extra_body=EXTRA_BODY)
            )
        )
        scene_submission = future_scene.result()
        windows_submission = future_windows.result()

    return {
        "method": "segment_v2t",
        "video_path": str(video_path),
        "task_name": task_name,
        "fps": fps,
        "nframes": nframes,
        "windows": [
            {
                "window_id": w.window_id,
                "start_frame": w.start_frame,
                "end_frame": w.end_frame,
                "frame_ids": w.frame_ids,
            }
            for w in windows
        ],
        "scene_submission": scene_submission,
        "windows_submission": windows_submission,
    }


def collect_segment_job(state: dict, *, poll_interval_sec: int = 20) -> dict:
    video_path = Path(state["video_path"])
    fps = float(state["fps"])
    nframes = int(state["nframes"])
    task_name = state.get("task_name")
    windows = [Window(**w) for w in state.get("windows", [])]

    scene_submission = state.get("scene_submission")
    windows_submission = state.get("windows_submission") or {}

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_scene = pool.submit(
            collect_scene_classification,
            scene_submission,
            poll_interval_sec=poll_interval_sec,
        )
        future_windows = pool.submit(
            lambda: {} if not windows_submission.get("batch_id") else collect_batch_chat_requests(
                windows_submission["batch_id"],
                poll_interval_sec=poll_interval_sec,
                wait=True,
            )
        )
        scene = future_scene.result()
        windows_result = future_windows.result()

    if windows_submission.get("batch_id") and windows_result.get("status") != "completed":
        raise RuntimeError(f"Window batch ended with status: {windows_result.get('status')}")

    window_results = parse_window_results_map(windows, windows_result.get("results", {})) if windows else []

    segments = build_segments_via_cuts(windows, window_results, fps, nframes, task_name)
    task_caption = task_name or (segments[0]["instruction"] if segments else "perform the current task")
    log.info(f"[{video_path.stem}] Scene: {scene}")

    return {
        "scene": scene,
        "tasks": [
            {
                "instruction": task_caption,
                "frame_interval": [0, nframes],
                "atomic_actions": [
                    {
                        "frame_interval": seg["frame_interval"],
                        "caption": seg["instruction"],
                    }
                    for seg in segments
                ],
            }
        ],
    }


def process_video(video_path: Path, preview: bool = False, dry_run: bool = False, task_name: str | None = None) -> dict | None:
    """Process a single video file with file I/O and optional preview. Used by CLI."""
    video_path = Path(video_path).resolve()
    name = video_path.stem

    if dry_run:
        cap = cv2.VideoCapture(str(video_path))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        windows = build_windows(fps, nframes)
        log.info(f"[{name}] {nframes} frames, {fps:.0f} fps, {nframes / fps:.1f}s")
        log.info(f"[{name}] {len(windows)} windows")
        for w in windows:
            log.info(f"  Window {w.window_id}: frames [{w.start_frame}, {w.end_frame}] "
                     f"({(w.end_frame - w.start_frame) / fps:.1f}s), {len(w.frame_ids)} samples")
        return None

    caption = segment(video_path, task_name=task_name)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # Save
    out_path = video_path.parent / f"caption_{OUTPUT_SUFFIX}.json"
    out_path.write_text(json.dumps(caption, ensure_ascii=False, indent=2))
    log.info(f"[{name}] Saved: {out_path}")

    # Print summary
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"  Scene: {caption.get('scene', 'unknown')}")
    task = caption["tasks"][0]
    print(f"  Task: {task['instruction']}")
    print(f"  Segments: {len(task['atomic_actions'])}")
    for i, a in enumerate(task["atomic_actions"]):
        s, e = a["frame_interval"]
        dur = (e - s) / fps
        print(f"    Step {i + 1}: [{s:>4d}, {e:>4d}] ({dur:5.1f}s) — {a['caption']}")
    print(f"{'─' * 60}")

    # Preview
    if preview:
        preview_path = generate_preview(video_path, caption, fps)
        log.info(f"[{name}] Preview: {preview_path}")

    return caption


# ═══════════════════════════════════════════════════════════════════════════════
# Preview Generation
# ═══════════════════════════════════════════════════════════════════════════════

PREVIEW_W = 640  # Preview output width (height auto-scaled)


def generate_preview(video_path: Path, caption: dict, fps: float) -> Path:
    """Generate preview video with subtitle overlay, rotation correction, and 640p downscale."""
    rgb_path = str(video_path)
    out_dir = video_path.parent
    out_path = out_dir / f"preview_caption_{OUTPUT_SUFFIX}.mp4"
    tmp_path = out_dir / f"preview_caption_{OUTPUT_SUFFIX}.tmp.mp4"

    try:
        from ..video_utils import get_manual_rotation, apply_rotation as _apply_rot
    except ImportError:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from video_utils import get_manual_rotation, apply_rotation as _apply_rot

    rotation = get_manual_rotation(rgb_path)
    cap = cv2.VideoCapture(rgb_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # After rotation, dimensions may swap for 90/270
    if rotation in (90, 270):
        W, H = H, W

    # Scale to PREVIEW_W, keep aspect ratio
    scale = PREVIEW_W / W
    out_w = PREVIEW_W
    out_h = int(H * scale)
    # Ensure even dimensions for codec compatibility
    out_h = out_h if out_h % 2 == 0 else out_h + 1

    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    tasks = caption.get("tasks", [])
    actions = tasks[0].get("atomic_actions", []) if tasks else []

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = _apply_rot(frame, rotation)
        frame = cv2.resize(frame, (out_w, out_h))
        for a in actions:
            s, e = a["frame_interval"]
            if s <= fidx < e:
                step_idx = actions.index(a) + 1
                text = f"[Step {step_idx}] {a.get('caption', a.get('instruction', ''))}"
                frame = _render_subtitle(frame, text)
                break
        writer.write(frame)
        fidx += 1

    cap.release()
    writer.release()

    if tmp_path.exists():
        tmp_path.rename(out_path)
    return out_path


def _wrap_text_by_width(text: str, max_width: int, measure_fn) -> list[str]:
    """Word-wrap text so each line fits within max_width pixels."""
    if not text:
        return []
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            continue
        tokens = re.findall(r"\S+\s*", paragraph)
        current = ""
        for token in tokens:
            candidate = f"{current}{token}".rstrip()
            if current and measure_fn(candidate) > max_width:
                lines.append(current.rstrip())
                current = token.lstrip()
            else:
                current = f"{current}{token}"
        if current.strip():
            lines.append(current.rstrip())
    return lines or [text]


try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Cache loaded font across calls
_subtitle_font_cache: dict = {}


def _get_pil_font(size: int):
    """Load Inter-Bold or fallback font, cached."""
    if size in _subtitle_font_cache:
        return _subtitle_font_cache[size]
    font = None
    if _PIL_AVAILABLE:
        candidates = [
            Path(__file__).parent.parent / "assets" / "fonts" / "Inter-Bold.ttf",
            Path("/usr/share/fonts/truetype/inter/Inter-Bold.ttf"),
            Path("/usr/share/fonts/opentype/inter/Inter-Bold.otf"),
            Path("/usr/local/share/fonts/Inter-Bold.ttf"),
        ]
        for p in candidates:
            if p.exists():
                try:
                    font = ImageFont.truetype(str(p), size)
                    break
                except Exception:
                    continue
    _subtitle_font_cache[size] = font
    return font


def _render_subtitle(frame: np.ndarray, text: str) -> np.ndarray:
    """Render subtitle with auto-wrap, shadow, and translucent background.

    Follows the style from x_ego/steps/preview/standard.py.
    """
    h, w = frame.shape[:2]

    # Style parameters (proportional to frame size)
    font_size = max(16, int(h * 0.06))
    side_margin = max(16, int(w * 0.06))
    bottom_margin = max(12, int(h * 0.04))
    box_pad_x = max(8, int(w * 0.03))
    box_pad_y = max(6, int(h * 0.015))
    line_spacing = max(3, int(h * 0.008))
    text_opacity = 0.70
    shadow_opacity = 0.30
    shadow_offset = max(1, int(min(w, h) * 0.0025))

    max_text_width = w - 2 * side_margin - 2 * box_pad_x
    pil_font = _get_pil_font(font_size)

    if pil_font and _PIL_AVAILABLE:
        # ── PIL path (better font rendering) ──
        dummy = Image.new("RGB", (w, h))
        draw_dummy = ImageDraw.Draw(dummy)

        def measure(s):
            bbox = draw_dummy.textbbox((0, 0), s, font=pil_font)
            return bbox[2] - bbox[0]

        lines = _wrap_text_by_width(text, max_text_width, measure)
        if not lines:
            return frame

        sample_bbox = draw_dummy.textbbox((0, 0), "Ag", font=pil_font)
        line_h = sample_bbox[3] - sample_bbox[1]
        total_text_h = len(lines) * line_h + (len(lines) - 1) * line_spacing
        max_line_w = max(measure(line) for line in lines)

        box_w = max_line_w + 2 * box_pad_x
        box_h = total_text_h + 2 * box_pad_y
        box_x = max(side_margin, (w - box_w) // 2)
        box_y = max(side_margin, h - bottom_margin - box_h)

        base_frame = frame.copy()

        # Shadow layer
        shadow_img = Image.fromarray(cv2.cvtColor(base_frame.copy(), cv2.COLOR_BGR2RGB))
        draw_shadow = ImageDraw.Draw(shadow_img)
        ty = box_y + box_pad_y
        for line in lines:
            lw = measure(line)
            tx = box_x + (box_w - lw) // 2
            draw_shadow.text((tx + shadow_offset, ty + shadow_offset), line, font=pil_font, fill=(0, 0, 0))
            ty += line_h + line_spacing

        shadow_frame = cv2.cvtColor(np.array(shadow_img), cv2.COLOR_RGB2BGR)
        frame_with_shadow = cv2.addWeighted(shadow_frame, shadow_opacity, base_frame, 1.0 - shadow_opacity, 0)

        # Text layer
        pil_img = Image.fromarray(cv2.cvtColor(frame_with_shadow, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        ty = box_y + box_pad_y
        for line in lines:
            lw = measure(line)
            tx = box_x + (box_w - lw) // 2
            draw.text((tx, ty), line, font=pil_font, fill=(255, 255, 255))
            ty += line_h + line_spacing

        text_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return cv2.addWeighted(text_frame, text_opacity, frame_with_shadow, 1.0 - text_opacity, 0)

    else:
        # ── OpenCV fallback ──
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, h / 800.0)
        thick = max(1, int(scale * 2))

        def measure(s):
            (lw, _), _ = cv2.getTextSize(s, font, scale, thick)
            return lw

        lines = _wrap_text_by_width(text, max_text_width, measure)
        if not lines:
            return frame

        (_, line_h), baseline = cv2.getTextSize("Ag", font, scale, thick)
        line_h += baseline
        total_text_h = len(lines) * line_h + (len(lines) - 1) * line_spacing
        max_line_w = max(measure(line) for line in lines)

        box_w = max_line_w + 2 * box_pad_x
        box_h = total_text_h + 2 * box_pad_y
        box_x = max(side_margin, (w - box_w) // 2)
        box_y = max(side_margin, h - bottom_margin - box_h)

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        ty = box_y + box_pad_y + line_h
        for line in lines:
            lw = measure(line)
            tx = box_x + (box_w - lw) // 2
            cv2.putText(frame, line, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
            ty += line_h + line_spacing

        return frame


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global WINDOW_SEC, STEP_SEC, FRAMES_PER_WINDOW
    import argparse

    p = argparse.ArgumentParser(description="Video segmentation via VLM sliding-window analysis")
    p.add_argument("videos", nargs="+", type=str,
                   help="Path(s) to mp4 video file(s)")
    p.add_argument("--task-name", type=str, default=None,
                   help="Optional task description for VLM context (e.g. 'water pouring')")
    p.add_argument("--preview", action="store_true",
                   help="Generate preview videos with subtitle overlay")
    p.add_argument("--dry-run", action="store_true",
                   help="Show windowing plan without VLM calls")
    p.add_argument("--window-sec", type=float, default=WINDOW_SEC,
                   help=f"Window duration in seconds (default: {WINDOW_SEC})")
    p.add_argument("--step-sec", type=float, default=STEP_SEC,
                   help=f"Step between windows in seconds (default: {STEP_SEC})")
    p.add_argument("--frames-per-window", type=int, default=FRAMES_PER_WINDOW,
                   help=f"Frames sampled per window (default: {FRAMES_PER_WINDOW})")
    args = p.parse_args()

    # Apply config overrides
    WINDOW_SEC = args.window_sec
    STEP_SEC = args.step_sec
    FRAMES_PER_WINDOW = args.frames_per_window

    # Validate input files
    video_paths = []
    for v in args.videos:
        vp = Path(v)
        if not vp.exists():
            log.error(f"Video file not found: {vp}")
            continue
        if not vp.suffix.lower() == ".mp4":
            log.warning(f"Skipping non-mp4 file: {vp}")
            continue
        video_paths.append(vp)

    if not video_paths:
        log.error("No valid mp4 files provided.")
        return

    log.info(f"Processing {len(video_paths)} video(s)")

    # Process videos
    results = {}

    def _process(vp):
        try:
            caption = process_video(vp, preview=args.preview, dry_run=args.dry_run, task_name=args.task_name)
            return vp.stem, caption
        except Exception as e:
            log.error(f"[{vp.stem}] Failed: {e}", exc_info=True)
            return vp.stem, None

    max_workers = min(4, len(video_paths))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for name, caption in pool.map(_process, video_paths):
            if caption:
                results[name] = caption

    # Summary
    if results:
        print(f"\n{'═' * 60}")
        print(f"  Processed {len(results)} video(s)")
        print(f"  VLM calls: {_token_stats['calls']}")
        print(f"  Tokens: {_token_stats['input_tokens']} in / {_token_stats['output_tokens']} out")
        print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
