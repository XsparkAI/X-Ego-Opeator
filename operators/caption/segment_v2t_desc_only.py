#!/usr/bin/env python3
"""
Video segmentation using video2tasks windowing approach — SOP description-only variant.

Difference from segment_v2t.py: prompt includes SOP step descriptions (no visual_cue/completion_cue).

Core algorithm (from ~/Desktop/zijian/ego/Ego_Pipeline/video2tasks):
  1. Split video into overlapping windows, sample N frames per window
  2. Send frames to VLM with SOP context → detect transitions per window
  3. Cluster transitions across windows using Hanning-weighted voting
  4. Snap final cut points to IMU acceleration valleys (FusionX enhancement)
  5. Assemble segments with SOP step labels

Usage:
  python segment_v2t.py                           # Process all episodes
  python segment_v2t.py --task waterpour           # Process only waterpour tasks
  python segment_v2t.py --episode waterpour1       # Process single episode
  python segment_v2t.py --preview                  # Generate preview videos
  python segment_v2t.py --dry-run                  # Show plan without VLM calls
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
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("dashscope").setLevel(logging.CRITICAL)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_ROOT = SCRIPT_DIR / "FusionX-Multimodal-Sample-Data-V2"
SOP_DIR = SCRIPT_DIR / "sop"

# Task name → SOP file mapping (derived from episode directory prefix)
TASK_SOP_MAP = {
    "box_cut": "box_cut.json",
    "drill": "drill.json",
    "screwunscrew": "screwunscrew.json",
    "syringe": "syringe.json",
    "waterpour": "waterpour.json",
}

# ── VLM Config ───────────────────────────────────────────────────────────────
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MODEL = "qwen3.5-plus"

# ── Windowing Config (from video2tasks) ──────────────────────────────────────
WINDOW_SEC = 10.0          # Window duration in seconds
STEP_SEC = 5.0             # Step between windows (overlap = window - step)
FRAMES_PER_WINDOW = 12     # Number of frames sampled per window
TARGET_W = 640             # Resize width for VLM
TARGET_H = 480             # Resize height for VLM

# ── IMU Config ───────────────────────────────────────────────────────────────
SMOOTH_WINDOW = 31
SMOOTH_POLYORDER = 3
VALLEY_DISTANCE = 15
VALLEY_PROMINENCE = 0.3
SNAP_RADIUS_FRAMES = 45

# ── Segmentation Config ─────────────────────────────────────────────────────
MIN_SEGMENT_SEC = 0.8      # Minimum segment duration
CLUSTER_SEC = 2.0          # Cluster radius for merging nearby cuts

# ── Token tracking ───────────────────────────────────────────────────────────
_token_stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

# ── Per-phase metrics (for benchmark) ────────────────────────────────────────
import time

@dataclass
class PhaseMetrics:
    """Timing and token metrics for a single processing phase."""
    elapsed_sec: float = 0.0
    vlm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

@dataclass
class EpisodeMetrics:
    """Complete benchmark metrics for one episode."""
    episode: str = ""
    video_duration_sec: float = 0.0
    video_frames: int = 0
    video_fps: float = 0.0
    n_windows: int = 0
    n_segments: int = 0
    windowing: PhaseMetrics = field(default_factory=PhaseMetrics)   # VLM window analysis
    clustering: PhaseMetrics = field(default_factory=PhaseMetrics)  # cut clustering + IMU snap
    refine: PhaseMetrics = field(default_factory=PhaseMetrics)      # VLM text refine
    preview: PhaseMetrics = field(default_factory=PhaseMetrics)     # preview video generation
    total_sec: float = 0.0

OUTPUT_SUFFIX = "v2t_desc"


# ═══════════════════════════════════════════════════════════════════════════════
# Windowing (adapted from video2tasks/src/video2tasks/server/windowing.py)
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
    cap = cv2.VideoCapture(video_path)
    results = []
    for fid in sorted(set(frame_ids)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (TARGET_W, TARGET_H))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        results.append(base64.b64encode(buf).decode("ascii"))
    cap.release()
    return results


def save_frames_as_tmp_jpg(video_path: str, frame_ids: list[int], tmp_dir: str) -> list[str]:
    """Extract frames and save as temporary JPG files. Returns file paths."""
    cap = cv2.VideoCapture(video_path)
    paths = []
    for i, fid in enumerate(sorted(set(frame_ids))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (TARGET_W, TARGET_H))
        path = os.path.join(tmp_dir, f"frame_{i:03d}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(path)
    cap.release()
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# IMU Valley Detection (from FusionX experiments)
# ═══════════════════════════════════════════════════════════════════════════════

def find_imu_valleys(episode_dir: Path) -> np.ndarray | None:
    """Find IMU acceleration valleys as candidate boundaries.

    Returns valley frame indices, or None if no IMU data available.
    """
    parquet_path = episode_dir / "frames.parquet"
    if not parquet_path.exists():
        return None

    try:
        df = pd.read_parquet(parquet_path, columns=["lh_imu_accel", "rh_imu_accel"])
        lh = np.stack(df["lh_imu_accel"].values)
        rh = np.stack(df["rh_imu_accel"].values)
        lh_dynamic = lh - lh.mean(axis=0)
        rh_dynamic = rh - rh.mean(axis=0)
        accel_mag = np.linalg.norm(lh_dynamic, axis=1) + np.linalg.norm(rh_dynamic, axis=1)

        n = len(accel_mag)
        win = min(SMOOTH_WINDOW, n if n % 2 == 1 else n - 1)
        smooth = savgol_filter(accel_mag, window_length=win, polyorder=SMOOTH_POLYORDER)
        valleys, _ = find_peaks(-smooth, distance=VALLEY_DISTANCE, prominence=VALLEY_PROMINENCE)
        log.info(f"  IMU: {len(valleys)} valleys detected")
        return valleys
    except Exception as e:
        log.warning(f"  IMU read failed: {e}")
        return None


def snap_to_valley(frame: int, valleys: np.ndarray | None, radius: int = SNAP_RADIUS_FRAMES) -> int:
    """Snap a frame to the nearest IMU valley within radius."""
    if valleys is None or len(valleys) == 0:
        return frame
    distances = np.abs(valleys - frame)
    nearest_idx = np.argmin(distances)
    if distances[nearest_idx] <= radius:
        return int(valleys[nearest_idx])
    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# VLM Prompt & Inference
# ═══════════════════════════════════════════════════════════════════════════════

def build_window_prompt(sop: dict, n_images: int, window: Window, fps: float) -> str:
    """Build prompt with SOP description-only (no visual_cue / completion_cue)."""
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


def vlm_analyze_window(video_path: str, window: Window, sop: dict, fps: float) -> tuple[dict | None, int, int]:
    """Send window frames to VLM, get transition analysis.

    Returns (result_dict, input_tokens, output_tokens).
    """
    from dashscope import MultiModalConversation

    in_tok, out_tok = 0, 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_paths = save_frames_as_tmp_jpg(video_path, window.frame_ids, tmp_dir)
        if not frame_paths:
            return None, 0, 0

        prompt = build_window_prompt(sop, len(frame_paths), window, fps)

        content = [{"image": f"file://{p}"} for p in frame_paths]
        content.append({"text": prompt})

        messages = [{"role": "user", "content": content}]

        for attempt in range(3):
            try:
                response = MultiModalConversation.call(
                    api_key=API_KEY, model=MODEL, messages=messages,
                )
                if response.status_code != 200:
                    log.warning(f"  VLM API error (attempt {attempt+1}): {response.code} - {response.message}")
                    continue

                # Track tokens
                if hasattr(response, "usage") and response.usage:
                    _token_stats["calls"] += 1
                    cur_in = getattr(response.usage, "input_tokens", 0)
                    cur_out = getattr(response.usage, "output_tokens", 0)
                    _token_stats["input_tokens"] += cur_in
                    _token_stats["output_tokens"] += cur_out
                    in_tok += cur_in
                    out_tok += cur_out

                raw = response.output.choices[0].message.content
                if not isinstance(raw, str):
                    raw = raw[0]["text"]

                # Parse JSON
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result, in_tok, out_tok
                else:
                    log.warning(f"  Cannot parse JSON from VLM (attempt {attempt+1})")

            except Exception as e:
                log.warning(f"  VLM call failed (attempt {attempt+1}): {e}")

    return None, in_tok, out_tok


# ═══════════════════════════════════════════════════════════════════════════════
# Cut Clustering (from video2tasks/src/video2tasks/server/windowing.py)
# ═══════════════════════════════════════════════════════════════════════════════

def build_segments_via_cuts(
    windows: list[Window],
    window_results: list[dict | None],
    fps: float,
    nframes: int,
    sop: dict,
    valleys: np.ndarray | None,
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
        instruction = sop["task_name"]
        if window_instructions:
            instruction = window_instructions[0][2]
        return [{
            "step": 1,
            "frame_interval": [0, nframes],
            "instruction": instruction,
            "sop_step_index": 1,
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
        # Snap to IMU valley
        avg_frame = snap_to_valley(avg_frame, valleys)
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
            # Fallback: use SOP step description
            step_idx = min(i + 1, len(sop["steps"]))
            sop_step = next((s for s in sop["steps"] if s["step"] == step_idx), None)
            instruction = sop_step["description"] if sop_step else f"Step {step_idx}"

        # Match to closest SOP step
        sop_step_index = min(i + 1, len(sop["steps"]))

        segments.append({
            "step": i,
            "frame_interval": [seg_start, seg_end],
            "instruction": instruction,
            "sop_step_index": sop_step_index,
            "confidence": min(1.0, len([c for c in clusters if i < len(clusters)]) / n_windows) if clusters else 0.5,
        })

    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# Episode Discovery & Processing
# ═══════════════════════════════════════════════════════════════════════════════

def get_task_type(episode_name: str) -> str | None:
    """Extract task type from episode name (e.g., 'waterpour1' → 'waterpour')."""
    for task in TASK_SOP_MAP:
        if episode_name.startswith(task):
            return task
    return None


def _vlm_refine_segments(segments: list[dict], sop: dict) -> tuple[list[dict], int, int]:
    """Use VLM (text-only) to merge similar steps and refine instruction language based on SOP.

    Returns (refined_segments, input_tokens, output_tokens).

    The VLM receives the raw segments + full SOP, and returns optimized segments with:
    - Similar adjacent steps merged (same SOP step → single segment)
    - Instruction language aligned with SOP descriptions
    - Frame intervals preserved exactly as-is

    Post-validation ensures no frame intervals were altered.
    """

    steps_text = "\n".join(
        f"  Step {s['step']}: {s['description']}"
        for s in sop["steps"]
    )

    segments_text = json.dumps([
        {"seg_id": i, "frame_interval": seg["frame_interval"], "instruction": seg["instruction"]}
        for i, seg in enumerate(segments)
    ], ensure_ascii=False, indent=2)

    prompt = f"""\
You are optimizing video segmentation results for a manual task.

## SOP (Standard Operating Procedure)
Task: {sop['task_name']}
Steps:
{steps_text}

## Raw Segmentation Results
{segments_text}

## Your Task
1. **Map each segment to an SOP step**: Determine which SOP step each raw segment belongs to. Each segment must map to exactly ONE SOP step.
2. **Merge**: If adjacent segments map to the SAME SOP step, merge them into one segment. When merging, use the FIRST segment's start frame and the LAST segment's end frame.
3. **Do NOT merge different SOP steps**: If a single raw segment covers multiple SOP steps, keep it as-is and assign it to the MOST PROMINENT step. Do NOT combine segments that belong to different SOP steps.
4. **Relabel**: Assign the correct `sop_step_index` (1-based, matching the SOP step number). The sequence must be monotonically non-decreasing, and every SOP step should appear if the raw segments cover it.
5. **Rewrite instruction**: Rewrite each instruction to describe ONLY the action of its assigned SOP step. Keep it concise and aligned with the SOP step description. Each instruction should describe one atomic action, not multiple steps combined.

## CRITICAL RULES
- NEVER change any frame_interval values. Only merge adjacent segments by combining their intervals.
- The final segments must cover the same total frame range as the input.
- sop_step_index must be monotonically non-decreasing.
- Each final instruction must describe ONE SOP step only. Never combine descriptions of multiple steps into one instruction.
- Output ONLY a JSON array of objects, each with "frame_interval", "instruction", and "sop_step_index".

Output:"""

    import openai
    _refine_client = openai.OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    in_tok, out_tok = 0, 0

    for attempt in range(3):
        try:
            response = _refine_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )

            _token_stats["calls"] += 1
            if response.usage:
                cur_in = response.usage.prompt_tokens or 0
                cur_out = response.usage.completion_tokens or 0
                _token_stats["input_tokens"] += cur_in
                _token_stats["output_tokens"] += cur_out
                in_tok += cur_in
                out_tok += cur_out

            raw = response.choices[0].message.content or ""
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if not json_match:
                log.warning(f"  Refine: cannot parse JSON (attempt {attempt+1})")
                continue

            refined = json.loads(json_match.group())

            # ── Post-validation: frame intervals must be preserved ──
            # Build set of all original boundary frames
            orig_start = segments[0]["frame_interval"][0]
            orig_end = segments[-1]["frame_interval"][1]
            orig_boundaries = set()
            for seg in segments:
                orig_boundaries.add(seg["frame_interval"][0])
                orig_boundaries.add(seg["frame_interval"][1])

            # Validate refined segments
            valid = True
            for r in refined:
                s, e = r["frame_interval"]
                if s not in orig_boundaries or e not in orig_boundaries:
                    log.warning(f"  Refine: frame_interval [{s}, {e}] not in original boundaries")
                    valid = False
                    break

            if not valid:
                log.warning(f"  Refine: validation failed (attempt {attempt+1}), frame intervals were altered")
                continue

            # Check total coverage
            if refined[0]["frame_interval"][0] != orig_start or refined[-1]["frame_interval"][1] != orig_end:
                log.warning(f"  Refine: total coverage mismatch (attempt {attempt+1})")
                continue

            # Check continuity (no gaps or overlaps)
            continuous = True
            for i in range(1, len(refined)):
                if refined[i]["frame_interval"][0] != refined[i-1]["frame_interval"][1]:
                    continuous = False
                    break
            if not continuous:
                log.warning(f"  Refine: segments not continuous (attempt {attempt+1})")
                continue

            # Check sop_step_index monotonically non-decreasing
            indices = [r["sop_step_index"] for r in refined]
            if indices != sorted(indices):
                log.warning(f"  Refine: sop_step_index not monotonic (attempt {attempt+1})")
                continue

            # All checks passed — convert to segment format
            result = []
            for i, r in enumerate(refined):
                result.append({
                    "frame_interval": r["frame_interval"],
                    "instruction": r["instruction"],
                    "sop_step_index": r["sop_step_index"],
                    "step": i,
                })

            log.info(f"  Refine: {len(segments)} → {len(result)} segments (validated)")
            return result, in_tok, out_tok

        except Exception as e:
            log.warning(f"  Refine VLM failed (attempt {attempt+1}): {e}")

    # Fallback: simple exact-match merge
    log.warning("  Refine: all attempts failed, falling back to exact-match merge")
    return _merge_exact(segments), in_tok, out_tok


def _merge_exact(segments: list[dict]) -> list[dict]:
    """Fallback: merge adjacent segments with identical instruction, re-number steps."""
    if not segments:
        return segments

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        if seg["instruction"] == merged[-1]["instruction"]:
            merged[-1]["frame_interval"] = [merged[-1]["frame_interval"][0], seg["frame_interval"][1]]
        else:
            merged.append(seg.copy())

    for i, seg in enumerate(merged):
        seg["sop_step_index"] = i + 1
        seg["step"] = i

    return merged


def discover_episodes(task_filter: str | None = None, episode_filter: str | None = None) -> list[tuple[Path, dict]]:
    """Discover episodes and load their SOPs.

    Returns list of (episode_dir, sop_dict).
    """
    episodes = []
    for ep_dir in sorted(DATA_ROOT.iterdir()):
        if not ep_dir.is_dir():
            continue
        if not (ep_dir / "rgb.mp4").exists():
            continue

        name = ep_dir.name
        task_type = get_task_type(name)
        if task_type is None:
            log.debug(f"  Skipping {name}: no SOP mapping")
            continue

        if task_filter and task_type != task_filter:
            continue
        if episode_filter and name != episode_filter:
            continue

        sop_path = SOP_DIR / TASK_SOP_MAP[task_type]
        if not sop_path.exists():
            log.warning(f"  Skipping {name}: SOP not found at {sop_path}")
            continue

        sop = json.loads(sop_path.read_text())
        episodes.append((ep_dir, sop))

    return episodes


def process_episode(episode_dir: Path, sop: dict, preview: bool = False, dry_run: bool = False) -> tuple[dict | None, EpisodeMetrics | None]:
    """Process a single episode: window → VLM → cluster → snap → output.

    Returns (caption, metrics). metrics is None only for dry_run.
    """
    name = episode_dir.name
    rgb_path = str(episode_dir / "rgb.mp4")
    t_total_start = time.time()

    # Get video info
    cap = cv2.VideoCapture(rgb_path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    duration = nframes / fps
    log.info(f"[{name}] {nframes} frames, {fps:.0f} fps, {duration:.1f}s, task: {sop['task_name']}")

    metrics = EpisodeMetrics(
        episode=name, video_duration_sec=duration,
        video_frames=nframes, video_fps=fps,
    )

    # Build windows
    windows = build_windows(fps, nframes)
    metrics.n_windows = len(windows)
    log.info(f"[{name}] {len(windows)} windows (window={WINDOW_SEC}s, step={STEP_SEC}s, {FRAMES_PER_WINDOW} frames/win)")

    if dry_run:
        for w in windows:
            log.info(f"  Window {w.window_id}: frames [{w.start_frame}, {w.end_frame}] "
                     f"({(w.end_frame - w.start_frame) / fps:.1f}s), {len(w.frame_ids)} samples")
        return None, None

    # Find IMU valleys
    valleys = find_imu_valleys(episode_dir)

    # ── Phase 1: VLM window analysis (parallel) ──
    log.info(f"[{name}] Analyzing {len(windows)} windows in parallel...")
    window_results = [None] * len(windows)
    t_wind_start = time.time()
    wind_in_tok, wind_out_tok = 0, 0
    wind_calls = 0

    def _analyze(idx_win):
        idx, w = idx_win
        log.info(f"[{name}] Window {w.window_id}/{len(windows)-1} "
                 f"[{w.start_frame}-{w.end_frame}] ({(w.end_frame-w.start_frame)/fps:.1f}s)")
        result, w_in, w_out = vlm_analyze_window(rgb_path, w, sop, fps)
        if result:
            log.info(f"[{name}] Window {w.window_id} → {len(result.get('transitions', []))} transitions")
        else:
            log.warning(f"[{name}] Window {w.window_id} → VLM returned no result")
        return idx, result, w_in, w_out

    with ThreadPoolExecutor(max_workers=min(8, len(windows))) as pool:
        for idx, result, w_in, w_out in pool.map(_analyze, enumerate(windows)):
            window_results[idx] = result
            wind_in_tok += w_in
            wind_out_tok += w_out
            if result is not None:
                wind_calls += 1

    metrics.windowing = PhaseMetrics(
        elapsed_sec=time.time() - t_wind_start,
        vlm_calls=wind_calls, input_tokens=wind_in_tok, output_tokens=wind_out_tok,
    )

    # ── Phase 2: Cut clustering + IMU snap ──
    t_cluster_start = time.time()
    segments = build_segments_via_cuts(windows, window_results, fps, nframes, sop, valleys)
    metrics.clustering = PhaseMetrics(elapsed_sec=time.time() - t_cluster_start)

    # ── Phase 3: VLM refine ──
    t_refine_start = time.time()
    segments, ref_in, ref_out = _vlm_refine_segments(segments, sop)
    metrics.refine = PhaseMetrics(
        elapsed_sec=time.time() - t_refine_start,
        vlm_calls=1, input_tokens=ref_in, output_tokens=ref_out,
    )

    metrics.n_segments = len(segments)

    # Build output caption
    caption = {
        "instruction": sop["task_name"],
        "sop_file": str(Path("sop") / TASK_SOP_MAP.get(get_task_type(name) or "", "")),
        "method": "v2t_desc_only",
        "atomic_action": [
            {
                "frame_interval": seg["frame_interval"],
                "instruction": seg["instruction"],
                "sop_step_index": seg["sop_step_index"],
            }
            for seg in segments
        ],
    }

    # Save
    out_path = episode_dir / f"caption_{OUTPUT_SUFFIX}.json"
    out_path.write_text(json.dumps(caption, ensure_ascii=False, indent=2))
    log.info(f"[{name}] Saved: {out_path}")

    # Print summary
    print(f"\n{'─' * 60}")
    print(f"  {name}: {caption['instruction']}")
    print(f"  Segments: {len(caption['atomic_action'])}")
    for a in caption["atomic_action"]:
        s, e = a["frame_interval"]
        dur = (e - s) / fps
        print(f"    Step {a['sop_step_index']}: [{s:>4d}, {e:>4d}] ({dur:5.1f}s) — {a['instruction']}")
    print(f"{'─' * 60}")

    # ── Phase 4: Preview ──
    if preview:
        t_prev_start = time.time()
        preview_path = generate_preview(episode_dir, caption, fps)
        metrics.preview = PhaseMetrics(elapsed_sec=time.time() - t_prev_start)
        log.info(f"[{name}] Preview: {preview_path}")

    metrics.total_sec = time.time() - t_total_start
    return caption, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Preview Generation
# ═══════════════════════════════════════════════════════════════════════════════

def _get_video_rotation(video_path: str) -> int:
    """Get video rotation metadata (0, 90, 180, 270) via ffprobe."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_streams", video_path],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if "rotate" in line.lower():
                val = line.split("=")[-1].strip()
                return abs(int(val)) % 360
    except Exception:
        pass
    return 0


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Apply rotation correction to a frame."""
    if rotation == 180:
        return cv2.flip(frame, -1)
    elif rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def generate_preview(episode_dir: Path, caption: dict, fps: float) -> Path:
    """Generate preview video with subtitle overlay."""
    rgb_path = str(episode_dir / "rgb.mp4")
    out_path = episode_dir / f"preview_caption_{OUTPUT_SUFFIX}.mp4"
    tmp_path = episode_dir / f"preview_caption_{OUTPUT_SUFFIX}.tmp.mp4"

    rotation = _get_video_rotation(rgb_path)

    cap = cv2.VideoCapture(rgb_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # If 90/270 rotation, swap W/H for output
    if rotation in (90, 270):
        W, H = H, W
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    actions = caption.get("atomic_action", [])

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if rotation:
            frame = _apply_rotation(frame, rotation)
        for a in actions:
            s, e = a["frame_interval"]
            if s <= fidx < e:
                text = f"[Step {a.get('sop_step_index', '?')}] {a['instruction']}"
                frame = _render_subtitle(frame, text)
                break
        writer.write(frame)
        fidx += 1

    cap.release()
    writer.release()

    if tmp_path.exists():
        tmp_path.rename(out_path)
    return out_path


def _render_subtitle(frame: np.ndarray, text: str) -> np.ndarray:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, h / 1000.0)
    thick = max(1, int(scale * 2))
    margin = int(w * 0.03)
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)

    bx, by = margin, h - 40 - th
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx - 6, by - th - 6), (bx + tw + 6, by + baseline + 6), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    cv2.putText(frame, text, (bx, by), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Report
# ═══════════════════════════════════════════════════════════════════════════════

def _print_benchmark_report(all_metrics: list[EpisodeMetrics]):
    """Print detailed benchmark report with per-phase timing and token stats."""
    print(f"\n{'═' * 90}")
    print(f"  BENCHMARK REPORT — segment_v2t_desc_only")
    print(f"{'═' * 90}")

    # Per-episode table header
    header = (
        f"  {'Episode':<16s} │ {'Video':>6s} │ {'Windows':>4s} │ "
        f"{'VLM Win':>7s} │ {'Cluster':>7s} │ {'Refine':>7s} │ {'Preview':>7s} │ {'Total':>7s} │ {'Segs':>4s}"
    )
    print(header)
    print(f"  {'':─<16s}─┼─{'':─>6s}─┼─{'':─>4s}─┼─{'':─>7s}─┼─{'':─>7s}─┼─{'':─>7s}─┼─{'':─>7s}─┼─{'':─>7s}─┼─{'':─>4s}")

    for m in sorted(all_metrics, key=lambda x: x.episode):
        print(
            f"  {m.episode:<16s} │ {m.video_duration_sec:5.1f}s │ {m.n_windows:4d} │ "
            f"{m.windowing.elapsed_sec:6.1f}s │ {m.clustering.elapsed_sec:6.1f}s │ "
            f"{m.refine.elapsed_sec:6.1f}s │ {m.preview.elapsed_sec:6.1f}s │ "
            f"{m.total_sec:6.1f}s │ {m.n_segments:4d}"
        )

    # Token stats table
    print(f"\n  {'─' * 88}")
    print(f"  TOKEN STATS")
    tok_header = (
        f"  {'Episode':<16s} │ {'VLM Win In':>10s} │ {'VLM Win Out':>11s} │ "
        f"{'Refine In':>9s} │ {'Refine Out':>10s} │ {'Total In':>9s} │ {'Total Out':>10s}"
    )
    print(tok_header)
    print(f"  {'':─<16s}─┼─{'':─>10s}─┼─{'':─>11s}─┼─{'':─>9s}─┼─{'':─>10s}─┼─{'':─>9s}─┼─{'':─>10s}")

    sum_win_in = sum_win_out = sum_ref_in = sum_ref_out = 0
    for m in sorted(all_metrics, key=lambda x: x.episode):
        total_in = m.windowing.input_tokens + m.refine.input_tokens
        total_out = m.windowing.output_tokens + m.refine.output_tokens
        sum_win_in += m.windowing.input_tokens
        sum_win_out += m.windowing.output_tokens
        sum_ref_in += m.refine.input_tokens
        sum_ref_out += m.refine.output_tokens
        print(
            f"  {m.episode:<16s} │ {m.windowing.input_tokens:>10,d} │ {m.windowing.output_tokens:>11,d} │ "
            f"{m.refine.input_tokens:>9,d} │ {m.refine.output_tokens:>10,d} │ "
            f"{total_in:>9,d} │ {total_out:>10,d}"
        )

    # Totals
    print(f"  {'':─<16s}─┼─{'':─>10s}─┼─{'':─>11s}─┼─{'':─>9s}─┼─{'':─>10s}─┼─{'':─>9s}─┼─{'':─>10s}")
    print(
        f"  {'TOTAL':<16s} │ {sum_win_in:>10,d} │ {sum_win_out:>11,d} │ "
        f"{sum_ref_in:>9,d} │ {sum_ref_out:>10,d} │ "
        f"{sum_win_in + sum_ref_in:>9,d} │ {sum_win_out + sum_ref_out:>10,d}"
    )

    # Averages
    n = len(all_metrics)
    avg_video = sum(m.video_duration_sec for m in all_metrics) / n
    avg_total = sum(m.total_sec for m in all_metrics) / n
    avg_ratio = sum(m.total_sec / m.video_duration_sec for m in all_metrics) / n
    print(f"\n  Avg video duration: {avg_video:.1f}s")
    print(f"  Avg processing time: {avg_total:.1f}s")
    print(f"  Avg processing/video ratio: {avg_ratio:.2f}x")
    print(f"  Avg tokens per episode: {(sum_win_in + sum_ref_in) // n:,d} in / {(sum_win_out + sum_ref_out) // n:,d} out")
    print(f"{'═' * 90}")

    # Save benchmark JSON
    benchmark_data = []
    for m in all_metrics:
        benchmark_data.append({
            "episode": m.episode,
            "video_duration_sec": round(m.video_duration_sec, 1),
            "video_frames": m.video_frames,
            "video_fps": round(m.video_fps, 1),
            "n_windows": m.n_windows,
            "n_segments": m.n_segments,
            "timing": {
                "windowing_sec": round(m.windowing.elapsed_sec, 2),
                "clustering_sec": round(m.clustering.elapsed_sec, 2),
                "refine_sec": round(m.refine.elapsed_sec, 2),
                "preview_sec": round(m.preview.elapsed_sec, 2),
                "total_sec": round(m.total_sec, 2),
            },
            "tokens": {
                "windowing": {"input": m.windowing.input_tokens, "output": m.windowing.output_tokens},
                "refine": {"input": m.refine.input_tokens, "output": m.refine.output_tokens},
            },
        })

    report_path = SCRIPT_DIR / "benchmark_report.json"
    report_path.write_text(json.dumps(benchmark_data, indent=2))
    print(f"  Benchmark saved to: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global WINDOW_SEC, STEP_SEC, FRAMES_PER_WINDOW, SNAP_RADIUS_FRAMES
    import argparse

    p = argparse.ArgumentParser(description="Video segmentation using video2tasks windowing approach")
    p.add_argument("--task", type=str, default=None,
                   help="Filter by task type (waterpour, box_cut, drill, screwunscrew, syringe)")
    p.add_argument("--episode", type=str, default=None,
                   help="Process single episode by name (e.g., waterpour1)")
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
    p.add_argument("--snap-radius", type=int, default=SNAP_RADIUS_FRAMES,
                   help=f"IMU valley snap radius in frames (default: {SNAP_RADIUS_FRAMES})")
    p.add_argument("--benchmark", action="store_true",
                   help="Print detailed benchmark report (per-phase timing + token stats)")
    args = p.parse_args()

    # Apply config overrides
    WINDOW_SEC = args.window_sec
    STEP_SEC = args.step_sec
    FRAMES_PER_WINDOW = args.frames_per_window
    SNAP_RADIUS_FRAMES = args.snap_radius

    # Discover episodes
    episodes = discover_episodes(task_filter=args.task, episode_filter=args.episode)
    if not episodes:
        log.error("No episodes found. Check --task or --episode filter.")
        return

    log.info(f"Found {len(episodes)} episode(s) to process")

    # Process episodes in parallel
    results = {}
    all_metrics: list[EpisodeMetrics] = []

    def _process(ep_sop):
        ep_dir, sop = ep_sop
        try:
            caption, metrics = process_episode(ep_dir, sop, preview=args.preview, dry_run=args.dry_run)
            return ep_dir.name, caption, metrics
        except Exception as e:
            log.error(f"[{ep_dir.name}] Failed: {e}", exc_info=True)
            return ep_dir.name, None, None

    max_episodes = min(4, len(episodes))  # Cap episode-level concurrency
    with ThreadPoolExecutor(max_workers=max_episodes) as pool:
        for name, caption, metrics in pool.map(_process, episodes):
            if caption:
                results[name] = caption
            if metrics:
                all_metrics.append(metrics)

    # Summary
    if results:
        print(f"\n{'═' * 60}")
        print(f"  Processed {len(results)} episodes")
        print(f"  VLM calls: {_token_stats['calls']}")
        print(f"  Tokens: {_token_stats['input_tokens']} in / {_token_stats['output_tokens']} out")
        print(f"{'═' * 60}")

    # Benchmark report
    if all_metrics and args.benchmark:
        _print_benchmark_report(all_metrics)


if __name__ == "__main__":
    main()
