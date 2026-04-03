#!/usr/bin/env python3
"""
Video segmentation using video2tasks windowing approach adapted for FusionX.

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
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-13a4a1a0b4464373a339681a07fe121e")
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

OUTPUT_SUFFIX = "v2t"


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


def _get_video_rotation(video_path: str) -> int:
    """Detect video rotation from metadata using ffprobe. Returns degrees (0, 90, 180, 270)."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream_tags=rotate", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=5,
        )
        rot = int(out.stdout.strip()) if out.stdout.strip() else 0
        return rot % 360
    except Exception:
        return 0


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate frame to correct orientation based on metadata rotation."""
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def extract_frames_b64(video_path: str, frame_ids: list[int]) -> list[str]:
    """Extract specific frames from video as base64-encoded JPG strings."""
    rotation = _get_video_rotation(video_path)
    cap = cv2.VideoCapture(video_path)
    results = []
    for fid in sorted(set(frame_ids)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = _apply_rotation(frame, rotation)
        frame = cv2.resize(frame, (TARGET_W, TARGET_H))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        results.append(base64.b64encode(buf).decode("ascii"))
    cap.release()
    return results


def save_frames_as_tmp_jpg(video_path: str, frame_ids: list[int], tmp_dir: str) -> list[str]:
    """Extract frames and save as temporary JPG files. Returns file paths."""
    rotation = _get_video_rotation(video_path)
    cap = cv2.VideoCapture(video_path)
    paths = []
    for i, fid in enumerate(sorted(set(frame_ids))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = _apply_rotation(frame, rotation)
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
    """Build prompt for a single window — no SOP info, pure visual analysis."""
    t_start = window.start_frame / fps
    t_end = window.end_frame / fps

    return f"""\
You are analyzing a segment of an egocentric video of a manual task.
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


def vlm_analyze_window(video_path: str, window: Window, sop: dict, fps: float) -> dict | None:
    """Send window frames to VLM, get transition analysis."""
    from dashscope import MultiModalConversation

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_paths = save_frames_as_tmp_jpg(video_path, window.frame_ids, tmp_dir)
        if not frame_paths:
            return None

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
                    _token_stats["input_tokens"] += getattr(response.usage, "input_tokens", 0)
                    _token_stats["output_tokens"] += getattr(response.usage, "output_tokens", 0)

                raw = response.output.choices[0].message.content
                if not isinstance(raw, str):
                    raw = raw[0]["text"]

                # Parse JSON
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
                else:
                    log.warning(f"  Cannot parse JSON from VLM (attempt {attempt+1})")

            except Exception as e:
                log.warning(f"  VLM call failed (attempt {attempt+1}): {e}")

    return None


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
            instruction = f"Step {i + 1}"

        sop_step_index = i + 1

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


def discover_episodes(
    task_filter: str | None = None,
    episode_filter: str | None = None,
    sop_override: dict | None = None,
) -> list[tuple[Path, dict | None]]:
    """Discover episodes and optionally load their SOPs.

    Returns list of (episode_dir, sop_dict_or_None).
    When *sop_override* is given, every discovered episode uses that SOP.
    Otherwise the function tries to resolve SOP from the episode name prefix;
    episodes without a matching SOP are still included (sop=None).
    """
    episodes = []
    for ep_dir in sorted(DATA_ROOT.iterdir()):
        if not ep_dir.is_dir():
            continue
        if not (ep_dir / "rgb.mp4").exists():
            continue

        name = ep_dir.name

        if episode_filter and name != episode_filter:
            continue

        # Resolve SOP: explicit override > name-prefix lookup > None
        if sop_override is not None:
            sop = sop_override
        else:
            task_type = get_task_type(name)
            if task_filter and (task_type is None or task_type != task_filter):
                continue
            if task_type and (SOP_DIR / TASK_SOP_MAP[task_type]).exists():
                sop = json.loads((SOP_DIR / TASK_SOP_MAP[task_type]).read_text())
            else:
                sop = None

        episodes.append((ep_dir, sop))

    return episodes


def process_episode(episode_dir: Path, sop: dict | None, preview: bool = False, dry_run: bool = False) -> dict | None:
    """Process a single episode: window → VLM → cluster → snap → output."""
    name = episode_dir.name
    rgb_path = str(episode_dir / "rgb.mp4")

    # Build a minimal SOP stub when none is provided
    if sop is None:
        sop = {"task_name": name, "steps": []}

    # Get video info
    cap = cv2.VideoCapture(rgb_path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    duration = nframes / fps
    log.info(f"[{name}] {nframes} frames, {fps:.0f} fps, {duration:.1f}s, task: {sop['task_name']}")

    # Build windows
    windows = build_windows(fps, nframes)
    log.info(f"[{name}] {len(windows)} windows (window={WINDOW_SEC}s, step={STEP_SEC}s, {FRAMES_PER_WINDOW} frames/win)")

    if dry_run:
        for w in windows:
            log.info(f"  Window {w.window_id}: frames [{w.start_frame}, {w.end_frame}] "
                     f"({(w.end_frame - w.start_frame) / fps:.1f}s), {len(w.frame_ids)} samples")
        return None

    # Find IMU valleys
    valleys = find_imu_valleys(episode_dir)

    # Analyze all windows with VLM in parallel
    log.info(f"[{name}] Analyzing {len(windows)} windows in parallel...")
    window_results = [None] * len(windows)

    def _analyze(idx_win):
        idx, w = idx_win
        log.info(f"[{name}] Window {w.window_id}/{len(windows)-1} "
                 f"[{w.start_frame}-{w.end_frame}] ({(w.end_frame-w.start_frame)/fps:.1f}s)")
        result = vlm_analyze_window(rgb_path, w, sop, fps)
        if result:
            log.info(f"[{name}] Window {w.window_id} → {len(result.get('transitions', []))} transitions")
        else:
            log.warning(f"[{name}] Window {w.window_id} → VLM returned no result")
        return idx, result

    with ThreadPoolExecutor(max_workers=min(8, len(windows))) as pool:
        for idx, result in pool.map(_analyze, enumerate(windows)):
            window_results[idx] = result

    # Build segments via cut clustering
    segments = build_segments_via_cuts(windows, window_results, fps, nframes, sop, valleys)

    # Build output caption
    task_type = get_task_type(name)
    sop_file = str(Path("sop") / TASK_SOP_MAP[task_type]) if task_type else None
    caption = {
        "instruction": sop["task_name"],
        "sop_file": sop_file,
        "method": "v2t_windowing",
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

    # Preview
    if preview:
        preview_path = generate_preview(episode_dir, caption, fps)
        log.info(f"[{name}] Preview: {preview_path}")

    return caption


# ═══════════════════════════════════════════════════════════════════════════════
# Preview Generation
# ═══════════════════════════════════════════════════════════════════════════════

PREVIEW_W = 640  # Preview output width (height auto-scaled)


def generate_preview(episode_dir: Path, caption: dict, fps: float) -> Path:
    """Generate preview video with subtitle overlay, rotation correction, and 640p downscale."""
    rgb_path = str(episode_dir / "rgb.mp4")
    out_path = episode_dir / f"preview_caption_{OUTPUT_SUFFIX}.mp4"
    tmp_path = episode_dir / f"preview_caption_{OUTPUT_SUFFIX}.tmp.mp4"

    rotation = _get_video_rotation(rgb_path)
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

    actions = caption.get("atomic_action", [])

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = _apply_rotation(frame, rotation)
        frame = cv2.resize(frame, (out_w, out_h))
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
    global WINDOW_SEC, STEP_SEC, FRAMES_PER_WINDOW, SNAP_RADIUS_FRAMES
    import argparse

    p = argparse.ArgumentParser(description="Video segmentation using video2tasks windowing approach")
    p.add_argument("--data-root", type=str, default=None,
                   help=f"Data directory containing episode folders (default: {DATA_ROOT})")
    p.add_argument("--sop", type=str, default=None,
                   help="Path to SOP JSON file (applies to all episodes; omit for auto-detect or no-SOP mode)")
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
    args = p.parse_args()

    # Apply config overrides
    WINDOW_SEC = args.window_sec
    STEP_SEC = args.step_sec
    FRAMES_PER_WINDOW = args.frames_per_window
    SNAP_RADIUS_FRAMES = args.snap_radius

    if args.data_root:
        DATA_ROOT = Path(args.data_root)

    # Load explicit SOP if provided
    sop_override = None
    if args.sop:
        sop_path = Path(args.sop)
        if not sop_path.exists():
            log.error(f"SOP file not found: {sop_path}")
            return
        sop_override = json.loads(sop_path.read_text(encoding="utf-8"))

    # Discover episodes
    episodes = discover_episodes(task_filter=args.task, episode_filter=args.episode, sop_override=sop_override)
    if not episodes:
        log.error("No episodes found. Check --task or --episode filter.")
        return

    log.info(f"Found {len(episodes)} episode(s) to process")

    # Process episodes in parallel
    results = {}

    def _process(ep_sop):
        ep_dir, sop = ep_sop
        try:
            caption = process_episode(ep_dir, sop, preview=args.preview, dry_run=args.dry_run)
            return ep_dir.name, caption
        except Exception as e:
            log.error(f"[{ep_dir.name}] Failed: {e}", exc_info=True)
            return ep_dir.name, None

    max_episodes = min(4, len(episodes))  # Cap episode-level concurrency
    with ThreadPoolExecutor(max_workers=max_episodes) as pool:
        for name, caption in pool.map(_process, episodes):
            if caption:
                results[name] = caption

    # Summary
    if results:
        print(f"\n{'═' * 60}")
        print(f"  Processed {len(results)} episodes")
        print(f"  VLM calls: {_token_stats['calls']}")
        print(f"  Tokens: {_token_stats['input_tokens']} in / {_token_stats['output_tokens']} out")
        print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
