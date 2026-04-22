#!/usr/bin/env python3
"""
Multi-task captioning: scene -> tasks -> atomic_actions.

This script keeps the existing segment_v2t variants untouched and provides a
new optional captioning method that outputs the unified caption schema:

{
  "scene": "...",
  "tasks": [
    {
      "instruction": "...",
      "frame_interval": [0, 120],
      "atomic_actions": [{"caption": "...", "frame_interval": [0, 20]}]
    }
  ]
}
"""

from __future__ import annotations

import base64
import json
import logging
import re
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from ..vlm_limit import vlm_api_slot
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vlm_limit import vlm_api_slot

try:
    from .vlm_api import (
        build_multimodal_message,
        collect_batch_chat_requests,
        get_default_model,
        submit_batch_chat_requests,
        submit_batch_chat_requests_async,
        submit_direct_chat_requests,
    )
except ImportError:
    from vlm_api import (
        build_multimodal_message,
        collect_batch_chat_requests,
        get_default_model,
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

MODEL = get_default_model("caption", fallback="qwen3.5-plus")
THINKING_BUDGET = 384
EXTRA_BODY = {"enable_thinking": True, "thinking_budget": THINKING_BUDGET}

TASK_WINDOW_SEC = 12.0
TASK_STEP_SEC = 6.0
TASK_FRAMES_PER_WINDOW = 12
ACTION_WINDOW_SEC = 6.0
ACTION_STEP_SEC = 3.0
ACTION_FRAMES_PER_WINDOW = 8
TARGET_W = 640
TARGET_H = 480
MIN_SEGMENT_SEC = 0.8
CLUSTER_SEC = 2.0


@dataclass
class Window:
    window_id: int
    start_frame: int
    end_frame: int
    frame_ids: list[int]


def normalize_action_caption(text: str | None, default: str) -> str:
    if not text:
        return default
    normalized = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return normalized or default


def read_video_info(video_path: str | Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps, nframes


def build_windows_for_range(
    fps: float,
    start_frame: int,
    end_frame: int,
    *,
    window_sec: float,
    step_sec: float,
    frames_per_window: int,
) -> list[Window]:
    start_frame = max(0, int(start_frame))
    end_frame = max(start_frame, int(end_frame))
    nframes = end_frame - start_frame
    if nframes <= 0:
        return []

    window_frames = max(1, int(round(window_sec * fps)))
    step_frames = max(1, int(round(step_sec * fps)))

    if nframes <= window_frames:
        frame_ids = np.linspace(start_frame, end_frame - 1, min(frames_per_window, nframes), dtype=int).tolist()
        return [Window(0, start_frame, end_frame - 1, frame_ids)]

    windows = []
    wid = 0
    s = start_frame
    while s < end_frame:
        e = min(s + window_frames, end_frame)
        actual_len = e - s
        if actual_len <= 0:
            break
        sample_count = min(frames_per_window, actual_len)
        frame_ids = np.linspace(s, e - 1, sample_count, dtype=int).tolist()
        windows.append(Window(wid, s, e - 1, frame_ids))
        wid += 1
        if e >= end_frame:
            break
        s += step_frames
    return windows


def save_frames_as_tmp_jpg(video_path: str | Path, frame_ids: list[int], tmp_dir: str) -> list[str]:
    try:
        from ..frame_cache.cache_utils import ensure_cached_frame_paths
    except ImportError:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from frame_cache.cache_utils import ensure_cached_frame_paths

    try:
        from ..video_utils import apply_rotation, get_manual_rotation
    except ImportError:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from video_utils import apply_rotation, get_manual_rotation

    cached = ensure_cached_frame_paths(Path(video_path).parent, frame_ids)
    if cached:
        return cached

    video_path = str(video_path)
    rotation = get_manual_rotation(video_path)
    cap = cv2.VideoCapture(video_path)
    paths: list[str] = []
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


def extract_frames_b64(video_path: str, frame_ids: list[int]) -> list[str]:
    try:
        from ..frame_cache.cache_utils import ensure_cached_frame_b64
    except ImportError:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from frame_cache.cache_utils import ensure_cached_frame_b64

    try:
        from ..video_utils import apply_rotation, get_manual_rotation
    except ImportError:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from video_utils import apply_rotation, get_manual_rotation

    cached = ensure_cached_frame_b64(Path(video_path).parent, frame_ids)
    if cached:
        return cached

    rotation = get_manual_rotation(video_path)
    cap = cv2.VideoCapture(video_path)
    results: list[str] = []
    for fid in sorted(set(frame_ids)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = apply_rotation(frame, rotation)
        frame = cv2.resize(frame, (TARGET_W, TARGET_H))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            results.append(base64.b64encode(buf).decode("ascii"))
    cap.release()
    return results


def _extract_json(raw: Any) -> dict | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raw = raw[0]["text"]
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def build_task_prompt(n_images: int, window: Window, fps: float) -> str:
    return f"""\
You are analyzing an egocentric video segment from {window.start_frame / fps:.1f}s to {window.end_frame / fps:.1f}s.
You are given {n_images} frames sampled evenly from this segment.

Goal:
- Detect task-level boundaries inside this segment.
- A task is a coherent multi-step objective such as make tea, wash dishes, assemble a part, move a box.
- Adjacent manipulations that serve the same objective should stay in the same task.
- instruction should be one short sentence describing the full task segment.

IMPORTANT:
- Frame indices are 0 to {n_images - 1}.
- Output only confident transitions.
- len(tasks) must equal len(transitions) + 1.

Output ONLY valid JSON:
{{
  "thought": "brief reasoning",
  "transitions": [2, 6],
  "tasks": [
    {{"instruction": "fill the kettle and heat water"}},
    {{"instruction": "place the cup and tea bag on the table"}},
    {{"instruction": "pour the hot water into the cup"}}
  ]
}}"""


def build_action_prompt(n_images: int, window: Window, fps: float, task_instruction: str) -> str:
    return f"""\
You are analyzing an egocentric video segment from {window.start_frame / fps:.1f}s to {window.end_frame / fps:.1f}s.
This segment belongs to task: "{task_instruction}".
You are given {n_images} frames sampled evenly from this segment.

Goal:
- Detect atomic action boundaries inside this task.
- Use short verb-style action names such as reach, grasp, lift, place, turn_on, pour, align, inspect.
- If multiple frames still belong to the same atomic action, do not split them.

IMPORTANT:
- Frame indices are 0 to {n_images - 1}.
- Output only confident transitions.
- len(actions) must equal len(transitions) + 1.
- Each action name must be lowercase snake_case.

Output ONLY valid JSON:
{{
  "thought": "brief reasoning",
  "transitions": [1, 4],
  "actions": ["reach", "grasp", "lift"]
}}"""


def run_window_batch(video_path: str | Path, jobs: list[dict[str, Any]]) -> list[dict | None]:
    requests = build_window_requests(video_path, jobs)
    if not requests:
        return [None] * len(jobs)

    with vlm_api_slot():
        responses = submit_batch_chat_requests(requests, model=MODEL, extra_body=EXTRA_BODY)

    return parse_window_results(jobs, responses)


def run_window_direct(
    video_path: str | Path,
    jobs: list[dict[str, Any]],
    *,
    max_workers: int = 8,
) -> list[dict | None]:
    requests = build_window_requests(video_path, jobs)
    if not requests:
        return [None] * len(jobs)

    with vlm_api_slot():
        responses = submit_direct_chat_requests(
            requests,
            model=MODEL,
            extra_body=EXTRA_BODY,
            max_workers=max_workers,
        )
    return parse_window_results(jobs, responses)


def build_window_requests(video_path: str | Path, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests = []
    for job in jobs:
        frames_b64 = extract_frames_b64(str(video_path), job["frame_ids"])
        if not frames_b64:
            continue
        requests.append(
            {
                "custom_id": job["custom_id"],
                "model": MODEL,
                "messages": build_multimodal_message(frames_b64, job["prompt"]),
            }
        )
    return requests


def parse_window_results(
    jobs: list[dict[str, Any]],
    responses: dict[str, dict[str, Any]],
) -> list[dict | None]:
    by_id = {job["custom_id"]: _extract_json(responses.get(job["custom_id"], {}).get("text")) for job in jobs}
    return [by_id.get(job["custom_id"]) for job in jobs]


def _payload_key(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _choose_payload(candidates: list[Any]) -> Any | None:
    if not candidates:
        return None
    counts = Counter(_payload_key(c) for c in candidates)
    best_key, _ = counts.most_common(1)[0]
    return json.loads(best_key)


def build_segments_from_window_results(
    windows: list[Window],
    window_results: list[dict | None],
    *,
    fps: float,
    range_start: int,
    range_end: int,
    payload_field: str,
    default_payload: Any,
) -> list[dict]:
    n_windows = len(windows)
    if n_windows == 0:
        return []

    raw_cuts: list[tuple[int, float]] = []
    payload_timeline: dict[int, list[Any]] = {}
    hanning = np.hanning(n_windows + 2)[1:-1] if n_windows > 1 else np.array([1.0])

    for i, (window, result) in enumerate(zip(windows, window_results)):
        if result is None:
            continue
        transitions = result.get("transitions", [])
        payloads = result.get(payload_field, [])
        n_sampled = len(window.frame_ids)
        if n_sampled == 0:
            continue

        valid_transitions = [int(t) for t in transitions if isinstance(t, (int, float)) and 0 <= int(t) < n_sampled]
        for t_idx in valid_transitions:
            global_frame = window.frame_ids[min(t_idx, n_sampled - 1)]
            raw_cuts.append((global_frame, hanning[i]))

        boundaries = [0] + valid_transitions + [n_sampled - 1]
        for j in range(len(boundaries) - 1):
            if j >= len(payloads):
                continue
            s_local = boundaries[j]
            e_local = boundaries[j + 1]
            payload = payloads[j]
            for k in range(s_local, e_local + 1):
                frame_id = window.frame_ids[min(k, n_sampled - 1)]
                payload_timeline.setdefault(frame_id, []).append(payload)

    if not raw_cuts:
        return [{"start_frame": range_start, "end_frame": range_end, "payload": default_payload}]

    cluster_radius = max(1, int(CLUSTER_SEC * fps))
    raw_cuts.sort(key=lambda x: x[0])
    clusters = []
    current = [raw_cuts[0]]
    for frame, weight in raw_cuts[1:]:
        if frame - current[-1][0] <= cluster_radius:
            current.append((frame, weight))
        else:
            clusters.append(current)
            current = [(frame, weight)]
    clusters.append(current)

    cut_points: list[int] = []
    for cluster in clusters:
        total_w = sum(weight for _, weight in cluster)
        avg_frame = int(sum(frame * weight for frame, weight in cluster) / total_w)
        cut_points.append(avg_frame)

    cut_points = sorted(set(cp for cp in cut_points if range_start < cp < range_end))
    min_frames = max(1, int(MIN_SEGMENT_SEC * fps))
    filtered = []
    prev = range_start
    for cp in cut_points:
        if cp - prev >= min_frames:
            filtered.append(cp)
            prev = cp
    if range_end - prev < min_frames and filtered:
        filtered.pop()

    boundaries = [range_start] + filtered + [range_end]
    segments = []
    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        center_start = seg_start + int(0.4 * (seg_end - seg_start))
        center_end = seg_start + int(0.6 * (seg_end - seg_start))

        candidates = []
        for frame_id in range(center_start, max(center_start + 1, center_end + 1)):
            candidates.extend(payload_timeline.get(frame_id, []))
        if not candidates:
            for frame_id in range(seg_start, seg_end):
                candidates.extend(payload_timeline.get(frame_id, []))

        payload = _choose_payload(candidates) or default_payload
        segments.append({"start_frame": seg_start, "end_frame": seg_end, "payload": payload})

    merged: list[dict] = []
    for seg in segments:
        if merged and _payload_key(merged[-1]["payload"]) == _payload_key(seg["payload"]):
            merged[-1]["end_frame"] = seg["end_frame"]
        else:
            merged.append(seg.copy())
    return merged


def _merge_exact_tasks(task_segments: list[dict]) -> list[dict]:
    if not task_segments:
        return []

    merged = [task_segments[0].copy()]
    for seg in task_segments[1:]:
        prev_instruction = str(merged[-1]["payload"].get("instruction", "")).strip().lower()
        curr_instruction = str(seg["payload"].get("instruction", "")).strip().lower()
        if prev_instruction and prev_instruction == curr_instruction:
            merged[-1]["end_frame"] = seg["end_frame"]
        else:
            merged.append(seg.copy())
    return merged


def _vlm_refine_tasks(
    task_segments: list[dict],
    *,
    batch_enabled: bool = True,
) -> list[dict]:
    """Use text-only VLM to merge similar adjacent task segments."""
    if len(task_segments) <= 1:
        return task_segments

    segments_text = json.dumps(
        [
            {
                "seg_id": i,
                "frame_interval": [seg["start_frame"], seg["end_frame"]],
                "instruction": str(seg["payload"].get("instruction", "")).strip() or "perform the current task",
            }
            for i, seg in enumerate(task_segments)
        ],
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""\
You are optimizing task-level segmentation results for an egocentric video.

## Raw Task Segments
{segments_text}

## Your Task
1. Decide whether adjacent task segments describe the SAME task objective.
2. Merge maximal runs of adjacent segments that describe the same task objective.
   If 2 or more consecutive segments belong to the same task, merge ALL of them into one final segment.
   Use the FIRST segment's start frame and the LAST segment's end frame.
3. Do not stop at pairwise merging. If segments A, B, and C are all the same task, the final output should contain one merged segment [A+C], not partial merges.
4. If two adjacent segments are different tasks, keep them separate.
5. Rewrite each final instruction so it is a concise task-level sentence describing the merged task objective only.

## CRITICAL RULES
- NEVER invent new frame boundaries. You may only reuse existing boundaries from the input.
- The final segments must cover the exact same total frame range as the input.
- The final segments must remain continuous, with no gaps and no overlaps.
- Prefer to output a JSON array of objects with "frame_interval" and "instruction" so the result can be parsed reliably.
- Do not include commentary, markdown, or explanations outside the final structured answer.

Output:"""

    request = {
        "custom_id": "refine_tasks",
        "model": MODEL,
        "messages": build_multimodal_message([], prompt),
    }

    try:
        if batch_enabled:
            responses = submit_batch_chat_requests([request], model=MODEL, extra_body=EXTRA_BODY)
        else:
            responses = submit_direct_chat_requests([request], model=MODEL, extra_body=EXTRA_BODY, max_workers=1)
        raw = responses.get("refine_tasks", {}).get("text") or ""
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not json_match:
            log.warning("Task refine: cannot parse JSON, falling back to exact merge")
            return _merge_exact_tasks(task_segments)

        refined = json.loads(json_match.group())
        if not isinstance(refined, list) or not refined:
            log.warning("Task refine: empty/invalid result, falling back to exact merge")
            return _merge_exact_tasks(task_segments)

        orig_start = task_segments[0]["start_frame"]
        orig_end = task_segments[-1]["end_frame"]
        orig_boundaries = set()
        for seg in task_segments:
            orig_boundaries.add(seg["start_frame"])
            orig_boundaries.add(seg["end_frame"])

        converted: list[dict] = []
        for item in refined:
            frame_interval = item.get("frame_interval")
            instruction = str(item.get("instruction", "")).strip()
            if not isinstance(frame_interval, list) or len(frame_interval) != 2:
                log.warning("Task refine: invalid frame interval, falling back to exact merge")
                return _merge_exact_tasks(task_segments)
            start_frame, end_frame = int(frame_interval[0]), int(frame_interval[1])
            if start_frame not in orig_boundaries or end_frame not in orig_boundaries:
                log.warning("Task refine: non-original boundaries detected, falling back to exact merge")
                return _merge_exact_tasks(task_segments)
            converted.append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "payload": {"instruction": instruction or "perform the current task"},
                }
            )

        if converted[0]["start_frame"] != orig_start or converted[-1]["end_frame"] != orig_end:
            log.warning("Task refine: total coverage mismatch, falling back to exact merge")
            return _merge_exact_tasks(task_segments)
        for i in range(1, len(converted)):
            if converted[i]["start_frame"] != converted[i - 1]["end_frame"]:
                log.warning("Task refine: segments not continuous, falling back to exact merge")
                return _merge_exact_tasks(task_segments)

        return converted
    except Exception as e:
        log.warning("Task refine failed: %s, falling back to exact merge", e)
        return _merge_exact_tasks(task_segments)


def segment(
    video_path: str | Path,
    *,
    task_window_sec: float = TASK_WINDOW_SEC,
    task_step_sec: float = TASK_STEP_SEC,
    task_frames_per_window: int = TASK_FRAMES_PER_WINDOW,
    action_window_sec: float = ACTION_WINDOW_SEC,
    action_step_sec: float = ACTION_STEP_SEC,
    action_frames_per_window: int = ACTION_FRAMES_PER_WINDOW,
    max_workers: int = 8,
    batch_enabled: bool = True,
) -> dict:
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fps, nframes = read_video_info(video_path)

    task_windows = build_windows_for_range(
        fps,
        0,
        nframes,
        window_sec=task_window_sec,
        step_sec=task_step_sec,
        frames_per_window=task_frames_per_window,
    )
    mode_desc = "batch" if batch_enabled else "direct"
    log.info("[%s] Submitting scene + task analysis on %s windows via %s API", video_path.stem, len(task_windows), mode_desc)

    task_jobs = [
        {
            "custom_id": f"task_{window.window_id}",
            "frame_ids": window.frame_ids,
            "prompt": build_task_prompt(len(window.frame_ids), window, fps),
        }
        for window in task_windows
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_scene = pool.submit(
            classify_video_scene if batch_enabled else classify_video_scene_direct,
            video_path,
            fps=fps,
            nframes=nframes,
        )
        future_tasks = pool.submit(
            run_window_batch if batch_enabled else run_window_direct,
            video_path,
            task_jobs,
            **({} if batch_enabled else {"max_workers": max_workers}),
        )
        scene = future_scene.result()
        task_results = future_tasks.result()
    log.info("[%s] Scene: %s", video_path.stem, scene)

    task_segments = build_segments_from_window_results(
        task_windows,
        task_results,
        fps=fps,
        range_start=0,
        range_end=nframes,
        payload_field="tasks",
        default_payload={"instruction": "perform the current task"},
    )
    task_segments = _vlm_refine_tasks(task_segments, batch_enabled=batch_enabled)

    output_tasks = [None] * len(task_segments)

    def _process_task(task_idx_seg: tuple[int, dict]) -> tuple[int, dict]:
        task_idx, task_seg = task_idx_seg
        task_payload = task_seg["payload"] if isinstance(task_seg["payload"], dict) else {}
        task_instruction = str(task_payload.get("instruction") or "perform the current task").strip()
        seg_start = task_seg["start_frame"]
        seg_end = task_seg["end_frame"]

        action_windows = build_windows_for_range(
            fps,
            seg_start,
            seg_end,
            window_sec=action_window_sec,
            step_sec=action_step_sec,
            frames_per_window=action_frames_per_window,
        )
        action_jobs = [
            {
                "custom_id": f"action_{seg_start}_{window.window_id}",
                "frame_ids": window.frame_ids,
                "prompt": build_action_prompt(len(window.frame_ids), window, fps, task_instruction),
            }
            for window in action_windows
        ]
        if action_jobs:
            if batch_enabled:
                action_results = run_window_batch(video_path, action_jobs)
            else:
                action_results = run_window_direct(video_path, action_jobs, max_workers=max_workers)
        else:
            action_results = []

        action_segments = build_segments_from_window_results(
            action_windows,
            action_results,
            fps=fps,
            range_start=seg_start,
            range_end=seg_end,
            payload_field="actions",
            default_payload="act",
        )

        atomic_actions = []
        for action_seg in action_segments:
            action_caption = normalize_action_caption(
                action_seg["payload"] if isinstance(action_seg["payload"], str) else "act",
                "act",
            )
            atomic_actions.append(
                {
                    "caption": action_caption,
                    "frame_interval": [action_seg["start_frame"], action_seg["end_frame"]],
                }
            )

        return task_idx, {
            "instruction": task_instruction,
            "frame_interval": [seg_start, seg_end],
            "atomic_actions": atomic_actions,
        }

    if task_segments:
        with ThreadPoolExecutor(max_workers=min(max(1, max_workers), len(task_segments))) as pool:
            for task_idx, task_output in pool.map(_process_task, enumerate(task_segments)):
                output_tasks[task_idx] = task_output

    return {
        "scene": scene,
        "tasks": [task for task in output_tasks if task is not None],
    }


def process_video(video_path: Path) -> dict:
    caption = segment(video_path)
    out_path = Path(video_path).resolve().parent / "caption_v2t.json"
    out_path.write_text(json.dumps(caption, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Saved: %s", out_path)
    return caption


def submit_segment_job(
    video_path: str | Path,
    *,
    task_window_sec: float = TASK_WINDOW_SEC,
    task_step_sec: float = TASK_STEP_SEC,
    task_frames_per_window: int = TASK_FRAMES_PER_WINDOW,
    action_window_sec: float = ACTION_WINDOW_SEC,
    action_step_sec: float = ACTION_STEP_SEC,
    action_frames_per_window: int = ACTION_FRAMES_PER_WINDOW,
    max_workers: int = 8,
) -> dict:
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fps, nframes = read_video_info(video_path)
    task_windows = build_windows_for_range(
        fps,
        0,
        nframes,
        window_sec=task_window_sec,
        step_sec=task_step_sec,
        frames_per_window=task_frames_per_window,
    )
    task_jobs = [
        {
            "custom_id": f"task_{window.window_id}",
            "frame_ids": window.frame_ids,
            "prompt": build_task_prompt(len(window.frame_ids), window, fps),
        }
        for window in task_windows
    ]
    task_requests = build_window_requests(video_path, task_jobs)

    log.info("[%s] Submitting scene + task analysis on %s windows", video_path.stem, len(task_windows))
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_scene = pool.submit(submit_scene_classification, video_path, fps=fps, nframes=nframes)
        future_tasks = pool.submit(
            lambda: (
                {"batch_id": None, "request_count": 0}
                if not task_requests
                else submit_batch_chat_requests_async(task_requests, model=MODEL, extra_body=EXTRA_BODY)
            )
        )
        scene_submission = future_scene.result()
        task_submission = future_tasks.result()

    return {
        "method": "task_action_v2t",
        "video_path": str(video_path),
        "fps": fps,
        "nframes": nframes,
        "task_window_sec": task_window_sec,
        "task_step_sec": task_step_sec,
        "task_frames_per_window": task_frames_per_window,
        "action_window_sec": action_window_sec,
        "action_step_sec": action_step_sec,
        "action_frames_per_window": action_frames_per_window,
        "max_workers": max_workers,
        "task_windows": [window.__dict__ for window in task_windows],
        "task_jobs": task_jobs,
        "scene_submission": scene_submission,
        "task_submission": task_submission,
    }


def collect_segment_job(state: dict, *, poll_interval_sec: int = 20) -> dict:
    video_path = Path(state["video_path"])
    fps = float(state["fps"])
    nframes = int(state["nframes"])

    task_windows = [Window(**window) for window in state.get("task_windows", [])]
    task_jobs = state.get("task_jobs", [])
    scene_submission = state.get("scene_submission")
    task_submission = state.get("task_submission") or {}

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_scene = pool.submit(
            collect_scene_classification,
            scene_submission,
            poll_interval_sec=poll_interval_sec,
        )
        future_tasks = pool.submit(
            lambda: {} if not task_submission.get("batch_id") else collect_batch_chat_requests(
                task_submission["batch_id"],
                poll_interval_sec=poll_interval_sec,
                wait=True,
            )
        )
        scene = future_scene.result()
        task_result = future_tasks.result()

    if task_submission.get("batch_id") and task_result.get("status") != "completed":
        raise RuntimeError(f"Task batch ended with status: {task_result.get('status')}")

    task_results = parse_window_results(task_jobs, task_result.get("results", {})) if task_jobs else []

    task_segments = build_segments_from_window_results(
        task_windows,
        task_results,
        fps=fps,
        range_start=0,
        range_end=nframes,
        payload_field="tasks",
        default_payload={"instruction": "perform the current task"},
    )

    output_tasks = [None] * len(task_segments)

    def _process_task(task_idx_seg: tuple[int, dict]) -> tuple[int, dict]:
        task_idx, task_seg = task_idx_seg
        task_payload = task_seg["payload"] if isinstance(task_seg["payload"], dict) else {}
        task_instruction = str(task_payload.get("instruction") or "perform the current task").strip()
        seg_start = task_seg["start_frame"]
        seg_end = task_seg["end_frame"]

        action_windows = build_windows_for_range(
            fps,
            seg_start,
            seg_end,
            window_sec=float(state["action_window_sec"]),
            step_sec=float(state["action_step_sec"]),
            frames_per_window=int(state["action_frames_per_window"]),
        )
        action_jobs = [
            {
                "custom_id": f"action_{seg_start}_{window.window_id}",
                "frame_ids": window.frame_ids,
                "prompt": build_action_prompt(len(window.frame_ids), window, fps, task_instruction),
            }
            for window in action_windows
        ]
        action_requests = build_window_requests(video_path, action_jobs)
        if action_requests:
            with vlm_api_slot():
                action_submission = submit_batch_chat_requests_async(action_requests, model=MODEL, extra_body=EXTRA_BODY)
                action_result = collect_batch_chat_requests(
                    action_submission["batch_id"],
                    poll_interval_sec=poll_interval_sec,
                    wait=True,
                )
            if action_result.get("status") != "completed":
                raise RuntimeError(f"Action batch ended with status: {action_result.get('status')}")
            action_results = parse_window_results(action_jobs, action_result.get("results", {}))
        else:
            action_results = []

        action_segments = build_segments_from_window_results(
            action_windows,
            action_results,
            fps=fps,
            range_start=seg_start,
            range_end=seg_end,
            payload_field="actions",
            default_payload="act",
        )

        atomic_actions = []
        for action_seg in action_segments:
            action_caption = normalize_action_caption(
                action_seg["payload"] if isinstance(action_seg["payload"], str) else "act",
                "act",
            )
            atomic_actions.append(
                {
                    "caption": action_caption,
                    "frame_interval": [action_seg["start_frame"], action_seg["end_frame"]],
                }
            )

        return task_idx, {
            "instruction": task_instruction,
            "frame_interval": [seg_start, seg_end],
            "atomic_actions": atomic_actions,
        }

    if task_segments:
        with ThreadPoolExecutor(max_workers=min(max(1, int(state["max_workers"])), len(task_segments))) as pool:
            for task_idx, task_output in pool.map(_process_task, enumerate(task_segments)):
                output_tasks[task_idx] = task_output

    log.info("[%s] Scene: %s", video_path.stem, scene)
    return {
        "scene": scene,
        "tasks": [task for task in output_tasks if task is not None],
    }
