#!/usr/bin/env python3
"""
VLM-based hand audit for egocentric video - PARALLEL SINGLE-FRAME MODE.

Each frame is sent to VLM independently with parallel execution.
Response must be 0/1/2 only.

Usage:
  python vlm_hand_audit_parallel.py --video /path/to/rgb.mp4
  python vlm_hand_audit_parallel.py --video /path/to/rgb.mp4 --max-workers 8
"""

import argparse
import json
import logging
import os
import re
import tempfile
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

# Optional: progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kw: x  # noqa: E731

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("dashscope").setLevel(logging.CRITICAL)

# ── VLM Config ──────────────────────────────────────────────────────────────
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MODEL = "qwen3.5-flash"

# ── Sampling ────────────────────────────────────────────────────────────────
DEFAULT_FRAME_STEP = 15
TARGET_W = 640
TARGET_H = 480

# ── Parallel Config ─────────────────────────────────────────────────────────
DEFAULT_MAX_WORKERS = 4          # concurrent VLM requests
REQUEST_DELAY_PER_WORKER = 0.3   # minimal delay between requests per worker
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5              # exponential backoff multiplier

# ── Thread-safe token tracking ──────────────────────────────────────────────
_token_stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
_token_lock = threading.Lock()


def _add_token_stats(input_tokens: int, output_tokens: int):
    """Thread-safe update of token statistics."""
    with _token_lock:
        _token_stats["calls"] += 1
        _token_stats["input_tokens"] += input_tokens
        _token_stats["output_tokens"] += output_tokens


# ═════════════════════════════════════════════════════════════════════════════
# Frame extraction (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def sample_frames(video_path: str, frame_step: int) -> tuple[list[int], float, int]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frame_ids = list(range(0, total, max(1, frame_step)))
    if not frame_ids or frame_ids[-1] != total - 1:
        frame_ids.append(total - 1)
    return frame_ids, fps, total


def extract_frames_to_dir(video_path: str, frame_ids: list[int], tmp_dir: str) -> list[str]:
    try:
        from ..video_utils import get_manual_rotation, apply_rotation
    except ImportError:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from video_utils import get_manual_rotation, apply_rotation

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


# ═════════════════════════════════════════════════════════════════════════════
# SINGLE-FRAME Prompt & Inference (parallel-ready)
# ═════════════════════════════════════════════════════════════════════════════

SINGLE_FRAME_PROMPT = """\
You are labeling an egocentric first-person image. Count how many camera-wearer's hands are visually present: 0, 1, or 2.

Rules:
• Only count hands directly visible in the image.
• Do not infer hands that are outside the frame or potentially behind objects.
• Ignore hands belonging to other people.
• Any visibility counts (even fingertips).
• Return ONLY the digit: 0, 1, or 2. No JSON, no words, no punctuation.
• CRITICAL: If the camera-wearer's hand is occluded by an object (e.g., holding a cup, tool, or phone), 
  do NOT count it. If another person's hand enters the frame (e.g., handing over an object), 
  strictly IGNORE it—only count the camera-wearer's hands.
"""


def parse_hand_count(text: str) -> int | None:
    """Extract 0/1/2 from VLM response. Handles JSON, plain text, markdown."""
    if text is None:
        return None
    text = str(text).strip().lower()
    
    if text in {"0", "1", "2"}:
        return int(text)
    
    matches = re.findall(r'\b([012])\b', text)
    if matches:
        return int(matches[0])
    
    try:
        json_obj = json.loads(text)
        if isinstance(json_obj, dict):
            val = json_obj.get("hand_count") or json_obj.get("ego_hand_count") or json_obj.get("count")
            if val in {0, 1, 2}:
                return int(val)
    except:
        pass
    
    match = re.search(r'[^0-9]([012])([^0-9]|$)', text)
    if match:
        return int(match.group(1))
    
    return None


def vlm_audit_single_frame(image_path: str, worker_id: int = 0) -> tuple[int | None, dict]:
    """
    Send one frame to VLM, return (hand_count, metadata).
    Thread-safe: uses global _add_token_stats for token tracking.
    """
    from dashscope import MultiModalConversation

    if not API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    messages = [{
        "role": "user",
        "content": [
            {"image": f"file://{image_path}"},
            {"text": SINGLE_FRAME_PROMPT}
        ]
    }]

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = MultiModalConversation.call(
                api_key=API_KEY, model=MODEL, messages=messages
            )
            if response.status_code != 200:
                err_msg = f"{response.code} - {response.message}"
                log.warning(f"  [Worker-{worker_id}] VLM error (attempt {attempt+1}): {err_msg}")
                last_error = err_msg
                time.sleep(REQUEST_DELAY_PER_WORKER * (RETRY_BACKOFF ** attempt))
                continue

            # Thread-safe token tracking
            if hasattr(response, "usage") and response.usage:
                _add_token_stats(
                    getattr(response.usage, "input_tokens", 0),
                    getattr(response.usage, "output_tokens", 0)
                )

            # Extract text response
            raw = response.output.choices[0].message.content
            if isinstance(raw, list):
                raw = raw[0].get("text", "")
            raw = str(raw).strip()

            count = parse_hand_count(raw)
            if count is not None:
                return count, {"success": True, "raw_response": raw[:50]}
            else:
                log.warning(f"  [Worker-{worker_id}] Parse failed (attempt {attempt+1}), raw: {raw[:80]}")
                last_error = "parse_failed"

        except Exception as e:
            log.warning(f"  [Worker-{worker_id}] Call failed (attempt {attempt+1}): {e}")
            last_error = str(e)
            time.sleep(REQUEST_DELAY_PER_WORKER * (RETRY_BACKOFF ** attempt))
    
    return None, {"success": False, "error": last_error}


def _process_frame_task(args: tuple) -> dict:
    """Wrapper for ThreadPoolExecutor: unpack args and call VLM."""
    idx, image_path, frame_id, fps, worker_id = args
    t0 = time.time()
    
    count, meta = vlm_audit_single_frame(image_path, worker_id)
    
    return {
        "frame_idx": idx,
        "global_frame": frame_id,
        "time_sec": round(frame_id / fps, 3),
        "ego_hand_count": count if count is not None else -1,
        "inference_time_sec": round(time.time() - t0, 2),
        "success": meta["success"],
        "error": meta.get("error")
    }


def vlm_hand_audit_parallel(
    video_path: str,
    frame_paths: list[str],
    frame_ids: list[int],
    fps: float,
    max_workers: int,
) -> list[dict]:
    """Audit frames in parallel using thread pool. Returns list of per-frame results."""
    tasks = [
        (i, fpath, fid, fps, i % max_workers)
        for i, (fpath, fid) in enumerate(zip(frame_paths, frame_ids))
    ]
    
    results = [None] * len(frame_paths)
    progress_iter = tqdm(tasks, desc="Processing frames", unit="frame") if HAS_TQDM else tasks
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_process_frame_task, task): task[0]
            for task in tasks
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                log.error(f"  Frame {idx} task failed: {e}")
                results[idx] = {
                    "frame_idx": idx,
                    "global_frame": frame_ids[idx],
                    "time_sec": round(frame_ids[idx] / fps, 3),
                    "ego_hand_count": -1,
                    "inference_time_sec": 0,
                    "success": False,
                    "error": str(e)
                }
    
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def audit_video(video_path: str | Path, frame_step: int = DEFAULT_FRAME_STEP, max_workers: int = DEFAULT_MAX_WORKERS) -> dict:
    video_path = Path(video_path)
    frame_ids, fps, total_frames = sample_frames(str(video_path), frame_step)
    duration = total_frames / fps

    log.info(f"Video: {video_path.name}, {total_frames} frames @ {fps:.1f} fps, duration={duration:.1f}s")
    log.info(f"Sampling {len(frame_ids)} frames for parallel VLM audit (workers={max_workers})")

    t_start = time.time()

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_paths = extract_frames_to_dir(str(video_path), frame_ids, tmp_dir)
        per_frame = vlm_hand_audit_parallel(str(video_path), frame_paths, frame_ids, fps, max_workers)

    elapsed = time.time() - t_start

    # ── Summary statistics ──────────────────────────────────────────────────
    valid_results = [r for r in per_frame if r["ego_hand_count"] >= 0]
    n = len(valid_results)
    ego_counts = [r["ego_hand_count"] for r in valid_results]

    summary = {
        "video": str(video_path),
        "total_frames_sampled": len(frame_paths),
        "valid_responses": n,
        "failed_responses": len(per_frame) - n,
        "fps": round(fps, 2),
        "duration_sec": round(duration, 2),
        "elapsed_sec": round(elapsed, 2),
        "avg_ego_hand_count": round(sum(ego_counts) / max(n, 1), 3),
        "ego_0_hands_ratio": round(sum(1 for c in ego_counts if c == 0) / max(n, 1), 4),
        "ego_1_hand_ratio": round(sum(1 for c in ego_counts if c == 1) / max(n, 1), 4),
        "ego_2_hands_ratio": round(sum(1 for c in ego_counts if c == 2) / max(n, 1), 4),
        "parallel_config": {
            "max_workers": max_workers,
            "avg_time_per_frame_sec": round(elapsed / max(len(frame_paths), 1), 3)
        },
        "token_usage": dict(_token_stats),
    }

    result = {
        "summary": summary,
        "frame_results": per_frame,
    }

    log.info("=" * 60)
    log.info(f"Audit complete: {len(frame_paths)} frames, {n} valid, {elapsed:.1f}s total")
    log.info(f"  Throughput: {len(frame_paths)/elapsed:.2f} frames/sec")
    log.info(f"  Avg ego hand count: {summary['avg_ego_hand_count']}")
    log.info(f"  Tokens: {_token_stats['input_tokens']} in / {_token_stats['output_tokens']} out")
    log.info("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(description="VLM hand audit (parallel single-frame mode)")
    parser.add_argument("--video", type=Path, required=True, help="Path to video file")
    parser.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP,
                        help=f"Sample every N-th frame (default: {DEFAULT_FRAME_STEP})")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"Concurrent VLM requests (default: {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    # Validate max_workers
    if args.max_workers < 1:
        log.warning(f"max-workers must be >= 1, got {args.max_workers}. Using default {DEFAULT_MAX_WORKERS}")
        args.max_workers = DEFAULT_MAX_WORKERS
    elif args.max_workers > 16:
        log.warning(f"max-workers={args.max_workers} may trigger API rate limits. Consider <= 8")

    result = audit_video(args.video, frame_step=args.frame_step, max_workers=args.max_workers)

    out_path = args.output or args.video.with_name("vlm_hand_audit_parallel.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
