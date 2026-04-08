#!/usr/bin/env python3
"""
VLM-based hand audit for egocentric video - SINGLE BATCH PER VIDEO.

All sampled frames are sent as independent requests within one batch job.
Each response includes ego-hand count, active-manipulation state, and whether
the scene is a single-person operation view.

Usage:
  python vlm_hand_audit_parallel.py --video /path/to/rgb.mp4
  python vlm_hand_audit_parallel.py --video /path/to/rgb.mp4 --max-workers 8
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import re
import tempfile
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from ..vlm_limit import vlm_api_slot
from ..caption.vlm_api import build_multimodal_message, submit_batch_chat_requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("dashscope").setLevel(logging.CRITICAL)

# ── VLM Config ──────────────────────────────────────────────────────────────
MODEL = "qwen3.5-flash"
EXTRA_BODY = {"enable_thinking": False}

# ── Sampling ────────────────────────────────────────────────────────────────
DEFAULT_FRAME_STEP = 30
TARGET_W = 640
TARGET_H = 480
MAX_RETRY_ROUNDS = 1

# ── Parallel Config ─────────────────────────────────────────────────────────
DEFAULT_MAX_WORKERS = 4
DEFAULT_BATCH_ENABLED = True
REQUEST_DELAY_PER_WORKER = 0.3
RETRY_BACKOFF = 1.5

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
        from ..frame_cache.cache_utils import get_cached_frame_paths
    except ImportError:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from frame_cache.cache_utils import get_cached_frame_paths

    try:
        from ..video_utils import get_manual_rotation, apply_rotation
    except ImportError:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from video_utils import get_manual_rotation, apply_rotation

    cached = get_cached_frame_paths(Path(video_path).parent, frame_ids)
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


# ═════════════════════════════════════════════════════════════════════════════
# SINGLE-FRAME Prompt & Inference
# ═════════════════════════════════════════════════════════════════════════════

SINGLE_FRAME_PROMPT = """\
You are labeling an egocentric first-person image.

Task 1:
Count how many camera-wearer's hands are visually present: 0, 1, or 2.

Task 2:
Determine whether the camera-wearer is actively manipulating an object at this exact moment.

Task 3:
Determine whether this frame is a single-person operation view: only the operator is present, and no other person is visible.

Definition:
"Active Manipulation" means the wearer is visibly using their hands to work on, modify, assemble, process, or handle physical objects, materials, or components in pursuit of a specific goal.

Rules for hand count:
• Only count hands directly visible in the image.
• Do not infer hands that are outside the frame or potentially behind objects.
• Ignore hands belonging to other people.
• Any visibility counts (even fingertips).
• If the camera-wearer's hand is fully occluded by an object, do NOT count it.

Rules for active manipulation:
• Do not infer actions that are not visible in the frame.
• If the action is ambiguous or not clearly happening, output "no".
• Ignore objects held by other people.

Rules for single-person operation:
• Output "yes" only when no other person is visible in the frame.
• If any other person's body part is visible, output "no".
• If visibility is ambiguous or uncertain, output "no".

Return ONLY valid JSON in this exact schema:
{"ego_hand_count": 0, "active_manipulation": "no", "single_person_operation": "yes"}

Constraints:
• "ego_hand_count" must be exactly one of 0, 1, 2.
• "active_manipulation" must be exactly "yes" or "no".
• "single_person_operation" must be exactly "yes" or "no".
• No markdown, no explanation, no extra keys.
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
            val = None
            for key in ("hand_count", "ego_hand_count", "count"):
                if key in json_obj:
                    val = json_obj[key]
                    break
            if val in {0, 1, 2}:
                return int(val)
    except Exception:
        pass
    
    match = re.search(r'[^0-9]([012])([^0-9]|$)', text)
    if match:
        return int(match.group(1))
    
    return None


def parse_active_manipulation(text: str) -> bool | None:
    """Extract active-manipulation yes/no from VLM response."""
    if text is None:
        return None

    text = str(text).strip().lower()
    if text in {"yes", "no"}:
        return text == "yes"

    try:
        json_obj = json.loads(text)
        if isinstance(json_obj, dict):
            val = json_obj.get("active_manipulation")
            if isinstance(val, bool):
                return val
            if isinstance(val, str) and val.lower() in {"yes", "no"}:
                return val.lower() == "yes"
    except Exception:
        pass

    match = re.search(r'"active_manipulation"\s*:\s*"(yes|no)"', text)
    if match:
        return match.group(1) == "yes"

    words = re.findall(r"\b(yes|no)\b", text)
    if words:
        return words[0] == "yes"

    return None


def parse_single_person_operation(text: str) -> bool | None:
    """Extract single-person-operation yes/no from VLM response."""
    if text is None:
        return None

    text = str(text).strip().lower()
    try:
        json_obj = json.loads(text)
        if isinstance(json_obj, dict):
            val = json_obj.get("single_person_operation")
            if val is None:
                val = json_obj.get("single_operator_only")
            if isinstance(val, bool):
                return val
            if isinstance(val, str) and val.lower() in {"yes", "no"}:
                return val.lower() == "yes"
    except Exception:
        pass

    match = re.search(r'"single_person_operation"\s*:\s*"(yes|no)"', text)
    if match:
        return match.group(1) == "yes"
    match = re.search(r'"single_operator_only"\s*:\s*"(yes|no)"', text)
    if match:
        return match.group(1) == "yes"

    return None


def _build_frame_requests(frame_paths: list[str], frame_ids: list[int]) -> list[dict]:
    requests = []
    for idx, (image_path, frame_id) in enumerate(zip(frame_paths, frame_ids)):
        image_bytes = Path(image_path).read_bytes()
        requests.append(
            {
                "custom_id": f"frame_{idx}",
                "model": MODEL,
                "messages": build_multimodal_message(
                    [base64.b64encode(image_bytes).decode("ascii")],
                    SINGLE_FRAME_PROMPT,
                ),
                "_meta": {"frame_idx": idx, "global_frame": frame_id},
            }
        )
    return requests


def _create_direct_client():
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    from openai import OpenAI

    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def _extract_response_text(response) -> str | None:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        if texts:
            return "\n".join(texts)
    return str(content) if content is not None else None


def _extract_usage_dict(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0,
    }


def _run_requests(requests: list[dict], fps: float) -> list[dict]:
    started_at = time.time()
    with vlm_api_slot():
        responses = submit_batch_chat_requests(requests, model=MODEL, extra_body=EXTRA_BODY)

    elapsed = time.time() - started_at
    per_frame_elapsed = round(elapsed / max(len(requests), 1), 2)
    results = []
    for req in requests:
        meta = req["_meta"]
        raw = responses.get(req["custom_id"], {}).get("text")
        usage = responses.get(req["custom_id"], {}).get("usage") or {}
        _add_token_stats(
            usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0),
            usage.get("completion_tokens", 0) or usage.get("output_tokens", 0),
        )
        count = parse_hand_count(raw or "")
        active_manipulation = parse_active_manipulation(raw or "")
        single_person_operation = parse_single_person_operation(raw or "")
        success = (
            count is not None
            and active_manipulation is not None
            and single_person_operation is not None
        )
        results.append(
            {
                "frame_idx": meta["frame_idx"],
                "global_frame": meta["global_frame"],
                "time_sec": round(meta["global_frame"] / fps, 3),
                "ego_hand_count": count if count is not None else -1,
                "active_manipulation": active_manipulation,
                "single_person_operation": single_person_operation,
                "inference_time_sec": per_frame_elapsed,
                "success": success,
                "retried": 0,
                "error": None if success else "parse_failed_or_empty",
            }
        )
    return results


def _run_requests_direct(requests: list[dict], fps: float, max_workers: int) -> list[dict]:
    if not requests:
        return []

    client = _create_direct_client()
    max_workers = max(1, max_workers)
    results = [None] * len(requests)

    def _submit_one(req_index: int, request: dict) -> dict:
        meta = request["_meta"]
        last_error = None
        for attempt in range(MAX_RETRY_ROUNDS + 1):
            started_at = time.time()
            try:
                with vlm_api_slot():
                    response = client.chat.completions.create(
                        model=request.get("model", MODEL),
                        messages=request["messages"],
                        extra_body=request.get("extra_body") or EXTRA_BODY,
                    )

                usage = _extract_usage_dict(response)
                _add_token_stats(
                    usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0),
                    usage.get("completion_tokens", 0) or usage.get("output_tokens", 0),
                )
                raw = _extract_response_text(response)
                count = parse_hand_count(raw or "")
                active_manipulation = parse_active_manipulation(raw or "")
                single_person_operation = parse_single_person_operation(raw or "")
                success = (
                    count is not None
                    and active_manipulation is not None
                    and single_person_operation is not None
                )
                result = {
                    "frame_idx": meta["frame_idx"],
                    "global_frame": meta["global_frame"],
                    "time_sec": round(meta["global_frame"] / fps, 3),
                    "ego_hand_count": count if count is not None else -1,
                    "active_manipulation": active_manipulation,
                    "single_person_operation": single_person_operation,
                    "inference_time_sec": round(time.time() - started_at, 2),
                    "success": success,
                    "retried": attempt,
                    "error": None if success else "parse_failed_or_empty",
                }
                if success:
                    return result
                last_error = result["error"]
            except Exception as e:
                last_error = str(e)

            if attempt < MAX_RETRY_ROUNDS:
                time.sleep(REQUEST_DELAY_PER_WORKER * (RETRY_BACKOFF ** attempt))

        return {
            "frame_idx": meta["frame_idx"],
            "global_frame": meta["global_frame"],
            "time_sec": round(meta["global_frame"] / fps, 3),
            "ego_hand_count": -1,
            "active_manipulation": None,
            "single_person_operation": None,
            "inference_time_sec": round(time.time() - started_at, 2),
            "success": False,
            "retried": MAX_RETRY_ROUNDS,
            "error": last_error or "request_failed",
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_submit_one, idx, request): idx
            for idx, request in enumerate(requests)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            results[idx] = future.result()

    dropped_count = sum(
        1 for item in results
        if item["ego_hand_count"] < 0
        or item.get("active_manipulation") is None
        or item.get("single_person_operation") is None
    )
    if dropped_count:
        log.info("Dropping %s hand-audit frames after direct-call retries", dropped_count)
    return [
        item for item in results
        if item["ego_hand_count"] >= 0
        and item.get("active_manipulation") is not None
        and item.get("single_person_operation") is not None
    ]


def _run_single_video_batch(frame_paths: list[str], frame_ids: list[int], fps: float) -> list[dict]:
    requests = _build_frame_requests(frame_paths, frame_ids)
    results = _run_requests(requests, fps)

    for retry_round in range(1, MAX_RETRY_ROUNDS + 1):
        failed = [
            item for item in results
            if item["ego_hand_count"] < 0
            or item.get("active_manipulation") is None
            or item.get("single_person_operation") is None
        ]
        if not failed:
            break
        log.info("Retrying %s hand-audit frames (round %s/%s)", len(failed), retry_round, MAX_RETRY_ROUNDS)
        retry_requests = [requests[item["frame_idx"]] for item in failed]
        retry_results = _run_requests(retry_requests, fps)
        retry_by_idx = {item["frame_idx"]: item for item in retry_results}
        for item in results:
            retried = retry_by_idx.get(item["frame_idx"])
            if retried is not None:
                retried["retried"] = retry_round
                results[item["frame_idx"]] = retried

    dropped_count = sum(
        1 for item in results
        if item["ego_hand_count"] < 0
        or item.get("active_manipulation") is None
        or item.get("single_person_operation") is None
    )
    if dropped_count:
        log.info("Dropping %s hand-audit frames after retry exhaustion", dropped_count)
    return [
        item for item in results
        if item["ego_hand_count"] >= 0
        and item.get("active_manipulation") is not None
        and item.get("single_person_operation") is not None
    ]


def _run_single_video_direct(frame_paths: list[str], frame_ids: list[int], fps: float, max_workers: int) -> list[dict]:
    requests = _build_frame_requests(frame_paths, frame_ids)
    return _run_requests_direct(requests, fps, max_workers=max_workers)


def vlm_hand_audit_parallel(
    video_path: str,
    frame_paths: list[str],
    frame_ids: list[int],
    fps: float,
    max_workers: int,
    batch_enabled: bool = DEFAULT_BATCH_ENABLED,
) -> list[dict]:
    """Audit sampled frames using either one batch job or direct per-frame requests."""
    del video_path
    if not frame_paths or not frame_ids:
        return []
    try:
        if batch_enabled:
            return _run_single_video_batch(frame_paths, frame_ids, fps)
        return _run_single_video_direct(frame_paths, frame_ids, fps, max_workers=max_workers)
    except Exception as e:
        mode = "batch" if batch_enabled else "direct"
        log.error("  Hand %s mode failed: %s", mode, e)
        return []


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def audit_video(
    video_path: str | Path,
    frame_step: int = DEFAULT_FRAME_STEP,
    max_workers: int = DEFAULT_MAX_WORKERS,
    batch_enabled: bool = DEFAULT_BATCH_ENABLED,
) -> dict:
    video_path = Path(video_path)
    with _token_lock:
        _token_stats["calls"] = 0
        _token_stats["input_tokens"] = 0
        _token_stats["output_tokens"] = 0

    frame_ids, fps, total_frames = sample_frames(str(video_path), frame_step)
    duration = total_frames / fps

    log.info(f"Video: {video_path.name}, {total_frames} frames @ {fps:.1f} fps, duration={duration:.1f}s")
    mode_desc = "batch VLM audit (single batch job)" if batch_enabled else f"direct VLM audit (workers={max_workers})"
    log.info(f"Sampling {len(frame_ids)} frames for {mode_desc}")

    t_start = time.time()

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_paths = extract_frames_to_dir(str(video_path), frame_ids, tmp_dir)
        per_frame = vlm_hand_audit_parallel(
            str(video_path),
            frame_paths,
            frame_ids,
            fps,
            max_workers,
            batch_enabled=batch_enabled,
        )

    elapsed = time.time() - t_start

    # ── Summary statistics ──────────────────────────────────────────────────
    valid_results = [
        r for r in per_frame
        if r["ego_hand_count"] >= 0
        and r.get("active_manipulation") is not None
        and r.get("single_person_operation") is not None
    ]
    n = len(valid_results)
    ego_counts = [r["ego_hand_count"] for r in valid_results]
    active_flags = [bool(r["active_manipulation"]) for r in valid_results]
    single_person_flags = [bool(r["single_person_operation"]) for r in valid_results]

    summary = {
        "video": str(video_path),
        "total_frames_sampled": len(per_frame),
        "valid_responses": n,
        "retried_responses": sum(1 for r in per_frame if r.get("retried", 0) > 0),
        "fps": round(fps, 2),
        "duration_sec": round(duration, 2),
        "elapsed_sec": round(elapsed, 2),
        "avg_ego_hand_count": round(sum(ego_counts) / max(n, 1), 3),
        "ego_0_hands_ratio": round(sum(1 for c in ego_counts if c == 0) / max(n, 1), 4),
        "ego_1_hand_ratio": round(sum(1 for c in ego_counts if c == 1) / max(n, 1), 4),
        "ego_2_hands_ratio": round(sum(1 for c in ego_counts if c == 2) / max(n, 1), 4),
        "active_manipulation_ratio": round(sum(1 for v in active_flags if v) / max(n, 1), 4),
        "single_person_operation_ratio": round(sum(1 for v in single_person_flags if v) / max(n, 1), 4),
        "parallel_config": {
            "batch_enabled": batch_enabled,
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
    log.info(f"  Active manipulation ratio: {summary['active_manipulation_ratio']:.2%}")
    log.info(f"  Single-person operation ratio: {summary['single_person_operation_ratio']:.2%}")
    log.info(f"  Tokens: {_token_stats['input_tokens']} in / {_token_stats['output_tokens']} out")
    log.info("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(description="VLM hand audit")
    parser.add_argument("--video", type=Path, required=True, help="Path to video file")
    parser.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP,
                        help=f"Sample every N-th frame (default: {DEFAULT_FRAME_STEP})")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help="Concurrent direct requests when --no-batch is enabled")
    parser.add_argument("--no-batch", action="store_true", help="Disable batch submission and send direct per-frame requests")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY", "").strip():
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    if args.max_workers != DEFAULT_MAX_WORKERS and not args.no_batch:
        log.info("max-workers is ignored in batch mode")

    result = audit_video(
        args.video,
        frame_step=args.frame_step,
        max_workers=args.max_workers,
        batch_enabled=not args.no_batch,
    )

    out_path = args.output or args.video.with_name("vlm_hand_audit_parallel.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
