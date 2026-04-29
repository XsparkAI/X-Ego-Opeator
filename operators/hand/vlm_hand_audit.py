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
import time
import threading
from pathlib import Path

import cv2
from ..vlm_limit import vlm_api_slot
from ..caption.vlm_api import (
    build_multimodal_message,
    get_api_key,
    get_default_model,
    provider_supports_batch_api,
    submit_batch_chat_requests,
    submit_direct_chat_requests,
)
from ..frame_cache.frame_provider import FrameProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("dashscope").setLevel(logging.CRITICAL)

# ── VLM Config ──────────────────────────────────────────────────────────────
EXTRA_BODY = {"enable_thinking": False}


def _get_hand_model() -> str:
    return (
        os.getenv("VLM_MODEL", "").strip()
        or os.getenv("VLM_HAND_MODEL", "").strip()
        or get_default_model("hand")
    )

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
# Frame extraction
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


def extract_frames_b64(
    video_path: str,
    frame_ids: list[int],
    *,
    frame_provider: FrameProvider | None = None,
) -> list[str]:
    try:
        from ..frame_cache.cache_utils import ensure_cached_frame_b64
    except ImportError:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from frame_cache.cache_utils import ensure_cached_frame_b64

    try:
        from ..video_utils import get_manual_rotation, apply_rotation
    except ImportError:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from video_utils import get_manual_rotation, apply_rotation

    if frame_provider is not None:
        cached = frame_provider.get_b64(frame_ids)
        if cached:
            return cached

    cached = ensure_cached_frame_b64(Path(video_path).parent, frame_ids)
    if cached:
        return cached

    rotation = get_manual_rotation(video_path)
    cap = cv2.VideoCapture(video_path)
    frames_b64: list[str] = []
    for fid in sorted(set(frame_ids)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = apply_rotation(frame, rotation)
        frame = cv2.resize(frame, (TARGET_W, TARGET_H))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            frames_b64.append(base64.b64encode(buf).decode("ascii"))
    cap.release()
    return frames_b64


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


def _build_frame_requests(frames_b64: list[str], frame_ids: list[int]) -> list[dict]:
    requests = []
    for idx, (image_b64, frame_id) in enumerate(zip(frames_b64, frame_ids)):
        requests.append(
            {
                "custom_id": f"frame_{idx}",
                "model": _get_hand_model(),
                "messages": build_multimodal_message(
                    [image_b64],
                    SINGLE_FRAME_PROMPT,
                ),
                "_meta": {"frame_idx": idx, "global_frame": frame_id},
            }
        )
    return requests


def _run_requests(requests: list[dict], fps: float) -> list[dict]:
    started_at = time.time()
    with vlm_api_slot():
        responses = submit_batch_chat_requests(requests, model=_get_hand_model(), extra_body=EXTRA_BODY)

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

    max_workers = max(1, max_workers)
    results = [None] * len(requests)

    def _submit_one(req_index: int, request: dict) -> dict:
        meta = request["_meta"]
        last_error = None
        for attempt in range(MAX_RETRY_ROUNDS + 1):
            started_at = time.time()
            try:
                with vlm_api_slot():
                    response = submit_direct_chat_requests(
                        [request],
                        model=_get_hand_model(),
                        extra_body=EXTRA_BODY,
                        max_workers=1,
                    )
                payload = response.get(request["custom_id"], {})
                usage = payload.get("usage") or {}
                _add_token_stats(
                    usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0),
                    usage.get("completion_tokens", 0) or usage.get("output_tokens", 0),
                )
                raw = payload.get("text")
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


def _run_single_video_batch(frames_b64: list[str], frame_ids: list[int], fps: float) -> list[dict]:
    requests = _build_frame_requests(frames_b64, frame_ids)
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


def _run_single_video_direct(frames_b64: list[str], frame_ids: list[int], fps: float, max_workers: int) -> list[dict]:
    requests = _build_frame_requests(frames_b64, frame_ids)
    return _run_requests_direct(requests, fps, max_workers=max_workers)


def vlm_hand_audit_parallel(
    video_path: str,
    frame_ids: list[int],
    fps: float,
    max_workers: int,
    batch_enabled: bool = DEFAULT_BATCH_ENABLED,
    frame_provider: FrameProvider | None = None,
) -> list[dict]:
    """Audit sampled frames using either one batch job or direct per-frame requests."""
    if not frame_ids:
        return []
    frames_b64 = extract_frames_b64(
        video_path,
        frame_ids,
        frame_provider=frame_provider,
    )
    if not frames_b64:
        return []
    try:
        effective_batch_enabled = batch_enabled and provider_supports_batch_api()
        if batch_enabled and not effective_batch_enabled:
            log.info("Batch mode requested but current VLM provider does not support batch API; falling back to direct requests")
        if effective_batch_enabled:
            return _run_single_video_batch(frames_b64, frame_ids, fps)
        return _run_single_video_direct(frames_b64, frame_ids, fps, max_workers=max_workers)
    except Exception as e:
        mode = "batch" if (batch_enabled and provider_supports_batch_api()) else "direct"
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
    frame_provider = FrameProvider(video_path)
    effective_batch_enabled = batch_enabled and provider_supports_batch_api()

    log.info(f"Video: {video_path.name}, {total_frames} frames @ {fps:.1f} fps, duration={duration:.1f}s")
    mode_desc = (
        "batch VLM audit (single batch job)"
        if effective_batch_enabled
        else f"direct VLM audit (workers={max_workers})"
    )
    log.info(f"Sampling {len(frame_ids)} frames for {mode_desc}")

    t_start = time.time()
    if frame_ids:
        frame_provider.ensure_profile(frame_ids)
    per_frame = vlm_hand_audit_parallel(
        str(video_path),
        frame_ids,
        fps,
        max_workers,
        batch_enabled=batch_enabled,
        frame_provider=frame_provider,
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
        "total_frames": total_frames,
        "total_frames_sampled": len(frame_ids),
        "valid_responses": n,
        "failed_responses": max(len(frame_ids) - n, 0),
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
            "effective_batch_enabled": effective_batch_enabled,
            "max_workers": max_workers,
            "avg_time_per_frame_sec": round(elapsed / max(len(frame_ids), 1), 3)
        },
        "token_usage": dict(_token_stats),
    }

    result = {
        "summary": summary,
        "frame_results": per_frame,
    }

    log.info("=" * 60)
    log.info(f"Audit complete: {len(frame_ids)} frames, {n} valid, {elapsed:.1f}s total")
    log.info(f"  Throughput: {len(frame_ids)/elapsed:.2f} frames/sec")
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

    if not get_api_key():
        raise SystemExit("VLM API key is not set")

    if args.max_workers != DEFAULT_MAX_WORKERS and not args.no_batch:
        log.info("max-workers is ignored in batch mode")

    result = audit_video(
        args.video,
        frame_step=args.frame_step,
        max_workers=args.max_workers,
        batch_enabled=not args.no_batch,
    )

    out_path = args.output or args.video.with_name("hand_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
