from __future__ import annotations

import base64
import json
import logging
import os
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np

try:
    from ..vlm_limit import vlm_api_slot
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vlm_limit import vlm_api_slot

log = logging.getLogger(__name__)

try:
    from .vlm_api import get_api_key, get_default_model
except ImportError:
    from vlm_api import get_api_key, get_default_model

MODEL = get_default_model("scene", fallback="qwen3.5-flash")
EXTRA_BODY = {"enable_thinking": False}
TARGET_W = 640
TARGET_H = 480
SCENE_FRAMES = 16

SCENE_LABELS = (
    "household",
    "commercial",
    "industrial",
    "medical",
    "public_service",
    "education",
    "outdoor",
    "transportation",
    "entertainment",
    "unknown",
)


def normalize_scene_label(label: str | None) -> str:
    if not label:
        return "unknown"
    normalized = str(label).strip().lower()
    return normalized if normalized in SCENE_LABELS else "unknown"


def sample_scene_frame_ids(nframes: int, num_samples: int = SCENE_FRAMES) -> list[int]:
    if nframes <= 0:
        return []
    return np.linspace(0, nframes - 1, min(num_samples, nframes), dtype=int).tolist()


def save_scene_frames_as_tmp_jpg(video_path: str | Path, frame_ids: list[int], tmp_dir: str) -> list[str]:
    try:
        from ..video_utils import apply_rotation, get_manual_rotation
    except ImportError:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from video_utils import apply_rotation, get_manual_rotation

    video_path = str(video_path)
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
        path = os.path.join(tmp_dir, f"scene_{i:03d}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(path)
    cap.release()
    return paths


def build_scene_prompt(n_images: int, duration_sec: float | None = None) -> str:
    duration_text = f" The full video duration is about {duration_sec:.1f} seconds." if duration_sec else ""
    return f"""\
You are classifying the scene of a full egocentric video using {n_images} frames sampled evenly across the entire video.{duration_text}

Choose exactly one Scene label from this fixed set:
- household
- commercial
- industrial
- medical
- public_service
- education
- outdoor
- transportation
- entertainment
- unknown

Definition:
- Scene means the overall application domain the video most likely belongs to.
- Prefer the dominant context of the full video, not a brief moment.
- If the domain is ambiguous or unsupported, choose unknown.

Output ONLY valid JSON in this format:
{{
  "thought": "short reasoning",
  "scene": "one_label_from_the_fixed_set"
}}"""


def build_scene_request(
    video_path: str | Path,
    *,
    fps: float | None = None,
    nframes: int | None = None,
    num_samples: int = SCENE_FRAMES,
) -> dict[str, object] | None:
    try:
        from .vlm_api import build_multimodal_message
    except ImportError:
        from vlm_api import build_multimodal_message

    video_path = Path(video_path)
    if not video_path.exists():
        return None

    if fps is None or nframes is None:
        cap = cv2.VideoCapture(str(video_path))
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if nframes is None:
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    frame_ids = sample_scene_frame_ids(nframes or 0, num_samples=num_samples)
    if not frame_ids:
        return None

    duration_sec = (nframes / fps) if fps and nframes else None

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_paths = save_scene_frames_as_tmp_jpg(video_path, frame_ids, tmp_dir)
        if not frame_paths:
            return None
        image_b64_list = [base64.b64encode(Path(p).read_bytes()).decode("ascii") for p in frame_paths]

    return {
        "custom_id": f"scene__{video_path.stem}",
        "model": MODEL,
        "messages": build_multimodal_message(image_b64_list, build_scene_prompt(len(image_b64_list), duration_sec=duration_sec)),
    }


def parse_scene_response(raw: str | None) -> str:
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return "unknown"
    try:
        result = json.loads(match.group())
    except json.JSONDecodeError:
        return "unknown"
    return normalize_scene_label(result.get("scene"))


def submit_scene_classification(
    video_path: str | Path,
    *,
    fps: float | None = None,
    nframes: int | None = None,
    num_samples: int = SCENE_FRAMES,
) -> dict[str, object] | None:
    if not get_api_key():
        log.warning("Scene classification skipped: VLM API key is not set")
        return None

    request = build_scene_request(video_path, fps=fps, nframes=nframes, num_samples=num_samples)
    if request is None:
        return None

    try:
        from .vlm_api import submit_batch_chat_requests_async
    except ImportError:
        from vlm_api import submit_batch_chat_requests_async

    with vlm_api_slot():
        submission = submit_batch_chat_requests_async([request], model=MODEL, extra_body=EXTRA_BODY)

    submission["custom_id"] = request["custom_id"]
    return submission


def classify_video_scene_direct(
    video_path: str | Path,
    *,
    fps: float | None = None,
    nframes: int | None = None,
    num_samples: int = SCENE_FRAMES,
) -> str:
    if not get_api_key():
        log.warning("Scene classification skipped: VLM API key is not set")
        return "unknown"

    request = build_scene_request(video_path, fps=fps, nframes=nframes, num_samples=num_samples)
    if request is None:
        return "unknown"

    try:
        from .vlm_api import submit_direct_chat_requests
    except ImportError:
        from vlm_api import submit_direct_chat_requests

    with vlm_api_slot():
        responses = submit_direct_chat_requests([request], model=MODEL, extra_body=EXTRA_BODY, max_workers=1)
    raw = responses.get(str(request["custom_id"]), {}).get("text")
    return parse_scene_response(raw)


def collect_scene_classification(
    submission: dict[str, object] | None,
    *,
    poll_interval_sec: int = 20,
    wait: bool = True,
) -> str:
    if not submission:
        return "unknown"

    try:
        from .vlm_api import collect_batch_chat_requests
    except ImportError:
        from vlm_api import collect_batch_chat_requests

    result = collect_batch_chat_requests(
        str(submission["batch_id"]),
        poll_interval_sec=poll_interval_sec,
        wait=wait,
    )
    if result["status"] != "completed":
        return "unknown"

    custom_id = str(submission["custom_id"])
    raw = result["results"].get(custom_id, {}).get("text")
    return parse_scene_response(raw)


def classify_video_scene(
    video_path: str | Path,
    *,
    fps: float | None = None,
    nframes: int | None = None,
    num_samples: int = SCENE_FRAMES,
) -> str:
    submission = submit_scene_classification(
        video_path,
        fps=fps,
        nframes=nframes,
        num_samples=num_samples,
    )
    if not submission:
        return "unknown"
    try:
        return collect_scene_classification(submission)
    except Exception as e:
        log.warning("Scene classification via batch failed: %s", e)
        return "unknown"
