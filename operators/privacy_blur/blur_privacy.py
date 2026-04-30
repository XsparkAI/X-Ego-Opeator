#!/usr/bin/env python3
from __future__ import annotations

"""
Privacy blur operator using EgoBlur Gen2.

Detects and blurs faces and license plates in egocentric video.

Usage:
  python blur_privacy.py --video path/to/rgb.mp4
  python blur_privacy.py --video path/to/rgb.mp4 --output blurred.mp4
  python blur_privacy.py --episode path/to/episode_dir          # reads configured input video, writes rgb_blurred.mp4
  python blur_privacy.py --video path/to/rgb.mp4 --scale 1.3    # enlarge blur region by 30%
  python blur_privacy.py --video path/to/rgb.mp4 --preview      # 480p side-by-side comparison
"""

import argparse
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
import json
import logging
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from typing import Any

try:
    from ..video_path import resolve_episode_video_path
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from video_path import resolve_episode_video_path

try:
    from ego_blur import ClassID, EgoblurDetector
    from gen2.script.constants import (
        FACE_THRESHOLDS_GEN2,
        LP_THRESHOLDS_GEN2,
        RESIZE_MIN_GEN2,
        RESIZE_MAX_GEN2,
    )
    from gen2.script.predictor import PATCH_INSTANCES_FIELDS
    from gen2.script.detectron2.export.torchscript_patch import patch_instances
    from gen2.script.detectron2.utils.utils import ResizeShortestEdge
except ImportError:
    ClassID = None
    EgoblurDetector = None
    FACE_THRESHOLDS_GEN2 = {}
    LP_THRESHOLDS_GEN2 = {}
    RESIZE_MIN_GEN2 = 0
    RESIZE_MAX_GEN2 = 0
    PATCH_INSTANCES_FIELDS = None
    patch_instances = None
    ResizeShortestEdge = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
FACE_MODEL = PROJECT_ROOT / "weights" / "ego_blur_face_gen2.jit"
LP_MODEL = PROJECT_ROOT / "weights" / "ego_blur_lp_gen2.jit"
_FRAMECACHE_THREAD_STATE = threading.local()


def _resolve_blur_flags(
    blur_targets: str | None,
    *,
    face: bool = True,
    lp: bool = True,
) -> tuple[bool, bool]:
    """Resolve a unified blur target selector with backward-compatible booleans."""
    if blur_targets is None:
        return face, lp

    normalized = str(blur_targets).strip().lower()
    if normalized == "both":
        return True, True
    if normalized == "face":
        return True, False
    if normalized in {"lp", "plate", "license_plate", "license-plate"}:
        return False, True
    raise ValueError(
        f"privacy_blur.blur_targets must be 'face', 'lp', or 'both', got {blur_targets!r}"
    )


# ── detector management ─────────────────────────────────────────────


def _get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _compute_egoblur_target(orig_h: int, orig_w: int) -> tuple[int, int]:
    """Compute the exact (H, W) that EgoBlur's internal ResizeShortestEdge would produce."""
    if ResizeShortestEdge is None:
        raise RuntimeError("EgoBlur backend is unavailable because ego_blur is not installed")
    return ResizeShortestEdge.get_output_shape(
        orig_h, orig_w, RESIZE_MIN_GEN2, RESIZE_MAX_GEN2,
    )


def _build_detectors(
    device: str,
    face: bool = True,
    lp: bool = True,
    face_thresh: float | None = None,
    lp_thresh: float | None = None,
    nms_iou: float = 0.5,
) -> tuple[EgoblurDetector | None, EgoblurDetector | None]:
    """Initialise face and/or license-plate detectors.

    Detectors are built WITHOUT resize_aug so that the caller can
    externally resize frames to the exact EgoBlur target resolution,
    bypassing the expensive numpy-based internal resize.
    """
    if EgoblurDetector is None or ClassID is None:
        raise RuntimeError("EgoBlur backend is unavailable because ego_blur is not installed")

    face_det = None
    lp_det = None

    if face:
        ft = face_thresh if face_thresh is not None else FACE_THRESHOLDS_GEN2["camera-rgb"]
        face_det = EgoblurDetector(
            model_path=str(FACE_MODEL),
            device=device,
            detection_class=ClassID.FACE,
            score_threshold=ft,
            nms_iou_threshold=nms_iou,
            resize_aug=None,
        )
        log.info(f"Face detector loaded  (thresh={ft:.4f}, device={device})")

    if lp:
        lt = lp_thresh if lp_thresh is not None else LP_THRESHOLDS_GEN2["camera-rgb"]
        lp_det = EgoblurDetector(
            model_path=str(LP_MODEL),
            device=device,
            detection_class=ClassID.LICENSE_PLATE,
            score_threshold=lt,
            nms_iou_threshold=nms_iou,
            resize_aug=None,
        )
        log.info(f"LP detector loaded    (thresh={lt:.4f}, device={device})")

    return face_det, lp_det


def _build_yolo_detector(model_path: str | Path):
    """Initialise an Ultralytics YOLO model for generic bbox-based blurring."""
    from ultralytics import YOLO

    weights = Path(model_path)
    if not weights.is_absolute():
        repo_candidate = PROJECT_ROOT.parent.parent / weights
        if repo_candidate.exists():
            weights = repo_candidate
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    model = YOLO(str(weights))
    log.info(f"YOLO blur detector loaded from {weights}")
    return model


def _build_yolo_detectors(
    *,
    face_enabled: bool,
    lp_enabled: bool,
    face_model_path: str | Path | None = None,
    lp_model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Initialise dedicated face / lp YOLO models."""
    detectors: dict[str, Any] = {"face_model": None, "lp_model": None}
    if face_enabled and face_model_path:
        detectors["face_model"] = _build_yolo_detector(face_model_path)
    if lp_enabled and lp_model_path:
        detectors["lp_model"] = _build_yolo_detector(lp_model_path)
    missing = []
    if face_enabled and detectors["face_model"] is None:
        missing.append("yolo_face_model_path")
    if lp_enabled and detectors["lp_model"] is None:
        missing.append("yolo_lp_model_path")
    if missing:
        raise ValueError(
            "YOLO privacy blur requires dedicated model paths for enabled targets: "
            + ", ".join(missing)
        )
    return detectors


def _detect_cached_frame_yolo(
    cached_path: str,
    *,
    face_model_path: str | Path | None,
    lp_model_path: str | Path | None,
    conf_thresh: float,
    input_size: int | None,
    scale_x: float,
    scale_y: float,
) -> list[list[float]]:
    """Run YOLO detection on one cached frame using thread-local model instances."""
    state = _FRAMECACHE_THREAD_STATE
    face_key = str(face_model_path) if face_model_path else None
    lp_key = str(lp_model_path) if lp_model_path else None

    if getattr(state, "face_model_key", None) != face_key:
        state.face_model = _build_yolo_detector(face_model_path) if face_model_path else None
        state.face_model_key = face_key
    if getattr(state, "lp_model_key", None) != lp_key:
        state.lp_model = _build_yolo_detector(lp_model_path) if lp_model_path else None
        state.lp_model_key = lp_key

    det_frame = cv2.imread(cached_path)
    if det_frame is None:
        raise RuntimeError(f"Failed to read cached frame: {cached_path}")

    boxes: list[list[float]] = []
    if getattr(state, "face_model", None) is not None:
        boxes.extend(
            _detect_frame_yolo(
                det_frame,
                state.face_model,
                conf_thresh=conf_thresh,
                class_ids=None,
                input_size=input_size,
            )
        )
    if getattr(state, "lp_model", None) is not None:
        boxes.extend(
            _detect_frame_yolo(
                det_frame,
                state.lp_model,
                conf_thresh=conf_thresh,
                class_ids=None,
                input_size=input_size,
            )
        )
    boxes = _dedupe_boxes(boxes)
    return [
        [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
        for b in boxes
    ]


# ── blur helpers ─────────────────────────────────────────────────────


def _apply_blur(image: np.ndarray, boxes: list[list[float]], scale: float = 1.0) -> np.ndarray:
    """Gaussian-blur elliptical regions for each detection box."""
    out = image.copy()
    h, w = out.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = box
        # scale around centre
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
        x1 = max(int(cx - bw / 2), 0)
        y1 = max(int(cy - bh / 2), 0)
        x2 = min(int(cx + bw / 2), w)
        y2 = min(int(cy + bh / 2), h)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = out[y1:y2, x1:x2]
        ksize = max(((min(roi.shape[:2]) // 2) | 1), 31)  # odd kernel >= 31
        blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)

        # elliptical mask for natural look
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        center = (roi.shape[1] // 2, roi.shape[0] // 2)
        axes = (roi.shape[1] // 2, roi.shape[0] // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        roi_out = np.where(mask[..., None] > 0, blurred_roi, roi)
        out[y1:y2, x1:x2] = roi_out

    return out


def _extract_boxes(result: list) -> list[list[float]]:
    """Safely extract boxes from EgoblurDetector.run() output.

    Returns [] when empty, or flattened list of [x1,y1,x2,y2] boxes.
    run() returns [] on no detection, or [[[x1,y1,x2,y2], ...]] otherwise.
    """
    if not result:
        return []
    return result[0]


def _detect_frame(
    bgr: np.ndarray,
    face_det: EgoblurDetector | None,
    lp_det: EgoblurDetector | None,
    device: str,
) -> list[list[float]]:
    """Run detectors on a single BGR frame, return merged box list."""
    tensor = torch.from_numpy(np.transpose(bgr, (2, 0, 1))).to(device)
    boxes: list[list[float]] = []
    if face_det:
        boxes.extend(_extract_boxes(face_det.run(tensor)))
    if lp_det:
        boxes.extend(_extract_boxes(lp_det.run(tensor)))
    return boxes


def _detect_frame_yolo(
    bgr: np.ndarray,
    model: Any,
    conf_thresh: float,
    class_ids: list[int] | None = None,
    input_size: int | None = None,
) -> list[list[float]]:
    """Run YOLO bbox detection on a single BGR frame."""
    results = model.predict(
        source=bgr,
        conf=conf_thresh,
        classes=class_ids,
        imgsz=input_size,
        verbose=False,
    )
    if not results:
        return []

    boxes_result = results[0].boxes
    if boxes_result is None or len(boxes_result) == 0:
        return []

    xyxy = boxes_result.xyxy.cpu().numpy()
    return [[float(v) for v in row.tolist()] for row in xyxy]


def _dedupe_boxes(boxes: list[list[float]], iou_thresh: float = 0.9) -> list[list[float]]:
    """Drop near-identical boxes when merging outputs from multiple detectors."""
    if not boxes:
        return []

    def _iou(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    kept: list[list[float]] = []
    for box in boxes:
        if all(_iou(box, existing) < iou_thresh for existing in kept):
            kept.append(box)
    return kept


def _detect_backend_boxes(
    det_frame: np.ndarray,
    *,
    backend: str,
    device: str,
    face_det: EgoblurDetector | None,
    lp_det: EgoblurDetector | None,
    yolo_detectors: dict[str, Any] | None,
    face_enabled: bool,
    lp_enabled: bool,
    yolo_conf_thresh: float,
    yolo_input_size: int | None,
) -> list[list[float]]:
    """Run the configured backend on one detection frame and return merged boxes."""
    if backend == "egoblur":
        return _detect_frame(det_frame, face_det, lp_det, device)

    if backend == "yolo":
        boxes = []
        if face_enabled and yolo_detectors and yolo_detectors["face_model"] is not None:
            boxes.extend(
                _detect_frame_yolo(
                    det_frame,
                    yolo_detectors["face_model"],
                    conf_thresh=yolo_conf_thresh,
                    class_ids=None,
                    input_size=yolo_input_size,
                )
            )
        if lp_enabled and yolo_detectors and yolo_detectors["lp_model"] is not None:
            boxes.extend(
                _detect_frame_yolo(
                    det_frame,
                    yolo_detectors["lp_model"],
                    conf_thresh=yolo_conf_thresh,
                    class_ids=None,
                    input_size=yolo_input_size,
                )
            )
        return _dedupe_boxes(boxes)


def _read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    """Seek to a frame index and read it back."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def _merge_positive_range(
    ranges: list[tuple[int, int]],
    start: int,
    end: int,
) -> None:
    """Merge a newly confirmed positive frame range into sorted intervals."""
    if start > end:
        return

    merged: list[tuple[int, int]] = []
    placed = False
    cur_start, cur_end = start, end
    for existing_start, existing_end in ranges:
        if existing_end + 1 < cur_start:
            merged.append((existing_start, existing_end))
            continue
        if cur_end + 1 < existing_start:
            if not placed:
                merged.append((cur_start, cur_end))
                placed = True
            merged.append((existing_start, existing_end))
            continue

        cur_start = min(cur_start, existing_start)
        cur_end = max(cur_end, existing_end)

    if not placed:
        merged.append((cur_start, cur_end))

    ranges[:] = merged


def _find_positive_range(
    ranges: list[tuple[int, int]],
    frame_idx: int,
) -> tuple[int, int] | None:
    """Return the confirmed positive interval containing frame_idx, if any."""
    for start, end in ranges:
        if start <= frame_idx <= end:
            return (start, end)
    return None


# ── video probe ─────────────────────────────────────────────────────


# Source codec → preferred encoder.
# GPU (NVENC) is tried first; if unavailable, falls back to CPU encoder.
_ENCODER_MAP_GPU = {
    "h264": "h264_nvenc",
    "hevc": "hevc_nvenc",
    "h265": "hevc_nvenc",
}
_ENCODER_MAP_CPU = {
    "h264": "libx264",
    "hevc": "libx265",
    "h265": "libx265",
    "vp9": "libvpx-vp9",
    "av1": "libsvtav1",
}

def _nvenc_available() -> bool:
    """Quick check: probe a 1-frame null encode with hevc_nvenc.

    NVENC requires minimum ~145x97 resolution; use 256x256 for the probe.
    """
    import subprocess as _sp
    try:
        r = _sp.run(
            ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
             "-frames:v", "1", "-c:v", "hevc_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False

_USE_NVENC: bool | None = None  # lazy init


def _probe_video(path: str) -> dict:
    """Probe original video encoding parameters via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    br = int(video.get("bit_rate", 0))
    if br == 0:
        br = int(info.get("format", {}).get("bit_rate", 0))

    return {
        "codec": video["codec_name"],
        "pix_fmt": video.get("pix_fmt", "yuv420p"),
        "bitrate": br,
        "profile": video.get("profile", ""),
    }


def _build_ffmpeg_cmd(
    output_path: str,
    frame_w: int,
    frame_h: int,
    fps: float,
    src: dict,
) -> list[str]:
    """Build ffmpeg command that matches original video encoding config.

    Prefers NVENC GPU encoding when available; falls back to CPU encoders.
    """
    global _USE_NVENC
    if _USE_NVENC is None:
        _USE_NVENC = _nvenc_available()
        import logging as _log
        _log.getLogger(__name__).info(
            f"NVENC GPU encoding: {'enabled' if _USE_NVENC else 'unavailable, using CPU'}"
        )

    codec = src["codec"]
    if _USE_NVENC and codec in _ENCODER_MAP_GPU:
        encoder = _ENCODER_MAP_GPU[codec]
        use_gpu = True
    else:
        encoder = _ENCODER_MAP_CPU.get(codec, "libx264")
        use_gpu = False

    pix_fmt = src["pix_fmt"] or "yuv420p"
    bitrate = src["bitrate"]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{frame_w}x{frame_h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", encoder,
        "-pix_fmt", pix_fmt,
    ]

    if use_gpu:
        # NVENC: use CBR with target bitrate, or high-quality VBR
        if bitrate > 0:
            cmd += ["-b:v", str(bitrate), "-maxrate:v", str(int(bitrate * 1.5)),
                    "-bufsize:v", str(bitrate * 2)]
        else:
            cmd += ["-cq", "18", "-preset", "p4"]  # NVENC quality preset
    else:
        # CPU encoders
        if bitrate > 0:
            cmd += ["-b:v", str(bitrate)]
            if encoder == "libx264":
                cmd += ["-preset", "veryfast", "-profile:v", "high"]
            elif encoder == "libx265":
                cmd += ["-preset", "veryfast", "-profile:v", "main"]
        else:
            if encoder == "libx264":
                cmd += ["-crf", "17", "-preset", "veryfast"]
            elif encoder == "libx265":
                cmd += ["-crf", "18", "-preset", "veryfast"]

    cmd.append(output_path)
    return cmd


# ── main video processing ───────────────────────────────────────────


def blur_video(
    video_path: Path,
    output_path: Path,
    detector_backend: str = "egoblur",
    blur_targets: str = "both",
    detection_mode: str = "sampling_expand",
    frame_sampling_step: int = 1,
    use_frame_cache: bool = True,
    frame_cache_num_workers: int = 1,
    face: bool = True,
    lp: bool = True,
    scale: float = 1.0,
    face_thresh: float | None = None,
    lp_thresh: float | None = None,
    yolo_face_model_path: str | Path | None = None,
    yolo_lp_model_path: str | Path | None = None,
    yolo_conf_thresh: float = 0.25,
    yolo_input_size: int | None = 960,
    resize: int | None = None,
    detectors: Any = None,
) -> dict:
    """
    Process a video: detect faces/LPs and write blurred output.

    Frames are externally resized to EgoBlur's exact internal target
    resolution (computed via ResizeShortestEdge), and detectors run with
    resize_aug=None to skip the expensive numpy-based internal resize.
    Blur is applied on the original resolution frame.

    Args:
        detectors: Optional pre-loaded detector object(s) to avoid reloading
                   models on every call.

    Returns a summary dict with detection stats.
    """
    backend = str(detector_backend).lower()
    detection_mode = str(detection_mode).strip().lower()
    if detection_mode not in {"sampling_expand", "legacy_per_frame"}:
        raise ValueError(
            "privacy_blur.detection_mode must be 'sampling_expand' or 'legacy_per_frame', "
            f"got {detection_mode!r}"
        )
    face_enabled, lp_enabled = _resolve_blur_flags(
        blur_targets,
        face=face,
        lp=lp,
    )
    device = _get_device()
    face_det = lp_det = None
    yolo_detectors = None
    total_started = time.time()
    cache_prepare_sec = 0.0
    ffmpeg_prepare_sec = 0.0
    detect_scan_sec = 0.0
    write_pass_sec = 0.0
    writer_close_wait_sec = 0.0
    detect_imread_sec = 0.0
    detect_infer_sec = 0.0
    detect_post_sec = 0.0
    main_video_read_sec = 0.0
    blur_apply_sec = 0.0
    writer_write_sec = 0.0
    detect_cache_hits = 0

    if backend == "egoblur":
        if detectors is not None:
            face_det, lp_det = detectors
        else:
            face_det, lp_det = _build_detectors(
                device,
                face=face_enabled,
                lp=lp_enabled,
                face_thresh=face_thresh,
                lp_thresh=lp_thresh,
            )
    elif backend == "yolo":
        if detectors is not None:
            yolo_detectors = detectors
        else:
            yolo_detectors = _build_yolo_detectors(
                face_enabled=face_enabled,
                lp_enabled=lp_enabled,
                face_model_path=yolo_face_model_path,
                lp_model_path=yolo_lp_model_path,
            )
    else:
        raise ValueError(
            f"Unsupported detector_backend={detector_backend!r}, expected 'egoblur' or 'yolo'"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sampling_step = max(1, int(frame_sampling_step))

    cache_profile = None
    cached_paths: list[str] | None = None
    frame_cache_enabled = False
    frame_cache_executor: ThreadPoolExecutor | None = None
    pending_detects: dict[int, Future] = {}
    frame_cache_worker_count = max(1, int(frame_cache_num_workers))

    if backend == "egoblur":
        # Compute EgoBlur's exact target resolution and resize externally.
        det_h, det_w = _compute_egoblur_target(frame_h, frame_w)
        if det_h == frame_h and det_w == frame_w:
            needs_resize = False
            scale_back_h = scale_back_w = 1.0
        else:
            needs_resize = True
            scale_back_h = frame_h / det_h
            scale_back_w = frame_w / det_w
    elif backend == "yolo":
        det_h, det_w = frame_h, frame_w
        needs_resize = False
        scale_back_h = scale_back_w = 1.0
    else:
        det_h, det_w = frame_h, frame_w
        needs_resize = False
        scale_back_h = scale_back_w = 1.0

    if backend == "yolo" and use_frame_cache:
        cache_prepare_started = time.time()
        try:
            from ..frame_cache.cache_utils import (
                PROFILE_VLM,
                TARGET_H,
                TARGET_W,
                ensure_cached_frame_paths,
            )
        except ImportError:
            from frame_cache.cache_utils import (  # type: ignore
                PROFILE_VLM,
                TARGET_H,
                TARGET_W,
                ensure_cached_frame_paths,
            )

        try:
            cached_paths = ensure_cached_frame_paths(video_path.parent, list(range(total_frames)), profile=PROFILE_VLM)
        except Exception as e:
            log.warning(f"frame_cache unavailable for privacy_blur ({e}), falling back to video decoder")
            cached_paths = None
        if cached_paths is not None and len(cached_paths) == total_frames:
            frame_cache_enabled = True
            cache_profile = PROFILE_VLM
            det_w, det_h = TARGET_W, TARGET_H
            needs_resize = False
            scale_back_w = frame_w / det_w
            scale_back_h = frame_h / det_h
            if device == "cpu" and frame_cache_worker_count > 1:
                frame_cache_executor = ThreadPoolExecutor(max_workers=frame_cache_worker_count)
        cache_prepare_sec = time.time() - cache_prepare_started

    ffmpeg_prepare_started = time.time()
    src = _probe_video(str(video_path))
    ffmpeg_cmd = _build_ffmpeg_cmd(str(output_path), frame_w, frame_h, fps, src)
    log.info(f"Source encoding: {src['codec']}, pix_fmt={src['pix_fmt']}, bitrate={src['bitrate'] // 1000}kbps")
    stderr_tmp = tempfile.TemporaryFile()
    writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=stderr_tmp)
    ffmpeg_prepare_sec = time.time() - ffmpeg_prepare_started
    detect_desc = (
        "exact EgoBlur target, internal resize bypassed"
        if backend == "egoblur"
        else (
            f"frame_cache {det_w}x{det_h}, workers={frame_cache_worker_count}, YOLO imgsz={yolo_input_size or 'native'}"
            if frame_cache_enabled
            else f"YOLO imgsz={yolo_input_size or 'native'}"
        )
    )

    log.info(
        f"Processing {video_path.name}: {total_frames} frames, "
        f"{frame_w}x{frame_h} @ {fps:.1f} fps, "
        f"detect at {det_w}x{det_h} ({detect_desc})"
    )

    frame_idx = 0
    total_detections = 0
    frames_with_detections = 0
    t_start = time.time()

    detection_cache: dict[int, list[list[float]]] = {}
    positive_ranges: list[tuple[int, int]] = []
    unique_detection_frames = 0

    def submit_detect(frame_index: int) -> None:
        if frame_cache_executor is None or cached_paths is None:
            return
        if frame_index < 0 or frame_index >= total_frames:
            return
        if frame_index in detection_cache or frame_index in pending_detects:
            return
        pending_detects[frame_index] = frame_cache_executor.submit(
            _detect_cached_frame_yolo,
            cached_paths[frame_index],
            face_model_path=yolo_face_model_path if face_enabled else None,
            lp_model_path=yolo_lp_model_path if lp_enabled else None,
            conf_thresh=yolo_conf_thresh,
            input_size=yolo_input_size,
            scale_x=scale_back_w,
            scale_y=scale_back_h,
        )

    def prefetch_detects(frame_indices: list[int]) -> None:
        for candidate in frame_indices:
            submit_detect(candidate)

    def detect_boxes_for_frame_idx(frame_index: int, frame_bgr: np.ndarray | None = None) -> list[list[float]]:
        nonlocal unique_detection_frames
        nonlocal detect_imread_sec
        nonlocal detect_infer_sec
        nonlocal detect_post_sec
        nonlocal detect_cache_hits
        cached = detection_cache.get(frame_index)
        if cached is not None:
            detect_cache_hits += 1
            return cached

        if frame_cache_enabled and cached_paths is not None:
            future = pending_detects.pop(frame_index, None)
            if future is None:
                if frame_cache_executor is not None:
                    submit_detect(frame_index)
                    future = pending_detects.pop(frame_index)
                else:
                    imread_started = time.time()
                    det_frame = cv2.imread(cached_paths[frame_index])
                    detect_imread_sec += time.time() - imread_started
                    if det_frame is None:
                        raise RuntimeError(f"Failed to read cached frame: {cached_paths[frame_index]}")
                    infer_started = time.time()
                    boxes = _detect_backend_boxes(
                        det_frame,
                        backend=backend,
                        device=device,
                        face_det=face_det,
                        lp_det=lp_det,
                        yolo_detectors=yolo_detectors,
                        face_enabled=face_enabled,
                        lp_enabled=lp_enabled,
                        yolo_conf_thresh=yolo_conf_thresh,
                        yolo_input_size=yolo_input_size,
                    )
                    detect_infer_sec += time.time() - infer_started
                    post_started = time.time()
                    boxes = [
                        [b[0] * scale_back_w, b[1] * scale_back_h, b[2] * scale_back_w, b[3] * scale_back_h]
                        for b in boxes
                    ]
                    detect_post_sec += time.time() - post_started
                    unique_detection_frames += 1
                    detection_cache[frame_index] = boxes
                    return boxes
            infer_started = time.time()
            boxes = future.result()
            detect_infer_sec += time.time() - infer_started
            unique_detection_frames += 1
            detection_cache[frame_index] = boxes
            return boxes

        local_frame = frame_bgr
        if local_frame is None:
            local_frame = _read_frame_at(cap_seek, frame_index)
        if local_frame is None:
            detection_cache[frame_index] = []
            return detection_cache[frame_index]

        if needs_resize:
            post_started = time.time()
            det_frame = cv2.resize(local_frame, (det_w, det_h))
            detect_post_sec += time.time() - post_started
        else:
            det_frame = local_frame

        infer_started = time.time()
        boxes = _detect_backend_boxes(
            det_frame,
            backend=backend,
            device=device,
                    face_det=face_det,
                    lp_det=lp_det,
                    yolo_detectors=yolo_detectors,
                    face_enabled=face_enabled,
                    lp_enabled=lp_enabled,
                    yolo_conf_thresh=yolo_conf_thresh,
            yolo_input_size=yolo_input_size,
        )
        detect_infer_sec += time.time() - infer_started
        unique_detection_frames += 1
        if boxes and needs_resize:
            post_started = time.time()
            boxes = [
                [b[0] * scale_back_w, b[1] * scale_back_h, b[2] * scale_back_w, b[3] * scale_back_h]
                for b in boxes
            ]
            detect_post_sec += time.time() - post_started
        detection_cache[frame_index] = boxes
        return boxes

    sample_hits = 0
    unique_expanded_frames = 0
    cap_detect = None
    cap_seek = None
    if detection_mode == "sampling_expand" and not frame_cache_enabled:
        cap_detect = cv2.VideoCapture(str(video_path))
        if not cap_detect.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video for detection scan: {video_path}")
        cap_seek = cv2.VideoCapture(str(video_path))
        if not cap_seek.isOpened():
            cap.release()
            cap_detect.release()
            raise RuntimeError(f"Cannot open video for random-access scan: {video_path}")

    patch_context = (
        patch_instances(fields=PATCH_INSTANCES_FIELDS)
        if patch_instances is not None and PATCH_INSTANCES_FIELDS is not None
        else nullcontext()
    )
    with torch.no_grad(), patch_context:
        if detection_mode == "legacy_per_frame":
            prefetch_width = max(4, frame_cache_worker_count * 2)
            write_pass_started = time.time()
            while True:
                if frame_cache_enabled and frame_cache_executor is not None:
                    prefetch_detects(list(range(frame_idx, min(total_frames, frame_idx + prefetch_width))))
                read_started = time.time()
                ret, bgr = cap.read()
                main_video_read_sec += time.time() - read_started
                if not ret:
                    break

                if frame_cache_enabled:
                    boxes = detect_boxes_for_frame_idx(frame_idx)
                elif needs_resize:
                    det_frame = cv2.resize(bgr, (det_w, det_h))
                    boxes = _detect_backend_boxes(
                        det_frame,
                        backend=backend,
                        device=device,
                        face_det=face_det,
                        lp_det=lp_det,
                        yolo_detectors=yolo_detectors,
                        face_enabled=face_enabled,
                        lp_enabled=lp_enabled,
                        yolo_conf_thresh=yolo_conf_thresh,
                        yolo_input_size=yolo_input_size,
                    )
                    unique_detection_frames += 1
                    if boxes and needs_resize:
                        boxes = [
                            [b[0] * scale_back_w, b[1] * scale_back_h, b[2] * scale_back_w, b[3] * scale_back_h]
                            for b in boxes
                        ]
                else:
                    det_frame = bgr
                    boxes = _detect_backend_boxes(
                        det_frame,
                        backend=backend,
                        device=device,
                        face_det=face_det,
                        lp_det=lp_det,
                        yolo_detectors=yolo_detectors,
                        face_enabled=face_enabled,
                        lp_enabled=lp_enabled,
                        yolo_conf_thresh=yolo_conf_thresh,
                        yolo_input_size=yolo_input_size,
                    )
                    unique_detection_frames += 1

                if boxes:
                    blur_started = time.time()
                    bgr = _apply_blur(bgr, boxes, scale)
                    blur_apply_sec += time.time() - blur_started
                    total_detections += len(boxes)
                    frames_with_detections += 1
                    detection_cache[frame_idx] = boxes

                write_started = time.time()
                writer.stdin.write(bgr.tobytes())
                writer_write_sec += time.time() - write_started
                frame_idx += 1

                if frame_idx % 500 == 0:
                    elapsed = time.time() - t_start
                    log.info(
                        f"  legacy frame {frame_idx}/{total_frames} "
                        f"({frame_idx / total_frames * 100:.0f}%), "
                        f"{frame_idx / elapsed:.1f} fps"
                    )
            write_pass_sec = time.time() - write_pass_started
        else:
            prefetch_width = max(4, frame_cache_worker_count * 2)
            detect_scan_started = time.time()
            if frame_cache_enabled:
                sample_indices = list(range(0, total_frames, sampling_step))
                for sample_pos, detect_scan_idx in enumerate(sample_indices):
                    if frame_cache_executor is not None:
                        prefetch_detects(sample_indices[sample_pos: sample_pos + prefetch_width])
                    sample_boxes = detect_boxes_for_frame_idx(detect_scan_idx)
                    if sample_boxes:
                        sample_hits += 1
                        if _find_positive_range(positive_ranges, detect_scan_idx) is None:
                            range_start = detect_scan_idx
                            range_end = detect_scan_idx

                            back_idx = detect_scan_idx - 1
                            while back_idx >= 0:
                                if frame_cache_executor is not None:
                                    prefetch_detects(list(range(max(0, back_idx - prefetch_width + 1), back_idx + 1)))
                                was_unknown = back_idx not in detection_cache
                                back_boxes = detect_boxes_for_frame_idx(back_idx)
                                if not back_boxes:
                                    break
                                if was_unknown:
                                    unique_expanded_frames += 1
                                range_start = back_idx
                                back_idx -= 1

                            forward_idx = detect_scan_idx + 1
                            while forward_idx < total_frames:
                                if frame_cache_executor is not None:
                                    prefetch_detects(list(range(forward_idx, min(total_frames, forward_idx + prefetch_width))))
                                was_unknown = forward_idx not in detection_cache
                                forward_boxes = detect_boxes_for_frame_idx(forward_idx)
                                if not forward_boxes:
                                    break
                                if was_unknown:
                                    unique_expanded_frames += 1
                                range_end = forward_idx
                                forward_idx += 1

                            _merge_positive_range(positive_ranges, range_start, range_end)
                    if (sample_pos + 1) % 500 == 0:
                        elapsed = time.time() - t_start
                        log.info(
                            f"  detect scan {sample_pos + 1}/{len(sample_indices)} sampled "
                            f"({(sample_pos + 1) / max(1, len(sample_indices)) * 100:.0f}%), "
                            f"{unique_detection_frames / elapsed:.1f} detect fps"
                        )
                detect_scan_sec = time.time() - detect_scan_started
            else:
                detect_scan_idx = 0
                while True:
                    ret, sample_bgr = cap_detect.read()
                    if not ret:
                        break

                    if detect_scan_idx % sampling_step == 0:
                        sample_boxes = detect_boxes_for_frame_idx(detect_scan_idx, sample_bgr)
                        if sample_boxes:
                            sample_hits += 1
                            if _find_positive_range(positive_ranges, detect_scan_idx) is None:
                                range_start = detect_scan_idx
                                range_end = detect_scan_idx

                                back_idx = detect_scan_idx - 1
                                while back_idx >= 0:
                                    was_unknown = back_idx not in detection_cache
                                    back_boxes = detect_boxes_for_frame_idx(back_idx)
                                    if not back_boxes:
                                        break
                                    if was_unknown:
                                        unique_expanded_frames += 1
                                    range_start = back_idx
                                    back_idx -= 1

                                forward_idx = detect_scan_idx + 1
                                while forward_idx < total_frames:
                                    was_unknown = forward_idx not in detection_cache
                                    forward_boxes = detect_boxes_for_frame_idx(forward_idx)
                                    if not forward_boxes:
                                        break
                                    if was_unknown:
                                        unique_expanded_frames += 1
                                    range_end = forward_idx
                                    forward_idx += 1

                                _merge_positive_range(positive_ranges, range_start, range_end)

                    detect_scan_idx += 1
                    if detect_scan_idx % 500 == 0:
                        elapsed = time.time() - t_start
                        log.info(
                            f"  detect scan {detect_scan_idx}/{total_frames} "
                            f"({detect_scan_idx / total_frames * 100:.0f}%), "
                            f"{detect_scan_idx / elapsed:.1f} fps"
                        )
                detect_scan_sec = time.time() - detect_scan_started

                cap_detect.release()
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            write_pass_started = time.time()
            while True:
                read_started = time.time()
                ret, bgr = cap.read()
                main_video_read_sec += time.time() - read_started
                if not ret:
                    break

                boxes = detection_cache.get(frame_idx, [])
                if boxes:
                    blur_started = time.time()
                    bgr = _apply_blur(bgr, boxes, scale)
                    blur_apply_sec += time.time() - blur_started
                    total_detections += len(boxes)
                    frames_with_detections += 1

                write_started = time.time()
                writer.stdin.write(bgr.tobytes())
                writer_write_sec += time.time() - write_started
                frame_idx += 1

                if frame_idx % 500 == 0:
                    elapsed = time.time() - t_start
                    log.info(
                        f"  write frame {frame_idx}/{total_frames} "
                        f"({frame_idx / total_frames * 100:.0f}%), "
                        f"{frame_idx / elapsed:.1f} fps"
                    )
            write_pass_sec = time.time() - write_pass_started

    cap.release()
    if cap_detect is not None and cap_detect.isOpened():
        cap_detect.release()
    if cap_seek is not None and cap_seek.isOpened():
        cap_seek.release()
    pending_detects.clear()
    if frame_cache_executor is not None:
        frame_cache_executor.shutdown(wait=True, cancel_futures=True)
    writer.stdin.close()
    writer_wait_started = time.time()
    writer.wait()
    writer_close_wait_sec = time.time() - writer_wait_started
    stderr_tmp.seek(0)
    ffmpeg_stderr = stderr_tmp.read()
    stderr_tmp.close()
    if writer.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{ffmpeg_stderr.decode()[-2000:]}")
    elapsed = time.time() - t_start

    summary = {
        "input": str(video_path),
        "output": str(output_path),
        "total_frames": frame_idx,
        "output_resolution": f"{frame_w}x{frame_h}",
        "detect_resolution": f"{det_w}x{det_h}",
        "fps": round(fps, 2),
        "elapsed_sec": round(elapsed, 2),
        "throughput_fps": round(frame_idx / elapsed, 2) if elapsed > 0 else 0,
        "detections_total": total_detections,
        "frames_with_detections": frames_with_detections,
        "detector_backend": backend,
        "detection_mode": detection_mode,
        "blur_targets": blur_targets,
        "frame_sampling_step": sampling_step,
        "sampled_frames": (total_frames + sampling_step - 1) // sampling_step if total_frames > 0 else 0,
        "sample_hit_frames": sample_hits,
        "unique_detection_frames": unique_detection_frames,
        "unique_expanded_frames": unique_expanded_frames,
        "positive_ranges": positive_ranges,
        "blur_face": face_enabled,
        "blur_lp": lp_enabled,
        "scale_factor": scale,
        "decoder": "frame_cache" if frame_cache_enabled else "video_decoder",
        "cache_profile": cache_profile,
        "frame_cache_num_workers": frame_cache_worker_count if frame_cache_enabled else 0,
        "timings": {
            "total_wall_sec": round(time.time() - total_started, 2),
            "cache_prepare_sec": round(cache_prepare_sec, 2),
            "ffmpeg_prepare_sec": round(ffmpeg_prepare_sec, 2),
            "detect_scan_sec": round(detect_scan_sec, 2),
            "write_pass_sec": round(write_pass_sec, 2),
            "writer_close_wait_sec": round(writer_close_wait_sec, 2),
            "detect_imread_sec": round(detect_imread_sec, 2),
            "detect_infer_sec": round(detect_infer_sec, 2),
            "detect_post_sec": round(detect_post_sec, 2),
            "main_video_read_sec": round(main_video_read_sec, 2),
            "blur_apply_sec": round(blur_apply_sec, 2),
            "writer_write_sec": round(writer_write_sec, 2),
        },
        "counters": {
            "detect_cache_hits": detect_cache_hits,
        },
    }
    if backend == "yolo":
        summary["yolo_face_model_path"] = str(yolo_face_model_path) if yolo_face_model_path else None
        summary["yolo_lp_model_path"] = str(yolo_lp_model_path) if yolo_lp_model_path else None
        summary["yolo_conf_thresh"] = yolo_conf_thresh
        summary["yolo_input_size"] = yolo_input_size
    log.info(
        f"Done: {frame_idx} frames in {elapsed:.1f}s "
        f"({summary['throughput_fps']:.1f} fps), saved to {output_path}"
    )
    return summary


def generate_preview(
    original: Path,
    blurred: Path,
    output: Path,
    preview_height: int = 480,
):
    """Create a side-by-side comparison video (original | blurred)."""
    cap_orig = cv2.VideoCapture(str(original))
    cap_blur = cv2.VideoCapture(str(blurred))
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = preview_height / orig_h
    pw = int(orig_w * scale)
    pw = pw if pw % 2 == 0 else pw + 1  # even width for codec

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (pw * 2, preview_height))

    while True:
        ret1, f1 = cap_orig.read()
        ret2, f2 = cap_blur.read()
        if not ret1 or not ret2:
            break
        f1 = cv2.resize(f1, (pw, preview_height))
        f2 = cv2.resize(f2, (pw, preview_height))

        # labels
        cv2.putText(f1, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(f2, "Blurred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        combined = np.hstack([f1, f2])
        writer.write(combined)

    cap_orig.release()
    cap_blur.release()
    writer.release()
    log.info(f"Preview saved to {output}")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Privacy blur operator (EgoBlur Gen2)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Path to input video")
    group.add_argument("--episode", type=Path, help="Episode directory (reads configured input video)")
    parser.add_argument("--output", type=Path, default=None, help="Output video path (--video mode)")
    parser.add_argument("--scale", type=float, default=1.0, help="Blur region scale factor (default: 1.0)")
    parser.add_argument("--detector-backend", type=str, default="egoblur", help="Detector backend: egoblur | yolo")
    parser.add_argument("--blur-targets", type=str, default="both", help="Blur targets: face | lp | both")
    parser.add_argument("--face-thresh", type=float, default=None, help="Face detection threshold")
    parser.add_argument("--lp-thresh", type=float, default=None, help="LP detection threshold")
    parser.add_argument("--yolo-face-model-path", type=str, default=None, help="Dedicated YOLO face weights path")
    parser.add_argument("--yolo-lp-model-path", type=str, default=None, help="Dedicated YOLO license-plate weights path")
    parser.add_argument("--yolo-conf-thresh", type=float, default=0.25, help="YOLO detection threshold")
    parser.add_argument("--yolo-input-size", type=int, default=960, help="YOLO imgsz")
    parser.add_argument("--preview", action="store_true", help="Generate side-by-side preview")
    args = parser.parse_args()

    if args.episode:
        video_path = resolve_episode_video_path(args.episode)
        output_path = args.episode / "rgb_blurred.mp4"
    else:
        video_path = args.video
        output_path = args.output or video_path.with_stem(video_path.stem + "_blurred")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    summary = blur_video(
        video_path,
        output_path,
        detector_backend=args.detector_backend,
        blur_targets=args.blur_targets,
        scale=args.scale,
        face_thresh=args.face_thresh, lp_thresh=args.lp_thresh,
        yolo_face_model_path=args.yolo_face_model_path,
        yolo_lp_model_path=args.yolo_lp_model_path,
        yolo_conf_thresh=args.yolo_conf_thresh,
        yolo_input_size=args.yolo_input_size,
    )

    if args.preview:
        preview_path = output_path.with_stem(output_path.stem + "_preview")
        generate_preview(video_path, output_path, preview_path)

    log.info(f"Summary: {summary}")


if __name__ == "__main__":
    main()
