from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..video_path import resolve_episode_video_path

from ..video_utils import apply_rotation, get_manual_rotation

log = logging.getLogger(__name__)

CACHE_DIRNAME = ".frame_cache"
PROFILE_VLM = "vlm_640x480_q85"
PROFILE_QUALITY = "quality_gray_w1920_png"
TARGET_W = 640
TARGET_H = 480
JPEG_QUALITY = 85
QUALITY_MAX_WIDTH = 1920


@dataclass
class CaptionSamplingSpec:
    window_sec: float
    step_sec: float
    frames_per_window: int


def probe_video(video_path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


def build_caption_frame_ids(
    fps: float,
    nframes: int,
    spec: CaptionSamplingSpec,
) -> list[int]:
    window_frames = int(spec.window_sec * fps)
    step_frames = int(spec.step_sec * fps)
    frames_per_window = max(1, spec.frames_per_window)

    if nframes <= 0:
        return []

    if nframes <= window_frames:
        return np.linspace(0, nframes - 1, frames_per_window, dtype=int).tolist()

    frame_ids: list[int] = []
    start = 0
    while start < nframes:
        end = min(start + window_frames - 1, nframes - 1)
        actual_len = end - start + 1
        n_samples = min(frames_per_window, actual_len)
        frame_ids.extend(np.linspace(start, end, n_samples, dtype=int).tolist())
        start += step_frames
        if end == nframes - 1:
            break
    return frame_ids


def build_stride_frame_ids(nframes: int, frame_step: int) -> list[int]:
    if nframes <= 0:
        return []
    step = max(1, frame_step)
    frame_ids = list(range(0, nframes, step))
    if not frame_ids or frame_ids[-1] != nframes - 1:
        frame_ids.append(nframes - 1)
    return frame_ids


def cache_root(episode_dir: Path) -> Path:
    return episode_dir / CACHE_DIRNAME


def profile_dir(episode_dir: Path, profile: str = PROFILE_VLM) -> Path:
    return cache_root(episode_dir) / profile


def manifest_path(episode_dir: Path) -> Path:
    return cache_root(episode_dir) / "manifest.json"


def load_manifest(episode_dir: Path) -> dict | None:
    path = manifest_path(episode_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_cached_frame_paths(
    episode_dir: Path,
    frame_ids: list[int],
    profile: str = PROFILE_VLM,
) -> list[str] | None:
    manifest = load_manifest(episode_dir)
    if not manifest or profile not in manifest.get("profiles", {}):
        return None

    paths: list[str] = []
    prof_dir = profile_dir(episode_dir, profile)
    for fid in sorted(set(frame_ids)):
        path = prof_dir / f"frame_{fid:06d}.jpg"
        if not path.exists():
            return None
        paths.append(str(path))
    return paths


def build_sample_fps_frame_ids(
    fps: float,
    nframes: int,
    sample_fps: float | None,
) -> list[int]:
    if nframes <= 0:
        return []
    if sample_fps and sample_fps < fps:
        step = max(1, int(round(fps / sample_fps)))
    else:
        step = 1
    return list(range(0, nframes, step))


def _update_manifest(
    episode_dir: Path,
    fps: float,
    total_frames: int,
    profile: str,
    profile_info: dict,
) -> dict:
    manifest = load_manifest(episode_dir) or {
        "video": str(resolve_episode_video_path(episode_dir)),
        "fps": round(fps, 4),
        "total_frames": total_frames,
        "profiles": {},
    }
    manifest["video"] = str(resolve_episode_video_path(episode_dir))
    manifest["fps"] = round(fps, 4)
    manifest["total_frames"] = total_frames
    manifest.setdefault("profiles", {})
    manifest["profiles"][profile] = profile_info
    manifest_path(episode_dir).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _build_transformed_cache(
    episode_dir: Path,
    frame_ids: list[int],
    profile: str,
    transform: Callable[[np.ndarray], np.ndarray],
    suffix: str,
    imwrite_params: list[int] | None = None,
    apply_manual_rotation: bool = False,
    profile_info_extra: dict | None = None,
) -> dict:
    video_path = resolve_episode_video_path(episode_dir)
    fps, total_frames = probe_video(video_path)
    wanted = sorted({fid for fid in frame_ids if 0 <= fid < total_frames})

    root = cache_root(episode_dir)
    prof_dir = profile_dir(episode_dir, profile)
    root.mkdir(exist_ok=True)
    prof_dir.mkdir(exist_ok=True)

    rotation = get_manual_rotation(str(video_path)) if apply_manual_rotation else 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    saved = 0
    wanted_idx = 0
    current_frame = 0
    cached_shape: list[int] | None = None

    while wanted_idx < len(wanted):
        ret, frame = cap.read()
        if not ret:
            break
        target = wanted[wanted_idx]
        if current_frame == target:
            if apply_manual_rotation:
                frame = apply_rotation(frame, rotation)
            frame = transform(frame)
            if cached_shape is None:
                cached_shape = [int(frame.shape[1]), int(frame.shape[0])]
            out_path = prof_dir / f"frame_{target:06d}{suffix}"
            ok = cv2.imwrite(str(out_path), frame, imwrite_params or [])
            if ok:
                saved += 1
            wanted_idx += 1
        current_frame += 1

    cap.release()

    profile_info = {
        "frame_count": saved,
        "frame_ids": wanted,
    }
    if cached_shape is not None:
        profile_info["cached_size"] = cached_shape
    if profile_info_extra:
        profile_info.update(profile_info_extra)

    return _update_manifest(
        episode_dir,
        fps=fps,
        total_frames=total_frames,
        profile=profile,
        profile_info=profile_info,
    )


def build_cache(
    episode_dir: Path,
    frame_ids: list[int],
    profile: str = PROFILE_VLM,
) -> dict:
    return _build_transformed_cache(
        episode_dir,
        frame_ids=frame_ids,
        profile=profile,
        transform=lambda frame: cv2.resize(frame, (TARGET_W, TARGET_H)),
        suffix=".jpg",
        imwrite_params=[cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        apply_manual_rotation=True,
        profile_info_extra={
            "target_size": [TARGET_W, TARGET_H],
            "jpeg_quality": JPEG_QUALITY,
        },
    )


def build_quality_cache(
    episode_dir: Path,
    frame_ids: list[int],
    profile: str = PROFILE_QUALITY,
) -> dict:
    def _transform(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if w > QUALITY_MAX_WIDTH:
            scale = QUALITY_MAX_WIDTH / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return gray

    return _build_transformed_cache(
        episode_dir,
        frame_ids=frame_ids,
        profile=profile,
        transform=_transform,
        suffix=".png",
        imwrite_params=[cv2.IMWRITE_PNG_COMPRESSION, 3],
        apply_manual_rotation=False,
        profile_info_extra={
            "max_width": QUALITY_MAX_WIDTH,
            "colorspace": "gray",
            "format": "png",
        },
    )


def load_cached_quality_frames(
    episode_dir: Path,
    frame_ids: list[int],
    profile: str = PROFILE_QUALITY,
) -> tuple[list[np.ndarray], list[int], dict] | None:
    manifest = load_manifest(episode_dir)
    profile_info = (manifest or {}).get("profiles", {}).get(profile)
    if not profile_info:
        return None

    prof_dir = profile_dir(episode_dir, profile)
    frames_gray: list[np.ndarray] = []
    cached_ids: list[int] = []
    cached_size = profile_info.get("cached_size", [0, 0])

    for fid in sorted(set(frame_ids)):
        path = prof_dir / f"frame_{fid:06d}.png"
        if not path.exists():
            return None
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        frames_gray.append(gray)
        cached_ids.append(fid)

    meta = {
        "video_path": str(resolve_episode_video_path(episode_dir)),
        "fps": manifest["fps"],
        "total_frames": manifest["total_frames"],
        "width": cached_size[0] if len(cached_size) >= 1 else 0,
        "height": cached_size[1] if len(cached_size) >= 2 else 0,
        "decoder": "frame_cache",
    }
    return frames_gray, cached_ids, meta


def cleanup_cache(episode_dir: Path) -> None:
    root = cache_root(episode_dir)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
        log.info(f"Removed temporary frame cache: {root}")
