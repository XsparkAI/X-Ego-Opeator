"""Pipeline adapter for privacy blur inference and optional segment merge."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from ..operator_base import OperatorResult
from ..video_path import resolve_episode_video_path

log = logging.getLogger(__name__)


def _episode_root(work_dir: Path) -> Path:
    if work_dir.parent.name == "segments":
        return work_dir.parent.parent
    return work_dir


def _concat_blurred_to_root(episode_root: Path) -> None:
    """Concatenate all seg_*/rgb_blurred.mp4 into episode_root/rgb_blurred.mp4.

    Uses ffmpeg concat demuxer (stream-copy, no re-encode).
    Only runs after every active segment already has its blurred output.
    """
    segments_dir = episode_root / "segments"
    if not segments_dir.exists():
        return

    seg_dirs = sorted(segments_dir.iterdir())
    # Only include segments produced in the current episode fan-out.
    active_seg_dirs = [d for d in seg_dirs if d.is_dir() and (d / "rgb.mp4").exists()]
    blurred_clips = [d / "rgb_blurred.mp4" for d in active_seg_dirs]

    # Another segment invocation will retry until the full set is ready.
    if not all(p.exists() and p.stat().st_size > 1024 for p in blurred_clips):
        return

    output_path = episode_root / "rgb_blurred.mp4"
    if output_path.exists():
        return

    if len(blurred_clips) == 1:
        # Single segment fan-out does not need concat.
        import shutil
        shutil.copy2(str(blurred_clips[0]), str(output_path))
        log.info(f"Copied single segment blurred video to {output_path}")
        return

    # Write concat list
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for clip in blurred_clips:
            f.write(f"file '{clip.resolve()}'\n")
        concat_list = f.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, timeout=120, check=True)
        log.info(f"Concatenated {len(blurred_clips)} segments → {output_path} "
                 f"({output_path.stat().st_size // 1024 // 1024} MB)")
    except Exception as e:
        log.warning(f"concat blurred video failed (non-fatal): {e}")
    finally:
        Path(concat_list).unlink(missing_ok=True)


@dataclass
class PrivacyBlurConfig:
    detector_backend: str = "egoblur"
    blur_targets: str = "both"  # "face" | "lp" | "both"
    detection_mode: str = "sampling_expand"  # "sampling_expand" | "legacy_per_frame"
    frame_sampling_step: int = 1
    use_frame_cache: bool = True
    frame_cache_num_workers: int = 1
    face: bool = True
    lp: bool = True
    scale: float = 1.0
    face_thresh: float | None = None
    lp_thresh: float | None = None
    yolo_face_model_path: str | None = None
    yolo_lp_model_path: str | None = None
    yolo_conf_thresh: float = 0.25
    yolo_input_size: int | None = 960
    max_concurrency: int = 1


class PrivacyBlurOperator:
    name = "privacy_blur"

    def __init__(self, config: PrivacyBlurConfig | None = None):
        self.config = config or PrivacyBlurConfig()
        self._detectors: tuple | None = None
        self._semaphore = threading.BoundedSemaphore(
            max(1, self.config.max_concurrency)
        )

    def _ensure_detectors(self):
        """Load face/LP detectors once, reuse across episodes."""
        if self._detectors is None:
            from .blur_privacy import (
                _build_detectors,
                _build_yolo_detectors,
                _get_device,
                _resolve_blur_flags,
            )

            face_enabled, lp_enabled = _resolve_blur_flags(
                self.config.blur_targets,
                face=self.config.face,
                lp=self.config.lp,
            )
            backend = self.config.detector_backend.lower()
            if backend == "yolo" and self.config.use_frame_cache:
                # frame_cache mode loads thread-local YOLO models inside blur_video.
                self._detectors = None
                return
            if backend == "egoblur":
                device = _get_device()
                self._detectors = _build_detectors(
                    device,
                    face=face_enabled,
                    lp=lp_enabled,
                    face_thresh=self.config.face_thresh,
                    lp_thresh=self.config.lp_thresh,
                )
            elif backend == "yolo":
                self._detectors = _build_yolo_detectors(
                    face_enabled=face_enabled,
                    lp_enabled=lp_enabled,
                    face_model_path=self.config.yolo_face_model_path,
                    lp_model_path=self.config.yolo_lp_model_path,
                )
            else:
                raise ValueError(
                    "privacy_blur.detector_backend must be 'egoblur' or 'yolo', "
                    f"got {self.config.detector_backend!r}"
                )
            log.info("Privacy blur detectors loaded (will reuse across episodes)")

    def run(self, episode_dir: Path, **kwargs) -> OperatorResult:
        from .blur_privacy import blur_video

        video_path = resolve_episode_video_path(episode_dir)
        output_path = episode_dir / "rgb_blurred.mp4"

        if not video_path.exists():
            return OperatorResult(
                status="error", operator=self.name,
                errors=[f"Video not found: {video_path}"],
            )

        with self._semaphore:
            self._ensure_detectors()

            try:
                summary = blur_video(
                    video_path, output_path,
                    detector_backend=self.config.detector_backend,
                    blur_targets=self.config.blur_targets,
                    detection_mode=self.config.detection_mode,
                    frame_sampling_step=self.config.frame_sampling_step,
                    use_frame_cache=self.config.use_frame_cache,
                    frame_cache_num_workers=self.config.frame_cache_num_workers,
                    face=self.config.face, lp=self.config.lp,
                    scale=self.config.scale,
                    face_thresh=self.config.face_thresh,
                    lp_thresh=self.config.lp_thresh,
                    yolo_face_model_path=self.config.yolo_face_model_path,
                    yolo_lp_model_path=self.config.yolo_lp_model_path,
                    yolo_conf_thresh=self.config.yolo_conf_thresh,
                    yolo_input_size=self.config.yolo_input_size,
                    detectors=self._detectors,
                )
                summary_path = episode_dir / "privacy_blur_summary.json"
                summary_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                # Segment-level runs opportunistically assemble an episode-level blurred video.
                ep_root = _episode_root(episode_dir)
                if ep_root != episode_dir:
                    try:
                        _concat_blurred_to_root(ep_root)
                    except Exception as e:
                        log.warning(f"concat to episode root failed (non-fatal): {e}")

                return OperatorResult(
                    status="ok", operator=self.name,
                    output_files=[str(output_path), str(summary_path)],
                    metrics=summary,
                )
            except Exception as e:
                log.exception("privacy_blur failed")
                return OperatorResult(
                    status="error", operator=self.name,
                    errors=[str(e)],
                )
