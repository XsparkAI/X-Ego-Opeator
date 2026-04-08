"""Episode-level sparse frame cache shared by downstream operators."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..operator_base import OperatorResult
from ..video_path import resolve_episode_video_path
from ..vlm_limit import cpu_task_slot
from .cache_utils import (
    CaptionSamplingSpec,
    build_cache,
    build_caption_frame_ids,
    build_quality_cache,
    build_sample_fps_frame_ids,
    build_stride_frame_ids,
    probe_video,
)

log = logging.getLogger(__name__)


@dataclass
class FrameCacheConfig:
    include_caption: bool = False
    caption_window_sec: float = 10.0
    caption_step_sec: float = 5.0
    caption_frames_per_window: int = 12
    include_hand_vlm: bool = False
    hand_frame_step: int = 120
    include_video_quality: bool = False
    quality_sample_fps: float | None = None


class FrameCacheOperator:
    name = "frame_cache"

    def __init__(self, config: FrameCacheConfig | None = None):
        self.config = config or FrameCacheConfig()

    def run(self, episode_dir: Path, **kwargs) -> OperatorResult:
        with cpu_task_slot():
            video_path = resolve_episode_video_path(episode_dir)
            if not video_path.exists():
                return OperatorResult(
                    status="error",
                    operator=self.name,
                    errors=[f"Video not found: {video_path}"],
                )

            fps, total_frames = probe_video(video_path)
            wanted: set[int] = set()

            if self.config.include_caption:
                wanted.update(
                    build_caption_frame_ids(
                        fps,
                        total_frames,
                        CaptionSamplingSpec(
                            window_sec=self.config.caption_window_sec,
                            step_sec=self.config.caption_step_sec,
                            frames_per_window=self.config.caption_frames_per_window,
                        ),
                    )
                )

            if self.config.include_hand_vlm:
                wanted.update(build_stride_frame_ids(total_frames, self.config.hand_frame_step))

            quality_cached = 0
            if self.config.include_video_quality and self.config.quality_sample_fps is not None:
                quality_frame_ids = build_sample_fps_frame_ids(
                    fps,
                    total_frames,
                    self.config.quality_sample_fps,
                )
                q_manifest = build_quality_cache(episode_dir, quality_frame_ids)
                quality_cached = q_manifest["profiles"]["quality_gray_w1920_png"]["frame_count"]

            # VLM-style consumers share one RGB cache; quality keeps a separate grayscale profile.
            if not wanted:
                cached = 0
            else:
                manifest = build_cache(episode_dir, sorted(wanted))
                cached = manifest["profiles"]["vlm_640x480_q85"]["frame_count"]

            if cached == 0 and quality_cached == 0:
                return OperatorResult(
                    status="skipped",
                    operator=self.name,
                    errors=["No precomputable frame requests for this episode"],
                )

            log.info(
                f"frame_cache: cached vlm={cached}, quality={quality_cached} frames for {episode_dir.name}"
            )
            return OperatorResult(
                status="ok",
                operator=self.name,
                output_files=[str(episode_dir / ".frame_cache")],
                metrics={
                    "cached_vlm_frame_count": cached,
                    "cached_quality_frame_count": quality_cached,
                    "fps": round(fps, 4),
                    "total_frames": total_frames,
                },
            )
