"""Adapter wrapping process_video() into the Operator protocol."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ..operator_base import OperatorResult

log = logging.getLogger(__name__)


@dataclass
class VideoQualityConfig:
    sample_fps: float | None = None


class VideoQualityOperator:
    name = "video_quality"

    def __init__(self, config: VideoQualityConfig | None = None):
        self.config = config or VideoQualityConfig()

    def run(self, episode_dir: Path, **kwargs) -> OperatorResult:
        from .assess import process_video

        video_path = episode_dir / "rgb.mp4"
        output_path = episode_dir / "quality_report.json"

        if not video_path.exists():
            return OperatorResult(
                status="error", operator=self.name,
                errors=[f"Video not found: {video_path}"],
            )

        try:
            result = process_video(
                str(video_path), sample_fps=self.config.sample_fps
            )
            output_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return OperatorResult(
                status="ok", operator=self.name,
                output_files=[str(output_path)],
                metrics={
                    "mean_laplacian": result.get("quality", {}).get("mean_laplacian"),
                    "translation_std": result.get("stability", {}).get("translation_std"),
                    "blur_ratio": result.get("quality", {}).get("blur_ratio"),
                },
            )
        except Exception as e:
            log.exception("video_quality failed")
            return OperatorResult(
                status="error", operator=self.name,
                errors=[str(e)],
            )
