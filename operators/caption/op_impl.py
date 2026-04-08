"""Adapter wrapping process_episode() into the Operator protocol."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ..operator_base import OperatorResult

log = logging.getLogger(__name__)


@dataclass
class SegmentationConfig:
    window_sec: float = 10.0
    step_sec: float = 5.0
    frames_per_window: int = 12
    snap_radius: int = 45
    preview: bool = False
    dry_run: bool = False


class SegmentationOperator:
    name = "video_segmentation"

    def __init__(self, config: SegmentationConfig | None = None):
        self.config = config or SegmentationConfig()

    def run(self, episode_dir: Path, **kwargs) -> OperatorResult:
        from .segment_v2t import segment

        video_path = episode_dir / "rgb.mp4"
        output_path = episode_dir / "caption_v2t.json"

        if not video_path.exists():
            return OperatorResult(
                status="error", operator=self.name,
                errors=[f"Video not found: {video_path}"],
            )

        try:
            caption = segment(
                video_path,
                window_sec=self.config.window_sec,
                step_sec=self.config.step_sec,
                frames_per_window=self.config.frames_per_window,
            )

            output_path.write_text(
                json.dumps(caption, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            segments = caption.get("atomic_action", [])
            metrics = {"num_segments": len(segments)}
            output_files = [str(output_path)]

            # Generate caption preview video (full video + subtitle overlay)
            try:
                from .segment_v2t import generate_preview
                preview_path = generate_preview(video_path, caption, caption["fps"])
                output_files.append(str(preview_path))
                log.info(f"Caption preview saved: {preview_path}")
            except Exception as e:
                log.warning(f"caption preview failed (non-fatal): {e}")

            return OperatorResult(
                status="ok", operator=self.name,
                output_files=output_files,
                metrics=metrics,
            )
        except Exception as e:
            log.exception("video_segmentation failed")
            return OperatorResult(
                status="error", operator=self.name,
                errors=[str(e)],
            )
