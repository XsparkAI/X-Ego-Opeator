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
        from . import segment_v2t as seg_module
        from .segment_v2t import (
            process_episode, get_task_type, TASK_SOP_MAP, SOP_DIR,
        )

        # Apply config to module globals (matching existing CLI pattern)
        seg_module.WINDOW_SEC = self.config.window_sec
        seg_module.STEP_SEC = self.config.step_sec
        seg_module.FRAMES_PER_WINDOW = self.config.frames_per_window
        seg_module.SNAP_RADIUS_FRAMES = self.config.snap_radius

        # Resolve SOP: try name-prefix lookup, fall back to None (no-SOP mode)
        task_type = get_task_type(episode_dir.name)
        sop = None
        if task_type:
            sop_path = SOP_DIR / TASK_SOP_MAP[task_type]
            if sop_path.exists():
                sop = json.loads(sop_path.read_text(encoding="utf-8"))

        try:
            caption = process_episode(
                episode_dir, sop,
                preview=self.config.preview,
                dry_run=self.config.dry_run,
            )
            output_path = episode_dir / "caption_v2t.json"
            metrics = {}
            if caption:
                segments = caption.get("atomic_action", caption.get("segments", []))
                metrics["num_segments"] = len(segments)
                if "token_stats" in caption:
                    metrics["token_stats"] = caption["token_stats"]

            return OperatorResult(
                status="ok", operator=self.name,
                output_files=[str(output_path)],
                metrics=metrics,
            )
        except Exception as e:
            log.exception("video_segmentation failed")
            return OperatorResult(
                status="error", operator=self.name,
                errors=[str(e)],
            )
