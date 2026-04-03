"""Adapter wrapping detect_hands_in_video() into the Operator protocol."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ..operator_base import OperatorResult

log = logging.getLogger(__name__)


@dataclass
class HandDetectionConfig:
    conf_thresh: float = 0.3
    frame_step: int = 1
    input_height: int = 720


class HandDetectionOperator:
    name = "hand_detection"

    def __init__(self, config: HandDetectionConfig | None = None):
        self.config = config or HandDetectionConfig()
        self._model = None

    def _ensure_model(self):
        """Load YOLO model once, reuse across episodes."""
        if self._model is None:
            from .detect_hand_in_frame import load_model
            self._model = load_model()
            log.info("Hand detection model loaded (will reuse across episodes)")

    def run(self, episode_dir: Path, **kwargs) -> OperatorResult:
        from .detect_hand_in_frame import detect_hands_in_video

        video_path = episode_dir / "rgb.mp4"
        output_path = episode_dir / "hand_detection.json"

        if not video_path.exists():
            return OperatorResult(
                status="error", operator=self.name,
                errors=[f"Video not found: {video_path}"],
            )

        self._ensure_model()

        try:
            result = detect_hands_in_video(
                video_path,
                conf_thresh=self.config.conf_thresh,
                frame_step=self.config.frame_step,
                input_height=self.config.input_height,
                model=self._model,
            )
            output_path.write_text(
                json.dumps(result["summary"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return OperatorResult(
                status="ok", operator=self.name,
                output_files=[str(output_path)],
                metrics=result["summary"],
            )
        except Exception as e:
            log.exception("hand_detection failed")
            return OperatorResult(
                status="error", operator=self.name,
                errors=[str(e)],
            )
