"""Adapter wrapping blur_video() into the Operator protocol."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..operator_base import OperatorResult

log = logging.getLogger(__name__)


@dataclass
class PrivacyBlurConfig:
    face: bool = True
    lp: bool = True
    scale: float = 1.0
    face_thresh: float | None = None
    lp_thresh: float | None = None


class PrivacyBlurOperator:
    name = "privacy_blur"

    def __init__(self, config: PrivacyBlurConfig | None = None):
        self.config = config or PrivacyBlurConfig()
        self._detectors: tuple | None = None

    def _ensure_detectors(self):
        """Load face/LP detectors once, reuse across episodes."""
        if self._detectors is None:
            from .blur_privacy import _build_detectors, _get_device
            device = _get_device()
            self._detectors = _build_detectors(
                device,
                face=self.config.face, lp=self.config.lp,
                face_thresh=self.config.face_thresh,
                lp_thresh=self.config.lp_thresh,
            )
            log.info("Privacy blur detectors loaded (will reuse across episodes)")

    def run(self, episode_dir: Path, **kwargs) -> OperatorResult:
        from .blur_privacy import blur_video

        video_path = episode_dir / "rgb.mp4"
        output_path = episode_dir / "rgb_blurred.mp4"

        if not video_path.exists():
            return OperatorResult(
                status="error", operator=self.name,
                errors=[f"Video not found: {video_path}"],
            )

        self._ensure_detectors()

        try:
            summary = blur_video(
                video_path, output_path,
                face=self.config.face, lp=self.config.lp,
                scale=self.config.scale,
                face_thresh=self.config.face_thresh,
                lp_thresh=self.config.lp_thresh,
                detectors=self._detectors,
            )
            return OperatorResult(
                status="ok", operator=self.name,
                output_files=[str(output_path)],
                metrics=summary,
            )
        except Exception as e:
            log.exception("privacy_blur failed")
            return OperatorResult(
                status="error", operator=self.name,
                errors=[str(e)],
            )
