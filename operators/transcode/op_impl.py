"""Adapter wrapping transcode plan+execute into the Operator protocol."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..operator_base import OperatorResult

log = logging.getLogger(__name__)


@dataclass
class TranscodeConfig:
    codec: str | None = None
    container: str | None = None
    resolution: str | None = None
    bitrate: str | None = None
    fps: float | None = None
    pix_fmt: str | None = None
    output_suffix: str = "_transcoded"


class TranscodeOperator:
    name = "transcode"

    def __init__(self, config: TranscodeConfig | None = None):
        self.config = config or TranscodeConfig()

    def run(self, episode_dir: Path, **kwargs) -> OperatorResult:
        from .transcode import TranscodeSpec, plan_transcode, execute

        # Prefer blurred video if it exists (pipeline ordering)
        blurred = episode_dir / "rgb_blurred.mp4"
        raw = episode_dir / "rgb.mp4"
        input_path = blurred if blurred.exists() else raw

        if not input_path.exists():
            return OperatorResult(
                status="error", operator=self.name,
                errors=[f"No video found in {episode_dir}"],
            )

        ext = f".{self.config.container}" if self.config.container else input_path.suffix
        output_path = episode_dir / f"{input_path.stem}{self.config.output_suffix}{ext}"

        spec = TranscodeSpec(
            codec=self.config.codec,
            container=self.config.container,
            resolution=self.config.resolution,
            bitrate=self.config.bitrate,
            fps=self.config.fps,
            pix_fmt=self.config.pix_fmt,
        )

        try:
            plan = plan_transcode(str(input_path), str(output_path), spec)
            result = execute(plan)
            return OperatorResult(
                status=result.get("status", "ok"), operator=self.name,
                output_files=[str(output_path)],
                metrics=result,
            )
        except Exception as e:
            log.exception("transcode failed")
            return OperatorResult(
                status="error", operator=self.name,
                errors=[str(e)],
            )
