"""Pipeline adapter for planned video transcoding."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..operator_base import OperatorResult
from ..video_path import resolve_episode_video_path
from ..vlm_limit import cpu_task_slot

log = logging.getLogger(__name__)


def _codec_key_for(codec_name: str) -> str | None:
    normalized = str(codec_name).lower()
    if normalized == "h264":
        return "h264"
    if normalized in {"hevc", "h265"}:
        return "h265"
    if normalized == "ffv1":
        return "ffv1"
    if normalized == "vp9":
        return "vp9"
    if normalized == "av1":
        return "av1"
    if normalized.startswith("prores"):
        return "prores"
    return None


def _remux_original_audio(video_path: Path, original_path: Path) -> dict | None:
    from .transcode import probe

    video_info = probe(str(video_path))
    original_info = probe(str(original_path))
    if video_info.audio_codec or not original_info.audio_codec:
        return None

    tmp_out = video_path.with_name(f"{video_path.stem}_audio_mux{video_path.suffix}")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(original_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        str(tmp_out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    tmp_out.replace(video_path)
    remuxed = probe(str(video_path))
    return {
        "audio_codec": remuxed.audio_codec,
        "audio_remuxed_from": str(original_path),
    }


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
        with cpu_task_slot():
            from .transcode import TranscodeSpec, execute, plan_transcode, probe

            blurred = episode_dir / "rgb_blurred.mp4"
            raw = resolve_episode_video_path(episode_dir)
            original = resolve_episode_video_path(episode_dir)
            # Prefer the privacy-filtered video when it exists.
            input_path = blurred if blurred.exists() else raw

            if not input_path.exists():
                return OperatorResult(
                    status="error", operator=self.name,
                    errors=[f"No video found in {episode_dir}"],
                )

            ext = f".{self.config.container}" if self.config.container else input_path.suffix
            output_path = episode_dir / f"{input_path.stem}{self.config.output_suffix}{ext}"

            original_info = probe(str(original))
            effective_codec = self.config.codec
            effective_container = self.config.container
            effective_resolution = self.config.resolution
            effective_bitrate = self.config.bitrate
            effective_fps = self.config.fps
            effective_pix_fmt = self.config.pix_fmt

            if input_path != original:
                effective_codec = effective_codec or _codec_key_for(original_info.codec)
                effective_container = effective_container or original.suffix.lstrip(".").lower()
                effective_resolution = effective_resolution or f"{original_info.width}x{original_info.height}"
                effective_fps = effective_fps or original_info.fps
                effective_pix_fmt = effective_pix_fmt or original_info.pix_fmt
                if effective_bitrate is None and original_info.bitrate_kbps > 0:
                    effective_bitrate = f"{original_info.bitrate_kbps}k"

            spec = TranscodeSpec(
                codec=effective_codec,
                container=effective_container,
                resolution=effective_resolution,
                bitrate=effective_bitrate,
                fps=effective_fps,
                pix_fmt=effective_pix_fmt,
            )

            try:
                plan = plan_transcode(str(input_path), str(output_path), spec)
                result = execute(plan)
                if result.get("status") == "ok":
                    remux_meta = _remux_original_audio(output_path, original)
                    if remux_meta:
                        result["audio_remux"] = remux_meta
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
