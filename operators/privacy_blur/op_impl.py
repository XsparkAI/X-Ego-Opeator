"""Adapter wrapping blur_video() into the Operator protocol."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..operator_base import OperatorResult

log = logging.getLogger(__name__)


def _episode_root(work_dir: Path) -> Path:
    if work_dir.parent.name == "segments":
        return work_dir.parent.parent
    return work_dir


def _concat_blurred_to_root(episode_root: Path) -> None:
    """Concatenate all seg_*/rgb_blurred.mp4 into episode_root/rgb_blurred.mp4.

    Uses ffmpeg concat demuxer (stream-copy, no re-encode).
    Only runs when called from the last segment — determined by checking
    whether all segment dirs already have rgb_blurred.mp4.
    """
    segments_dir = episode_root / "segments"
    if not segments_dir.exists():
        return

    seg_dirs = sorted(segments_dir.iterdir())
    # Only consider segments that have a source rgb.mp4 (current run's segments)
    active_seg_dirs = [d for d in seg_dirs if d.is_dir() and (d / "rgb.mp4").exists()]
    blurred_clips = [d / "rgb_blurred.mp4" for d in active_seg_dirs]

    # Wait until all active segments have been processed
    if not all(p.exists() and p.stat().st_size > 1024 for p in blurred_clips):
        return  # Not all done yet — another segment will trigger this later

    output_path = episode_root / "rgb_blurred.mp4"
    if output_path.exists():
        return  # Already concatenated

    if len(blurred_clips) == 1:
        # Single segment: just copy
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
            report_path = episode_dir / "blur_report.json"
            report_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

            # After segment blur completes, try to concat all segments → episode root
            ep_root = _episode_root(episode_dir)
            if ep_root != episode_dir:
                try:
                    _concat_blurred_to_root(ep_root)
                except Exception as e:
                    log.warning(f"concat to episode root failed (non-fatal): {e}")

            return OperatorResult(
                status="ok", operator=self.name,
                output_files=[str(output_path), str(report_path)],
                metrics=summary,
            )
        except Exception as e:
            log.exception("privacy_blur failed")
            return OperatorResult(
                status="error", operator=self.name,
                errors=[str(e)],
            )
