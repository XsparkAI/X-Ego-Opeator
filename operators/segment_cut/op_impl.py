"""Segment cut operator — split rgb.mp4 into per-segment clips.

Reads ``caption_v2t.json`` (produced by video_segmentation) and extracts
each atomic_action as a short video clip.

We intentionally use accurate seek + re-encode instead of stream-copy.
With HEVC/H.264 inputs, stream-copy cutting from non-keyframe boundaries
can pull earlier GOP content into the segment, which creates overlap
between adjacent clips and later shows up as repeated content after
episode-level concatenation.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..operator_base import OperatorResult

log = logging.getLogger(__name__)

_MIN_FILE_BYTES = 1024  # skip re-cut if file already > 1 KB


@dataclass
class SegmentCutConfig:
    min_duration_sec: float = 0.5  # skip segments shorter than this


class SegmentCutOperator:
    name = "segment_cut"

    def __init__(self, config: SegmentCutConfig | None = None):
        self.config = config or SegmentCutConfig()

    def run(self, episode_dir: Path, **kwargs: Any) -> OperatorResult:
        caption_path = episode_dir / "caption_v2t.json"
        video_path = episode_dir / "rgb.mp4"

        if not caption_path.exists():
            return OperatorResult(
                status="error", operator=self.name,
                errors=[f"caption_v2t.json not found in {episode_dir}"],
            )
        if not video_path.exists():
            return OperatorResult(
                status="error", operator=self.name,
                errors=[f"rgb.mp4 not found in {episode_dir}"],
            )

        caption = json.loads(caption_path.read_text(encoding="utf-8"))
        segments = caption.get("atomic_action", [])
        fps = caption.get("fps")

        # If fps not in caption, probe it
        if fps is None:
            fps = self._probe_fps(video_path)
        if fps is None or fps <= 0:
            return OperatorResult(
                status="error", operator=self.name,
                errors=["Cannot determine video fps"],
            )

        segments_root = episode_dir / "segments"
        segments_root.mkdir(exist_ok=True)

        segment_dirs: list[str] = []
        cut_count = 0
        skip_count = 0

        for i, seg in enumerate(segments):
            fi = seg.get("frame_interval", [])
            if len(fi) < 2:
                continue

            start_frame, end_frame = fi[0], fi[1]
            start_sec = start_frame / fps
            duration_sec = (end_frame - start_frame) / fps

            if duration_sec < self.config.min_duration_sec:
                skip_count += 1
                continue

            seg_dir = segments_root / f"seg_{i:03d}"
            seg_dir.mkdir(exist_ok=True)
            seg_video = seg_dir / "rgb.mp4"

            # Idempotent: skip if already cut
            if seg_video.exists() and seg_video.stat().st_size > _MIN_FILE_BYTES:
                segment_dirs.append(str(seg_dir))
                self._write_info(seg_dir, seg, fps)
                continue

            # Accurate cut with re-encode. Stream-copy from non-keyframes can
            # include earlier GOP frames, causing overlaps between segments.
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-ss", f"{start_sec:.6f}",
                "-t", f"{duration_sec:.6f}",
                "-map", "0:v:0",
                "-map", "0:a?",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                str(seg_video),
            ]
            try:
                subprocess.run(
                    cmd, capture_output=True, timeout=60, check=True,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                log.warning(f"ffmpeg cut failed for seg_{i:03d}: {e}")
                continue

            if seg_video.exists() and seg_video.stat().st_size > _MIN_FILE_BYTES:
                segment_dirs.append(str(seg_dir))
                self._write_info(seg_dir, seg, fps)
                cut_count += 1
            else:
                log.warning(f"seg_{i:03d} output too small, skipping")

        log.info(
            f"segment_cut: {len(segment_dirs)} segments "
            f"(cut={cut_count}, cached={len(segment_dirs) - cut_count}, "
            f"skipped={skip_count})"
        )

        return OperatorResult(
            status="ok",
            operator=self.name,
            output_files=[str(segments_root)],
            metrics={
                "segment_dirs": segment_dirs,
                "total_segments": len(segment_dirs),
                "cut_count": cut_count,
                "skipped_short": skip_count,
            },
        )

    @staticmethod
    def _write_info(seg_dir: Path, seg: dict, fps: float) -> None:
        fi = seg["frame_interval"]
        info = {
            "start_frame": fi[0],
            "end_frame": fi[1],
            "fps": fps,
            "duration_sec": round((fi[1] - fi[0]) / fps, 4),
            "instruction": seg.get("instruction", ""),
        }
        if "sop_step_index" in seg:
            info["sop_step_index"] = seg["sop_step_index"]
        (seg_dir / "segment_info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8",
        )

    @staticmethod
    def _probe_fps(video_path: Path) -> float | None:
        try:
            out = subprocess.check_output(
                [
                    "ffprobe", "-v", "quiet",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=r_frame_rate",
                    "-of", "csv=p=0",
                    str(video_path),
                ],
                text=True, timeout=10,
            )
            num, den = out.strip().split("/")
            return float(num) / float(den)
        except Exception:
            return None
